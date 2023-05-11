import os
import torch
import pandas as pd
from src.experiment.params_optimizer import RandomSearchLoader
from src.models.gain import GAIN
from src.data.datasets import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class RandomSearchExperiment:
    def __init__(self, model, dataset, iterations, results_path, accelerator='gpu',
                 max_iter_train=10000, gpu='auto'):

        self.results_path = f'{results_path}'
        self.selected_gpu = gpu

        os.makedirs(results_path, exist_ok=True)

        if os.path.exists(f'{results_path}/{model}_results.csv'):
            self.results_file = pd.read_csv(f'{results_path}/{model}_results.csv', index_col='Unnamed: 0')

        else:
            columns = [
                'mse',
                'mae',
                'rmse',
                'denorm_mse',
                'denorm_mae',
                'denorm_rmse',
                'denorm_mre',
                'params'
            ]
            self.results_file = pd.DataFrame(columns=columns)

        self.model = model
        self.iterations = iterations
        self.dataset = dataset
        self.params_loader = RandomSearchLoader(model, iterations)
        self.accelerator = accelerator
        self.max_iter_train = max_iter_train

    def prepare_data(self, batch_size):
        dm = DataModule(dataset=self.dataset, batch_size=batch_size)
        edge_index, edge_weights = dm.get_connectivity()
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).cuda()
            edge_weights = torch.from_numpy(edge_weights).cuda()

        return dm, edge_index, edge_weights, normalizer

    def train_test(self, dm, edge_index, edge_weights, normalizer, hyperparameters):
        model = GAIN(
            model_type=self.model,
            input_size=dm.input_size(),
            edge_index=edge_index,
            edge_weights=edge_weights,
            normalizer=normalizer,
            params=hyperparameters,
        )

        trainer = Trainer(
            max_steps=self.max_iter_train,
            default_root_dir='reports/logs_experiments',
            accelerator=self.accelerator,
            devices=self.selected_gpu,
            callbacks=[EarlyStopping(monitor='mse', mode='min', patience=1000)],
        )

        trainer.fit(model, datamodule=dm)
        results = trainer.test(model, datamodule=dm)[0]

        return results

    def save_results(self, results, params):
        row = [
            results['mse'],
            results['mae'],
            results['rmse'],
            results['denorm_mse'],
            results['denorm_mae'],
            results['denorm_rmse'],
            results['denorm_mre'],
            params,
        ]

        self.results_file.loc[self.results_file.shape[0]] = row
        self.results_file.to_csv(f'{self.results_path}/{self.model}_results.csv')

    def run(self):

        for i in range(self.results_file.shape[0], self.iterations):
            hyperparameters = self.params_loader.get_params(i)
            dm, edge_index, edge_weights, normalizer = self.prepare_data(hyperparameters['batch_size'][0])
            results = self.train_test(dm, edge_index, edge_weights, normalizer, hyperparameters)
            self.save_results(results, hyperparameters)
