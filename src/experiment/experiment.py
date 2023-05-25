import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from src.experiment.params_optimizer import RandomSearchLoader
from src.models.gain import GAIN
from src.data.datasets import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def print_dict(dictionary, max_iter_train):
    for key, value in dictionary.items():
        print(f'{key}: {value}')
    print(f'Max steps training: {max_iter_train}')


class RandomSearchExperiment:
    def __init__(self, model, dataset, iterations, results_path, accelerator='gpu',
                 max_iter_train=5000, gpu='auto', bi=False):

        self.results_path = f'{results_path}'
        self.selected_gpu = gpu


        os.makedirs(results_path, exist_ok=True)

        if os.path.exists(f'{results_path}/{model}_results.csv'):
            self.results_file = pd.read_csv(f'{results_path}/{model}_results.csv', index_col='Unnamed: 0')

        else:
            columns = [
                'mae',
                'mse',
                'rmse',
                'denorm_mae',
                'denorm_mse',
                'denorm_mre',
                'denorm_rmse',
                'params'
            ]
            self.results_file = pd.DataFrame(columns=columns)

        self.bi = bi
        self.model = model
        self.dataset = dataset
        self.iterations = iterations
        self.params_loader = RandomSearchLoader(model, iterations, bi=self.bi)
        self.accelerator = accelerator
        self.max_iter_train = max_iter_train
        self.dm, self.edge_index, self.edge_weights, self.normalizer = self.prepare_data(
            self.params_loader.random_params['batch_size'][0][0])

    def prepare_data(self, batch_size):
        dm = DataModule(dataset=self.dataset, batch_size=batch_size, use_time_gap_matrix=True)
        edge_index, edge_weights = dm.get_connectivity()
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        return dm, edge_index, edge_weights, normalizer

    def train_test(self, dm, edge_index, edge_weights, normalizer, hyperparameters):
        hyperparameters['use_time_gap_matrix'] = True
        hyperparameters['bi'] = self.bi
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
            callbacks=[EarlyStopping(monitor='denorm_mse', patience=1, mode='min')],
        )

        trainer.fit(model, datamodule=dm)
        results = trainer.test(model, datamodule=dm)[0]

        return results

    def save_results(self, results, params):
        row = [
            results['mae'],
            results['mse'],
            results['rmse'],
            results['denorm_mae'],
            results['denorm_mse'],
            results['denorm_mre'],
            results['denorm_rmse'],
            params,
        ]

        self.results_file.loc[self.results_file.shape[0]] = row
        self.results_file.to_csv(f'{self.results_path}/{self.model}_results.csv')

    def run(self):

        for i in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'Random Search with {self.model} in {self.dataset}'):
            hyperparameters = self.params_loader.get_params(i)
            print_dict(hyperparameters, self.max_iter_train)
            results = self.train_test(self.dm, self.edge_index, self.edge_weights, self.normalizer, hyperparameters)
            self.save_results(results, hyperparameters)
