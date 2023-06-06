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
                 max_iter_train=5000, gpu='auto', bi=False, time_gap=False, columns=None):

        self.bi = bi
        self.model = model
        self.columns = columns
        self.dataset = dataset
        self.selected_gpu = gpu
        self.time_gap = time_gap
        self.iterations = iterations
        self.accelerator = accelerator
        self.max_iter_train = max_iter_train
        self.params_loader = RandomSearchLoader(model, iterations, bi=self.bi)
        self.dm, self.edge_index, self.edge_weights, self.normalizer = self.prepare_data(
            self.params_loader.random_params['batch_size'][0][0])

        self.results_path = f'{results_path}'
        self.results_file = self.load_file()
        self.trainer = None
        self.dm = None

    def load_file(self):

        os.makedirs(results_path, exist_ok=True)

        if os.path.exists(f'{results_path}/{model}_results.csv'):
            results_file = pd.read_csv(f'{results_path}/{model}_results.csv', index_col='Unnamed: 0')

        else:
            results_file = pd.DataFrame(columns=self.columns)

        return results_file

    def prepare_data(self, batch_size):
        dm = DataModule(dataset=self.dataset, batch_size=batch_size, use_time_gap_matrix=self.time_gap)
        edge_index, edge_weights = dm.get_connectivity()
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        return dm, edge_index, edge_weights, normalizer

    def train(self, hyperparameters):
        hyperparameters['use_time_gap_matrix'] = self.time_gap
        hyperparameters['bi'] = self.bi
        model = GAIN(
            model_type=self.model,
            input_size=dm.input_size(),
            edge_index=self.edge_index,
            edge_weights=self.edge_weights,
            normalizer=self.normalizer,
            params=hyperparameters,
        )

        trainer = Trainer(
            max_steps=self.max_iter_train,
            default_root_dir='reports/logs_experiments',
            accelerator=self.accelerator,
            devices=self.selected_gpu,
            callbacks=[EarlyStopping(monitor='denorm_mse', patience=1, mode='min')],
        )

        trainer.fit(model, datamodule=self.dm)

        self.trainer = trainer

    def save_results_file(self, row):

        self.results_file.loc[self.results_file.shape[0]] = row
        self.results_file.to_csv(f'{self.results_path}/{self.model}_results.csv')

    def run(self):
        for i in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'Random Search with {self.model} in {self.dataset}'):
            hyperparameters = self.params_loader.get_params(i)
            print_dict(hyperparameters, self.max_iter_train)
            self.train(hyperparameters)
            results = self.test()
            self.save_results(results, hyperparameters)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

class BlockPointMissingExperiment(RandomSearchExperiment):
    def __init__(self, *args, **kwargs):
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
        super().__init__(columns=columns, *args, **kwargs)

    def test(self):
        return self.trainer.test(model, datamodule=self.dm)[0]

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

        super().save_results_file(row)


class InOutSampleExperiment(RandomSearchExperiment):
    def __init__(self, *args, **kwargs):
        columns = [
            'mae_in',
            'mse_in',
            'rmse_in',
            'denorm_mae_in',
            'denorm_mse_in',
            'denorm_mre_in',
            'denorm_rmse_in',
            'mae_out',
            'mse_out',
            'rmse_out',
            'denorm_mae_out',
            'denorm_mse_out',
            'denorm_mre_out',
            'denorm_rmse_out',
            'params'
        ]
        super().__init__(columns=columns, *args, **kwargs)

    def test(self):
        out_sample = self.trainer.test(model, datamodule=self.dm)[0]
        in_sample = self.trainer.test(model, dataloaders=self.dm.train_dataloader())[0]
        results = {'in': in_sample, 'out': out_sample}
        return results

    def save_results(self, results, params):
        row = [
            results['in']['mae'],
            results['in']['mse'],
            results['in']['rmse'],
            results['in']['denorm_mae'],
            results['in']['denorm_mse'],
            results['in']['denorm_mre'],
            results['in']['denorm_rmse'],

            results['out']['mae'],
            results['out']['mse'],
            results['out']['rmse'],
            results['out']['denorm_mae'],
            results['out']['denorm_mse'],
            results['out']['denorm_mre'],
            results['out']['denorm_rmse'],

            params,
        ]

        super().save_results_file(row)
