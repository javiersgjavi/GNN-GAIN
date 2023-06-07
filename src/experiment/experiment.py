import os
import json
import torch
import itertools
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
                 max_iter_train=5000, gpu='auto', bi=False, time_gap=False):

        self.columns = [
            'mse',
            'mae',
            'rmse',
            'denorm_mse',
            'denorm_mae',
            'denorm_rmse',
            'denorm_mre',
            'params'
        ]

        self.bi = bi
        self.model_name = model
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

    def load_file(self):

        os.makedirs(self.results_path, exist_ok=True)

        if os.path.exists(f'{self.results_path}/{self.model_name}_results.csv'):
            results_file = pd.read_csv(f'{self.results_path}/{self.model_name}_results.csv', index_col='Unnamed: 0')

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

    def train_test(self, hyperparameters):
        hyperparameters['use_time_gap_matrix'] = self.time_gap
        hyperparameters['bi'] = self.bi
        model = GAIN(
            model_type=self.model_name,
            input_size=self.dm.input_size(),
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

        results = trainer.test(model, datamodule=self.dm)[0]
        return results

    def save_results_file(self, results, params):
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
        self.results_file.to_csv(f'{self.results_path}/{self.model_name}_results.csv')

    def run(self):
        for i in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'Random Search with {self.model_name} in {self.dataset}'):
            hyperparameters = self.params_loader.get_params(i)
            print_dict(hyperparameters, self.max_iter_train)
            results = self.train_test(hyperparameters)
            self.save_results_file(results, hyperparameters)


class RandomSearch:

    def __init__(self, models=None, datasets=None, iterations=100, gpu='auto', max_iter_train=5000, bi=None,
                 time_gap=None, folder='results'):

        self.models = models
        self.datasets = datasets
        self.iterations = iterations

        self.folder = folder
        self.gpu = gpu
        self.max_iter_train = max_iter_train
        self.bi = bi
        self.time_gap = time_gap

    def make_summary_dataset(self, datasets, models):
        columns = [
            'model',
            'mse',
            'mae',
            'rmse',
            'denorm_mse',
            'denorm_mae',
            'denorm_rmse',
            'denorm_mre',
            'params'
        ]

        for dataset in datasets:
            results_path = f'./{self.folder}/{dataset}/'
            result_file = pd.DataFrame(columns=columns)
            for model in models:
                results_model = pd.read_csv(f'{results_path}{model}_results.csv')
                best_result = results_model.iloc[results_model['mae'].idxmin()]
                row = [
                    model,
                    best_result['mse'],
                    best_result['mae'],
                    best_result['rmse'],
                    best_result['denorm_mse'],
                    best_result['denorm_mae'],
                    best_result['denorm_rmse'],
                    best_result['denorm_mre'],
                    best_result['params']
                ]
                result_file.loc[len(result_file)] = row

            result_file.to_csv(f'{results_path}/results.csv', index=False)

    def make_summary_general(self, datasets):
        columns = [
            'dataset',
            'model',
            'mae',
            'mse',
            'rmse',
            'denorm_mae',
            'denorm_mse',
            'denorm_mre',
            'denorm_rmse',
            'params'
        ]
        result_file = pd.DataFrame(columns=columns)

        for dataset in datasets:
            results_dataset_path = f'./{self.folder}/{dataset}/results.csv'
            results_dataset = pd.read_csv(results_dataset_path)

            best_result = results_dataset.iloc[results_dataset['mae'].idxmin()]
            row = [
                dataset,
                best_result['model'],
                best_result['mae'],
                best_result['mse'],
                best_result['rmse'],
                best_result['denorm_mae'],
                best_result['denorm_mse'],
                best_result['denorm_mre'],
                best_result['denorm_rmse'],
                best_result['params']
            ]
            result_file.loc[len(result_file)] = row

        result_file.to_csv(f'./{self.folder}/results.csv', index=False)

    def run(self):
        for dataset, model in itertools.product(self.datasets, self.models):
            results_path = f'./{self.folder}/{dataset}/'
            random_search = RandomSearchExperiment(
                model=model,
                dataset=dataset,
                iterations=self.iterations,
                results_path=results_path,
                gpu=self.gpu,
                max_iter_train=self.max_iter_train,
                bi=self.bi,
                time_gap=self.time_gap
            )

            random_search.run()
        self.make_summary_dataset(self.datasets, self.models)
        self.make_summary_general(self.datasets)
