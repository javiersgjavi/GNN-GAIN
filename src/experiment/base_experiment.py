import os
import json
import time
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models.g_tigre import GTIGRE, GTIGRE_DYNAMIC
from src.data.traffic import MetrLADataset, PemsBayDataset
from src.data.mimic_iii import MIMICIIIDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR


def print_dict(dictionary, max_iter_train):
    for key, value in dictionary.items():
        print(f'{key}: {value}')
    print(f'Max steps training: {max_iter_train}')


class BaseExperiment:
    def __init__(self, model, dataset, iterations, results_path, accelerator='gpu', save_file=None,
                 max_iter_train=5000, gpu='auto', default_hyperparameters=None, time_gap=None):
        self.columns = [
            'mae',
            'mse',
            'rmse',
            'denorm_mae',
            'denorm_mse',
            'denorm_mre',
            'denorm_rmse',
            'time',
            'params'
        ]


        if default_hyperparameters is not None:
            default_hyperparameters = default_hyperparameters \
                .replace("'", '"') \
                .replace('True', 'true') \
                .replace('False', 'false')

            default_hyperparameters = json.loads(default_hyperparameters)

            self.time_gap = default_hyperparameters['use_time_gap_matrix']
            self.batch_size = default_hyperparameters['batch_size'][0] if 'batch_size' in default_hyperparameters.keys() else 64

        else:
            self.time_gap = time_gap

        self.model_name = model
        self.dataset = dataset
        self.selected_gpu = gpu
        self.iterations = iterations
        self.accelerator = accelerator
        self.max_iter_train = max_iter_train
        self.default_hyperparameters = default_hyperparameters
        self.results_path = results_path
        self.save_file = f'{results_path}/{save_file}.csv'
        self.results_file = self.load_file()
        self.model = None
        self.trainer = None
        self.exp_name = 'Basic experiment'

        self.dm, self.edge_index, self.edge_weights, self.normalizer = self.prepare_data()
        self.batch_size = self.dm.batch_size

    def load_file(self):

        os.makedirs(self.results_path, exist_ok=True)

        if os.path.exists(self.save_file):
            results_file = pd.read_csv(self.save_file, index_col='Unnamed: 0')

        else:
            results_file = pd.DataFrame(columns=self.columns)

        return results_file

    def prepare_data(self):
        name_dataset = self.dataset.split('_')[0]

        if name_dataset == 'la':
            dm = MetrLADataset(point=True)
        elif name_dataset == 'bay':
            dm = PemsBayDataset(point=True)
        elif name_dataset == 'mimic':
            dm = MIMICIIIDataset()

        edge_index, edge_weights = dm.get_connectivity()
        normalizer = dm.get_normalizer()
        dm.setup()
        print(dm.input_size())

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        return dm, edge_index, edge_weights, normalizer

    def train_test(self, hyperparameters):
        self.model = GTIGRE(
            model_type=self.model_name,
            input_size=self.dm.input_size(),
            edge_index=self.edge_index,
            edge_weights=self.edge_weights,
            normalizer=self.normalizer,
            params=hyperparameters,
            alpha=hyperparameters['alpha'] if 'alpha' in hyperparameters.keys() else None,
        )

        early_stopping = EarlyStopping(monitor='denorm_mse', patience=1, mode='min')
        self.trainer = Trainer(
            max_steps=self.max_iter_train,
            default_root_dir='reports/logs_experiments',
            accelerator=self.accelerator,
            devices=self.selected_gpu,
            gradient_clip_val=5.,
            gradient_clip_algorithm='norm',
            callbacks=[early_stopping],
            detect_anomaly=True
        )

        start_time = time.time()
        self.trainer.fit(self.model, datamodule=self.dm)
        elapsed_time = time.time() - start_time

        results = self.trainer.test(self.model, datamodule=self.dm)[0]
        results['time'] = elapsed_time
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
            results['time'],
            params
        ]

        self.results_file.loc[self.results_file.shape[0]] = row
        self.results_file.to_csv(self.save_file)

    def run(self):
        for _ in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'{self.exp_name} with {self.model_name} in {self.dataset}'):
            results = self.train_test(self.default_hyperparameters)
            self.save_results_file(results, self.default_hyperparameters)


class AverageResults:
    def __init__(self, iterations=5, gpu='auto', max_iter_train=5000, folder='results', input_file=None):
        columns = [
            'mae',
            'mse',
            'rmse',
            'denorm_mae',
            'denorm_mse',
            'denorm_mre',
            'denorm_rmse',
            'time'
        ]

        self.input_file_path = input_file
        self.input_file = pd.read_csv(input_file)
        self.iterations = iterations

        self.folder = folder
        self.gpu = gpu
        self.max_iter_train = max_iter_train

        columns = list(itertools.product(columns, ['mean', 'std']))
        self.columns = [f'{variable}-{suffix}' for variable, suffix in columns] + ['params']

    def get_row_and_name(self, results_path):
        name_dataset = results_path.split('/')[-1].split('.')[0]
        original_row = self.input_file.loc[self.input_file['dataset'] == name_dataset]

        return original_row, name_dataset
        
    def extract_results(self, results_path, model=None):
        results_model = pd.read_csv(f'{results_path}')
        original_row, name_dataset = self.get_row_and_name(results_path)

        model = original_row['model'].values[0] if model is None else model
        res = [name_dataset, model]
        for column in self.columns[:-1]:
            variable, suffix = column.split('-')
            if suffix == 'mean':
                value = results_model[variable].mean()
            else:
                value = results_model[variable].std()
            res.append(value)
        res.append(results_model['params'].values[0])
        return res

    def make_summary_dataset(self, model=None):
        columns = ['dataset', 'model'] + self.columns
        result_file = pd.DataFrame(columns=columns)

        for file in np.sort(os.listdir(self.folder)):
            if file != 'results.csv' and file.endswith('.csv'):
                results_path = f'./{self.folder}/{file}'
                row = self.extract_results(results_path, model)
                result_file.loc[len(result_file)] = row

        result_file.to_csv(f'{self.folder}/results.csv', index=False)

    def run(self):
        results_path = f'./{self.folder}'

        for i in range(len(self.input_file)):
            row = self.input_file.iloc[i]
            model = row['model']
            dataset = row['dataset']
            hyperparameters = row['params']
            experiment = BaseExperiment(
                model=model,
                dataset=dataset,
                iterations=self.iterations,
                results_path=results_path,
                gpu=self.gpu,
                max_iter_train=self.max_iter_train,
                default_hyperparameters=hyperparameters,
                save_file=dataset
            )
            experiment.run()

        self.make_summary_dataset()
