import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models.gain import GAIN
from src.data.datasets import DataModule, VirtualSensingDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR


def print_dict(dictionary, max_iter_train):
    for key, value in dictionary.items():
        print(f'{key}: {value}')
    print(f'Max steps training: {max_iter_train}')


class Experiment:
    def __init__(self, model, dataset, iterations, results_path, accelerator='gpu', save_file=None,
                 max_iter_train=5000, gpu='auto', default_hyperparameters=None, batch_size=None, time_gap=None):
        self.columns = [
            'mae',
            'mse',
            'rmse',
            'denorm_mae',
            'denorm_mse',
            'denorm_mre',
            'denorm_rmse',
            'params'
        ]

        if default_hyperparameters is not None:
            default_hyperparameters = default_hyperparameters \
                .replace("'", '"') \
                .replace('True', 'true') \
                .replace('False', 'false')

            default_hyperparameters = json.loads(default_hyperparameters)

            self.time_gap = default_hyperparameters['use_time_gap_matrix']
            self.batch_size = default_hyperparameters['batch_size'][0]

        else:
            self.time_gap = time_gap
            self.batch_size = batch_size

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

    def load_file(self):

        os.makedirs(self.results_path, exist_ok=True)

        if os.path.exists(self.save_file):
            results_file = pd.read_csv(self.save_file, index_col='Unnamed: 0')

        else:
            results_file = pd.DataFrame(columns=self.columns)

        return results_file

    def prepare_data(self):
        dm = DataModule(dataset=self.dataset, batch_size=self.batch_size, use_time_gap_matrix=self.time_gap)
        edge_index, edge_weights = dm.get_connectivity()
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        return dm, edge_index, edge_weights, normalizer

    def train_test(self, hyperparameters):
        self.model = GAIN(
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
        )

        self.trainer.fit(self.model, datamodule=self.dm)

        results = self.trainer.test(self.model, datamodule=self.dm)[0]
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
            params
        ]

        self.results_file.loc[self.results_file.shape[0]] = row
        self.results_file.to_csv(self.save_file)

    def run(self):
        for _ in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'{self.exp_name} with {self.model_name} in {self.dataset}'):
            results = self.train_test(self.default_hyperparameters)
            self.save_results_file(results, self.default_hyperparameters)


class ExperimentAblation(Experiment):
    def __init__(self, ablation=None, suffix=None, *args, **kwargs):
        self.ablation = ablation
        super().__init__(*args, **kwargs)
        self.save_file = self.save_file.replace('.csv', f'_{suffix}.csv')
        self.results_file = self.load_file()
        self.exp_name = 'Ablation experiment'

        self.make_architecture_ablation()

    def make_architecture_ablation(self):
        if 'no_bi' in self.ablation:
            self.default_hyperparameters['bi'] = False
        elif 'no_tg' in self.ablation:
            self.default_hyperparameters['use_time_gap_matrix'] = False

    def make_graph_ablation(self, edge_index, edge_weights):

        dtype_w = edge_weights.dtype
        dtype_e = edge_index.dtype

        if self.ablation == 'fc':
            edge_weights = np.ones(edge_weights.shape).astype(dtype_w)
        elif self.ablation == 'nc':
            num_nodes = edge_index.max() + 1
            edge_index_arange = np.arange(num_nodes)
            edge_index = np.array([edge_index_arange, edge_index_arange]).astype(dtype_e)
            edge_weights = np.ones(num_nodes).astype(dtype_w)

        return edge_index, edge_weights

    def prepare_data(self):
        dm = DataModule(dataset=self.dataset, batch_size=self.batch_size, use_time_gap_matrix=self.time_gap)
        edge_index, edge_weights = dm.get_connectivity()
        edge_index, edge_weights = self.make_graph_ablation(edge_index, edge_weights)
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        return dm, edge_index, edge_weights, normalizer

    def train_test(self, hyperparameters):
        self.model = GAIN(
            model_type=self.model_name,
            input_size=self.dm.input_size(),
            edge_index=self.edge_index,
            edge_weights=self.edge_weights,
            normalizer=self.normalizer,
            params=hyperparameters,
            alpha=hyperparameters['alpha'] if 'alpha' in hyperparameters.keys() else None,
            ablation_gan = True if self.ablation == 'no_gan' else False,
            ablation_reconstruction = True if self.ablation == 'no_reconstruction' else False
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
        )

        self.trainer.fit(self.model, datamodule=self.dm)

        results = self.trainer.test(self.model, datamodule=self.dm)[0]
        return results


class VirtualSensingExperiment(Experiment):
    def __init__(self, masked=None, *args, **kwargs):
        self.masked = masked
        super().__init__(*args, **kwargs)
        self.predictions = {i: None for i in range(self.iterations)}
        self.exp_name = 'Virtual sensing experiment'

    def prepare_data(self):
        dm = VirtualSensingDataModule(dataset=self.dataset, batch_size=self.batch_size,
                                      use_time_gap_matrix=self.time_gap, masked=self.masked)
        edge_index, edge_weights = dm.get_connectivity()
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        return dm, edge_index, edge_weights, normalizer

    def get_predictions(self):
        predictions = self.trainer.predict(model=self.model, datamodule=self.dm)
        return predictions

    def run(self):
        for i in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'{self.exp_name} with {self.model_name} in {self.dataset}'):
            _ = self.train_test(self.default_hyperparameters)
            self.predictions[i] = self.get_predictions()

        return self.predictions


class MissingDataSensitivityExperiment(Experiment):
    def __init__(self, p_noise, *args, **kwargs):
        self.p_noise = p_noise
        super().__init__(*args, **kwargs)
        self.exp_name = 'Missing data sensitivity experiment'

    def prepare_data(self):
        dm = DataModule(dataset=self.dataset, batch_size=self.batch_size, use_time_gap_matrix=self.time_gap,
                        p_noise=self.p_noise)
        edge_index, edge_weights = dm.get_connectivity()
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        percentage = dm.get_missing_rate()
        self.save_file = self.save_file.replace('.csv', f'_{int(round(percentage, -1))}.csv')
        self.results_file = self.load_file()

        return dm, edge_index, edge_weights, normalizer


class RandomSearchExperiment(Experiment):
    def __init__(self, bi=False, param_loader=None, *args, **kwargs):
        self.bi = bi
        self.params_loader = param_loader
        batch_size = self.params_loader.random_params['batch_size'][0][0]

        super().__init__(batch_size=batch_size, *args, **kwargs)

        self.exp_name = 'Random search experiment'

    def train_test(self, hyperparameters):
        hyperparameters['use_time_gap_matrix'] = self.time_gap
        hyperparameters['bi'] = self.bi
        return super().train_test(hyperparameters)

    def run(self):
        for i in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'{self.exp_name} with {self.model_name} in {self.dataset}'):
            hyperparameters = self.params_loader.get_params(i)
            print_dict(hyperparameters, self.max_iter_train)
            results = self.train_test(hyperparameters)
            self.save_results_file(results, hyperparameters)
