import os
import torch
import numpy as np
import pandas as pd
import copy

from src.data.splitters import RatioSplitter
from src.data.datamodule import DataModule
from src.utils import load_time_gap_matrix

class MIMICIIIDataset(DataModule):
    def __init__(self, added_missing=0.1, seed=None, use_time_gap_matrix=False, **kwargs):

        print(f'Added missing: {added_missing}')
        if seed is None:
            seed = 50
        np.random.seed(seed)

        self.prop_missing = added_missing
        self.dataset_name = f'mimiciii_{self.prop_missing}'
        self.use_time_gap_matrix = use_time_gap_matrix

        train_data = np.load('data/mimic-iii/train/x_dl.npy')
        val_data = np.load('data/mimic-iii/val/x_dl.npy')
        test_data = np.load('data/mimic-iii/test/x_dl.npy')

        data = np.concatenate([train_data, val_data, test_data], axis=0)
        print(data.shape)
        data = np.transpose(data, (0, 2, 1)).astype(np.float32).reshape(-1, data.shape[1])

        data_pd = pd.DataFrame(data)
        

        self.edge_index, self.edge_weights = self._calculate_connectivity(data_pd)
        eval_mask = ~np.isnan(data)

        if self.prop_missing != 0:
            training_mask = np.random.rand(*data.shape) > self.prop_missing
            training_mask = np.where(eval_mask, training_mask, False)
        else:
            training_mask = copy.deepcopy(eval_mask)

        print(training_mask.sum() / training_mask.size)

        if self.use_time_gap_matrix:
            os.makedirs(f'./data/{self.dataset_name}/', exist_ok=True)
            path_time_gap_matrix = f'./data/{self.dataset_name}/time_gap_matrix_{seed}'
            self.time_gap_matrix_f, self.time_gap_matrix_b = load_time_gap_matrix(data, path_time_gap_matrix)
        
        else:
            self.time_gap_matrix_f = np.zeros_like(training_mask)
            self.time_gap_matrix_b = np.zeros_like(training_mask)

        self.batch_size=32
        self.data = data_pd.fillna(0)
        self.base_data = np.where(data==np.nan, 0, data)
        self.mask = training_mask
        self.known_values = eval_mask

        super().__init__(**kwargs)

    def custom_setup(self, stage, train, val, test):
        return super().setup(stage, train, val, test)

    def setup(self, stage=None):
        # Hay que quitarle la opción de crear windows porque ya las tenemos
        self.splitter = RatioSplitter(
                data=self.data,
                mask=self.mask,
                known_values=self.known_values,
                time_gap_matrix_f=self.time_gap_matrix_f,
                time_gap_matrix_b=self.time_gap_matrix_b,
                windows_len=49,
                stride=49
            )
        train, val, test, shape = self.splitter.split()
        self.shape = shape
        return super().setup(stage, train, val, test)


    def _calculate_connectivity(self, data_pd):

        n_vars = data_pd.shape[1]

        edge_index = np.array([
            np.repeat(np.arange(n_vars), n_vars),
            np.tile(np.arange(n_vars), n_vars)
        ])

        correlation = data_pd.corr()

        edge_weights = []
        for i in range(edge_index.shape[1]):
            e1 = edge_index[0, i]
            e2 = edge_index[1, i]
            value = correlation.iloc[e1, e2]
            edge_weights.append(value)

        edge_weights = np.array(edge_weights).astype(np.float32)

        return edge_index, edge_weights

    def get_connectivity(self):
        return self.edge_index, self.edge_weights
    
    def dataframe(self):
        return self.data
    
class MIMICIIIDatasetToImpute(MIMICIIIDataset):
    def __init__(self, *args, **kwargs):
        # change prop_missing to 0
        kwargs['added_missing'] = 0
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        self.splitter = RatioSplitter(
                data=self.data,
                mask=self.mask,
                known_values=self.known_values,
                time_gap_matrix_f=self.time_gap_matrix_f,
                time_gap_matrix_b=self.time_gap_matrix_b,
                windows_len=49,
                stride=49
            )
        self.splitter.create_windows()

        self.data = self.splitter.data
        self.mask = self.splitter.mask
        self.known_values = self.splitter.known_values
        self.time_gap_matrix_f = self.splitter.tgm_f
        self.time_gap_matrix_b = self.splitter.tgm_b


        train = {'data': self.data,
                 'mask': self.mask,
                 'known_values': self.known_values,
                 'tgm_f': self.time_gap_matrix_f,
                 'tgm_b': self.time_gap_matrix_b}

        val  = copy.deepcopy(train)
        test = copy.deepcopy(train)
        self.shape = self.data.shape
        return super().custom_setup(stage, train, val, test)

class ImputedMIMICIIIDataset:

    def __init__(self, data, output_path):
        self.data = np.concatenate(data, axis=0)
        self.data = np.transpose(self.data, (0, 2, 1)).astype(np.float32)

        self.output_path = f'{output_path}mimic-iii'

    def save(self):

        train_len = 4951
        val_len = 617
        test_len = 613

        train = self.data[:train_len]
        val = self.data[train_len:train_len+val_len]
        test = self.data[train_len+val_len:]

        os.makedirs(f'{self.output_path}/train/', exist_ok=True)
        os.makedirs(f'{self.output_path}/val/', exist_ok=True)
        os.makedirs(f'{self.output_path}/test/', exist_ok=True)

        np.save(f'{self.output_path}/train/x_dl.npy', train)
        np.save(f'{self.output_path}/val/x_dl.npy', val)
        np.save(f'{self.output_path}/test/x_dl.npy', test)

        np.save(f'{self.output_path}/train/x_ml.npy', train.reshape(train.shape[0], -1))
        np.save(f'{self.output_path}/val/x_ml.npy', val.reshape(val.shape[0], -1))
        np.save(f'{self.output_path}/test/x_ml.npy', test.reshape(test.shape[0], -1))

        

        