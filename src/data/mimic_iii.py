import os
import numpy as np
import pandas as pd

from src.data.splitters import RatioSplitter
from src.data.datamodule import DataModule
from src.utils import load_time_gap_matrix

class MIMICIIIDataset(DataModule):
    def __init__(self, added_missing=0.1, seed=None, use_time_gap_matrix=False, **kwargs):

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
        training_mask = np.random.rand(*data.shape) > self.prop_missing
        training_mask = np.where(eval_mask, training_mask, False)

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