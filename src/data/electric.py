import os
import numpy as np
import pandas as pd

from src.data.datamodule import DataModule
from src.utils import load_time_gap_matrix

class ElectricDataset(DataModule):
    def __init__(self, prop_missing=0.1, seed=None, use_time_gap_matrix=False, **kwargs):

        if seed is None:
            seed = 50
        np.random.seed(seed)

        self.prop_missing = prop_missing
        self.dataset_name = f'electric_{self.prop_missing}'
        self.use_time_gap_matrix = use_time_gap_matrix
        print(self.prop_missing)

        data = pd.read_csv('data/electric/normal.csv').astype(np.float32)
        training_mask = np.random.rand(*data.shape) > self.prop_missing
        training_mask = training_mask[:, :, np.newaxis]
        eval_mask = np.ones_like(data)[:, :, np.newaxis]
        self.edge_index, self.edge_weights = self._calculate_connectivity()

        if self.use_time_gap_matrix:
            os.makedirs(f'./data/{self.dataset_name}/', exist_ok=True)
            path_time_gap_matrix = f'./data/{self.dataset_name}/time_gap_matrix_{seed}'
            self.time_gap_matrix_f, self.time_gap_matrix_b = load_time_gap_matrix(data, path_time_gap_matrix)
        else: 
            self.time_gap_matrix_f = np.zeros_like(self.training_mask)
            self.time_gap_matrix_b = np.zeros_like(self.training_mask)

        self.batch_size=32
        self.base_data = data.to_numpy()
        self.data = data
        self.mask = training_mask
        self.known_values = eval_mask
        
        super().__init__(**kwargs)
    def _calculate_connectivity(self):
        edge_index = np.array([
            np.repeat(np.arange(6), 6),
            np.tile(np.arange(6), 6)
        ])
        edge_weights = []
        correlation = self.data.corr()
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