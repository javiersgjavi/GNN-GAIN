import os
import numpy as np
from tsl.datasets import AirQuality

from src.utils import load_time_gap_matrix
from src.data.splitters import  AQICustomInSampleSplitter, AQICustomOutSampleSplitter

class AirDataset(DataModule):
    def __init__(self, out=False, use_time_gap_matrix=False, **kwargs):

        self.out = out
        self.dataset_name = f'{self.dataset}_out' if out else f'{self.dataset}_in'

        base_data = AirQuality(small=self.small)

        if self.use_time_gap_matrix:
            os.makedirs(f'./data/{self.dataset_name}/', exist_ok=True)
            path_time_gap_matrix = f'./data/{self.dataset_name}/time_gap_matrix'
            self.time_gap_matrix_f, self.time_gap_matrix_b = load_time_gap_matrix(base_data, path_time_gap_matrix)
        else:
            self.time_gap_matrix_f = np.zeros_like(base_data.training_mask)
            self.time_gap_matrix_b = np.zeros_like(base_data.training_mask)

        self.edge_index, self.edge_weights = base_data.get_connectivity(include_self=False, threshold=0.1)

        self.batch_size=64
        self.base_data = base_data
        self.data = self.base_data.dataframe()
        self.mask = self.base_data.training_mask
        self.known_values = self.base_data.eval_mask

        super().__init__(**kwargs)

    def setup(self, stage=None):
        if self.out:         
            self.splitter = AQICustomOutSampleSplitter(
                data=self.data,
                mask=self.mask,
                known_values=self.known_values,
                time_gap_matrix_f=self.time_gap_matrix_f,
                time_gap_matrix_b=self.time_gap_matrix_b,
                windows_len=36 if self.dataset is 'air-36' else 24,
                base_data=self.base_data,
                name_time_col='datetime' if self.dataset is 'air-36' else 'time',
            )
        else:
            self.splitter = AQICustomInSampleSplitter(
                    data=self.data,
                    mask=self.mask,
                    known_values=self.known_values,
                    time_gap_matrix_f=self.time_gap_matrix_f,
                    time_gap_matrix_b=self.time_gap_matrix_b,
                    windows_len=36 if self.dataset is 'air-36' else 24,
                    base_data=self.base_data,
                    name_time_col='datetime' if self.dataset is 'air-36' else 'time',
                )
                

        train, val, test, shape = self.splitter.split()
        self.shape = shape

        super().setup(stage, train, val, test)

class AQI(AirDataset):
    def __init__(self, **kwargs):
        self.dataset='air'
        self.small = False
        super().__init__(**kwargs)

class AQI36(AirDataset):
    def __init__(self, **kwargs):
        self.dataset='air-36'
        self.small = True
        super().__init__(**kwargs)
