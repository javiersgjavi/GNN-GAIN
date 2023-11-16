import os
import numpy as np
from src.data.datamodule import DataModule

from tsl.ops.imputation import add_missing_values
from tsl.datasets import MetrLA, PemsBay
from src.utils import load_time_gap_matrix
from src.data.splitters import RatioSplitter

class TrafficDataset(DataModule):
    def __init__(self, point=True, p_fault=None, p_noise=None, use_time_gap_matrix=False, **kwargs):

        self.dataset_name = f'{self.dataset}_point' if point else f'{self.dataset}_block'
        self.use_time_gap_matrix = use_time_gap_matrix

        p_fault_base = 0. if point else 0.0015
        p_noise_base = 0.25 if point else 0.05

        self.p_fault = p_fault_base if p_fault is None else p_fault
        self.p_noise = p_noise_base if p_noise is None else p_noise

        base_data = add_missing_values(
            self.data_class(),
            p_fault= self.p_fault,
            p_noise=self.p_noise,
            min_seq=12,
            max_seq=12 * 4,
            seed=self.seed
        )

        if self.use_time_gap_matrix:
            os.makedirs(f'./data/{self.dataset_name}/', exist_ok=True)
            path_time_gap_matrix = f'./data/{self.dataset_name}/time_gap_matrix_{self.p_fault}_{self.p_noise}_{9101112}'
            self.time_gap_matrix_f, self.time_gap_matrix_b = load_time_gap_matrix(base_data, path_time_gap_matrix)
        else: 
            self.time_gap_matrix_f = np.zeros_like(base_data.training_mask)
            self.time_gap_matrix_b = np.zeros_like(base_data.training_mask)

        self.edge_index, self.edge_weights = base_data.get_connectivity()

        self.batch_size=64
        self.base_data = base_data
        self.data = self.base_data.dataframe()
        self.mask = self.base_data.training_mask
        self.known_values = self.base_data.eval_mask

        super().__init__(**kwargs)

    def setup(self, stage=None):
        self.splitter = RatioSplitter(
                data=self.data,
                mask=self.mask,
                known_values=self.known_values,
                time_gap_matrix_f=self.time_gap_matrix_f,
                time_gap_matrix_b=self.time_gap_matrix_b,
            )

        train, val, test, shape = self.splitter.split()
        self.shape = shape

        super().setup(stage, train, val, test)

    def get_connectivity(self):
        return self.edge_index, self.edge_weights
    
class MetrLADataset(TrafficDataset):
    def __init__(self, **kwargs):
        self.data_class = MetrLA
        self.seed = 9101112
        self.dataset= f'la'
        super().__init__(**kwargs)

class PemsBayDataset(TrafficDataset):
    def __init__(self, **kwargs):
        self.data_class = PemsBay
        self.seed = 9101112
        self.dataset='bay'
        super().__init__(**kwargs)

        