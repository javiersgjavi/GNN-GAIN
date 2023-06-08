import os
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tsl.datasets import MetrLA, AirQuality, PemsBay
from tsl.ops.imputation import add_missing_values
from src.utils import load_time_gap_matrix
from src.data.splitters import RatioSplitter, AQICustomInSampleSplitter, AQICustomOutSampleSplitter


class ElectricDataset:
    def __init__(self, prop_missing=0.25):
        self.prop_missing = prop_missing
        self.data = pd.read_csv('data/electric/normal.csv')
        self.training_mask = np.random.rand(*self.data.shape) > self.prop_missing
        self.eval_mask = np.ones_like(self.data)
        self.edge_index, self.edge_weights = self._calculate_connectivity()

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


class DatasetLoader(Dataset):
    def __init__(self, data: npt.NDArray,
                 mask=None,
                 known_values=None,
                 edge_index=None,
                 edge_weights=None,
                 time_gap_matrix_f=None,
                 time_gap_matrix_b=None):
        """
        Initialize Dataset object

        Args:
        data (numpy.ndarray): Array containing data.
        prop_missing (float): Proportion of missing data to simulate.
        """

        self.data = data
        self.known_values = known_values
        self.input_mask_bool = mask.astype(bool)
        self.input_mask_int = mask.astype(int)
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.time_gap_matrix_f = time_gap_matrix_f
        self.time_gap_matrix_b = time_gap_matrix_b

        self.data_missing = np.where(self.input_mask_bool, self.data, 0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
        idx (int): Index of the item to retrieve.

        Returns:
        Tuple: A tuple containing the missing data, the complete data, and the input mask.
        """

        return self.data_missing[idx], self.data[idx], self.input_mask_bool[idx], self.input_mask_int[idx], \
            self.known_values[idx], {'forward': self.time_gap_matrix_f[idx], 'backward': self.time_gap_matrix_b[idx]}

    def get_missing_rate(self):
        print(f'Missing percentaje: {np.round(np.mean(self.input_mask_int == 0) * 100, 2)}')


class DataModule(pl.LightningModule):
    """
    A PyTorch Lightning Module for handling data loading and preprocessing.

    Args:
        dataset (str): The name of the dataset to load. Must be either 'credit' or 'spam'.
        batch_size (int): The size of each batch to load.
        val_len (float): The proportion of the data to use for validation.
        test_len (float): The proportion of the data to use for testing.
        prop_missing (float): The proportion of values in the data to set to NaN to simulate missing data.
    """

    def __init__(self,
                 dataset: str = 'credit',
                 batch_size: int = 128,
                 val_len: float = 0.2,
                 test_len: float = 0.1,
                 prop_missing: float = 0.2,
                 use_time_gap_matrix: bool = False):

        super().__init__()

        self.use_time_gap_matrix = use_time_gap_matrix
        # Load the data from a CSV file based on the specified dataset name

        if dataset.endswith('_point'):
            p_fault, p_noise = 0., 0.25
        elif dataset.endswith('_block'):
            p_fault, p_noise = 0.0015, 0.05

        if self.use_time_gap_matrix:
            os.makedirs(f'./data/{dataset}/', exist_ok=True)

        if dataset.startswith('la'):
            base_data = add_missing_values(MetrLA(),
                                           p_fault=p_fault,
                                           p_noise=p_noise,
                                           min_seq=12,
                                           max_seq=12 * 4,
                                           seed=9101112)

            if self.use_time_gap_matrix:
                path_time_gap_matrix = f'./data/{dataset}/time_gap_matrix_{p_fault}_{p_noise}_{9101112}'
                time_gap_matrix_f, time_gap_matrix_b = load_time_gap_matrix(base_data, path_time_gap_matrix)


        elif dataset.startswith('bay'):
            base_data = add_missing_values(PemsBay(),
                                           p_fault=p_fault,
                                           p_noise=p_noise,
                                           min_seq=12,
                                           max_seq=12 * 4,
                                           seed=56789)

            if self.use_time_gap_matrix:
                path_time_gap_matrix = f'./data/{dataset}/time_gap_matrix_{p_fault}_{p_noise}_{56789}'
                time_gap_matrix_f, time_gap_matrix_b = load_time_gap_matrix(base_data, path_time_gap_matrix)

        elif dataset.startswith('air'):
            base_data = AirQuality(small='36' in dataset)
            if self.use_time_gap_matrix:
                path_time_gap_matrix = f'./data/{dataset}/time_gap_matrix'
                time_gap_matrix_f, time_gap_matrix_b = load_time_gap_matrix(base_data, path_time_gap_matrix)

        elif dataset.startswith('electric'):
            base_data = ElectricDataset(prop_missing=prop_missing)

        if not self.use_time_gap_matrix:
            time_gap_matrix_f = np.zeros_like(base_data.training_mask)
            time_gap_matrix_b = np.zeros_like(base_data.training_mask)

        self.dataset_name = dataset
        self.base_data = base_data
        self.data = base_data.dataframe()
        self.mask = base_data.training_mask
        self.known_values = base_data.eval_mask
        self.edge_index, self.edge_weights = base_data.get_connectivity()
        self.time_gap_matrix_f = time_gap_matrix_f
        self.time_gap_matrix_b = time_gap_matrix_b

        self.normalizer = MinMaxScaler()
        self.data = pd.DataFrame(self.normalizer.fit_transform(self.data), columns=self.data.columns)

        self.splitter = None

        self.prop_missing = prop_missing
        self.batch_size = batch_size
        self.val_len = val_len
        self.test_len = test_len
        self.shape = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage=None):
        """
        Splits the data into train, validation, and test sets, and creates PyTorch DataLoader objects for each set.

        Args:
            stage: (str): The stage of training (fit or test). Unused in this implementation.
        """

        # Split the data into train, validation, and test sets using train_test_split
        if self.dataset_name.startswith('air'):
            if self.dataset_name.endswith('_in'):

                self.splitter = AQICustomInSampleSplitter(
                    data=self.data,
                    mask=self.mask,
                    known_values=self.known_values,
                    time_gap_matrix_f=self.time_gap_matrix_f,
                    time_gap_matrix_b=self.time_gap_matrix_b,
                    windows_len=36 if self.dataset_name.startswith('air-36') else 24,
                    base_data=self.base_data,
                    name_time_col='datetime' if self.dataset_name.startswith('air-36') else 'time',
                )
            else:
                self.splitter = AQICustomOutSampleSplitter(
                    data=self.data,
                    mask=self.mask,
                    known_values=self.known_values,
                    time_gap_matrix_f=self.time_gap_matrix_f,
                    time_gap_matrix_b=self.time_gap_matrix_b,
                    windows_len=36 if self.dataset_name.startswith('air-36') else 24,
                    base_data=self.base_data,
                    name_time_col='datetime' if self.dataset_name.startswith('air-36') else 'time',
                )
        else:
            self.splitter = RatioSplitter(
                data=self.data,
                mask=self.mask,
                known_values=self.known_values,
                time_gap_matrix_f=self.time_gap_matrix_f,
                time_gap_matrix_b=self.time_gap_matrix_b,
            )

        train, val, test, shape = self.splitter.split()
        self.shape = shape

        # Create Dataset objects for each set with missing values introduced according to prop_missing
        data_train = DatasetLoader(
            train['data'],
            mask=train['mask'],
            known_values=train['known_values'],
            time_gap_matrix_f=train['tgm_f'],
            time_gap_matrix_b=train['tgm_b']
        )

        data_val = DatasetLoader(
            val['data'],
            mask=val['mask'],
            known_values=val['known_values'],
            time_gap_matrix_f=val['tgm_f'],
            time_gap_matrix_b=val['tgm_b']
        )

        data_test = DatasetLoader(
            test['data'],
            mask=test['mask'],
            known_values=test['known_values'],
            time_gap_matrix_f=test['tgm_f'],
            time_gap_matrix_b=test['tgm_b']
        )

        # Create DataLoader objects for each set
        self.train_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(data_val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(data_test, batch_size=self.batch_size, shuffle=False)

        data_train.get_missing_rate()

    def input_size(self):
        return self.shape[1:]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def get_connectivity(self):
        return self.edge_index, self.edge_weights

    def get_normalizer(self):
        return self.normalizer
