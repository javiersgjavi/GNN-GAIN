import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tsl.datasets import MetrLA
from src.utils import create_windows_from_sequence

class Dataset(Dataset):
    def __init__(self, data: npt.NDArray,
                 mask=None,
                 edge_index=None,
                 edge_weights=None,
                 prop_missing: Optional = None):
        """
        Initialize Dataset object

        Args:
        data (numpy.ndarray): Array containing data.
        prop_missing (float): Proportion of missing data to simulate.
        """
        self.data = data
        self.known_values = mask
        self.mask = mask
        self.edge_index = edge_index
        self.edge_weights = edge_weights

        if prop_missing is not None:
            # Create a mask to simulate missing data
            self.input_mask = np.random.rand(*self.data.shape) > prop_missing
            if np.sum(self.known_values) != self.known_values.size:
                self.input_mask = np.where(self.known_values == 0, 0, self.input_mask)
            self.data_missing = np.where(self.input_mask, self.data, 0.0)
        else:
            self.input_mask = np.ones_like(self.data, dtype=bool)
            self.data_missing = self.data.copy()

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

        return self.data_missing[idx], self.data[idx], self.input_mask[idx].astype(bool), self.input_mask[idx].astype(int), self.known_values[idx]

    def get_missing_rate(self):
        print(f'Missing rate: {np.round(np.mean(self.input_mask == 0), 2)}')


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
                 val_len: float = 0.1,
                 test_len: float = 0.1,
                 prop_missing: float = 0.2):

        super().__init__()

        # Load the data from a CSV file based on the specified dataset name
        if dataset == 'metr-la':
            mtrla = MetrLA()
            self.data, _, self.mask = mtrla.load(impute_zeros=False)
            self.edge_index, self.edge_weights = mtrla.get_connectivity()

        elif dataset == 'electric':
            self.data = pd.read_csv('data/electric.csv')
            self.mask = np.ones_like(self.data)

            self.edge_index = np.array([
                np.repeat(np.arange(6), 6),
                np.tile(np.arange(6), 6)
            ])
            edge_weights = []
            correlation = self.data.corr()
            for i in range(self.edge_index.shape[1]):
                e1 = self.edge_index[0, i]
                e2 = self.edge_index[1, i]
                value = correlation.iloc[e1, e2]
                edge_weights.append(value)
            self.edge_weights = np.array(edge_weights).astype(np.float32)

        # Normalize the data if requested
        self.normalizer = MinMaxScaler()
        self.data = pd.DataFrame(self.normalizer.fit_transform(self.data), columns=self.data.columns)

        self.data, self.mask = create_windows_from_sequence(self.data, self.mask, window_len=24, stride=1)

        # Convert the data to a numpy array
        self.data_numpy = self.data.astype(np.float32)

        self.prop_missing = prop_missing
        self.batch_size = batch_size
        self.val_len = val_len
        self.test_len = test_len
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
        data_train, data_test, train_mask, test_mask = train_test_split(self.data_numpy, self.mask, test_size=self.val_len + self.test_len)
        data_val, data_test, val_mask, test_mask = train_test_split(data_test, test_mask, test_size=self.val_len / (self.val_len + self.test_len))

        # Create Dataset objects for each set with missing values introduced according to prop_missing
        data_train = Dataset(data_train, prop_missing=self.prop_missing, mask=train_mask)
        data_val = Dataset(data_val, prop_missing=self.prop_missing, mask=val_mask)
        data_test = Dataset(data_test, prop_missing=self.prop_missing, mask=test_mask)

        # Create DataLoader objects for each set
        self.train_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(data_val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(data_test, batch_size=self.batch_size, shuffle=False)

    def input_size(self):
        return self.data.shape[1:]

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
