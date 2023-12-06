import numpy as np
import pandas as pd
import numpy.typing as npt
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler

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
        return np.round(np.mean(self.input_mask_int == 0) * 100, 2)


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
                 val_len: float = 0.2,
                 test_len: float = 0.1):

        super().__init__()        
        
        self.normalizer = MinMaxScaler()
        self.data = pd.DataFrame(self.normalizer.fit_transform(self.data), columns=self.data.columns)

        self.splitter = None

        self.val_len = val_len
        self.test_len = test_len
        self.shape = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.missing_rate = None

    def setup(self, stage=None, train=None, val=None, test=None):
        """
        Splits the data into train, validation, and test sets, and creates PyTorch DataLoader objects for each set.

        Args:
            stage: (str): The stage of training (fit or test). Unused in this implementation.
        """

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

        self.missing_rate = data_train.get_missing_rate()
        
        print(f'Missing percentaje: {self.missing_rate}')

    def input_size(self):
        return self.shape[1:]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
    
    def predict_dataloader(self):
        return self.test_loader

    def get_connectivity(self):
        return self.edge_index, self.edge_weights

    def get_normalizer(self):
        return self.normalizer
    
    def get_missing_rate(self):
        return self.missing_rate
    
class VirtualSensingDataModule(DataModule):
    def __init__(self, masked=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked = self.data.columns[masked].get_level_values(0)
        print(f'Nodes to mask: {self.masked}')
        self.id_to_mask = [np.where(self.data.columns.get_level_values(0) == i)[0][0] for i in self.masked]
        self.mask[:, self.id_to_mask] = False,
        self.known_values[:, self.id_to_mask] = False
