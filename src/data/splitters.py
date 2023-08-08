import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from src.utils import create_windows_from_sequence


class Splitter:
    def __init__(self, data=None, mask=None, known_values=None, windows_len=24, stride=1, time_gap_matrix_f=None,
                 time_gap_matrix_b=None):
        self.data = data
        self.mask = mask
        self.known_values = known_values
        self.windows_len = windows_len
        self.stride = stride
        self.tgm_f = time_gap_matrix_f
        self.tgm_b = time_gap_matrix_b

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def create_windows(self):
        self.data, self.mask, self.known_values, self.tgm_f, self.tgm_b = create_windows_from_sequence(
            self.data,
            self.mask,
            self.known_values,
            self.tgm_f,
            self.tgm_b,
            window_len=self.windows_len,
            stride=self.stride
        )


class RatioSplitter(Splitter):
    def __init__(self, val_len=0.1, test_len=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_len = val_len
        self.test_len = test_len

    def split(self):
        self.create_windows()
        data_train, data_test, \
            train_mask, test_mask, \
            train_known_values, test_known_values, \
            train_tgm_f, test_tgm_f, \
            train_tgm_b, test_tgm_b = train_test_split(self.data,
                                                       self.mask,
                                                       self.known_values,
                                                       self.tgm_f,
                                                       self.tgm_b,
                                                       test_size=self.val_len + self.test_len)

        data_val, data_test, \
            val_mask, test_mask, \
            val_known_values, test_known_values, \
            val_tgm_f, test_tgm_f, \
            val_tgm_b, test_tgm_b = train_test_split(data_test,
                                                     test_mask,
                                                     test_known_values,
                                                     test_tgm_f,
                                                     test_tgm_b,
                                                     test_size=self.val_len / (
                                                             self.val_len + self.test_len))

        train = {'data': data_train,
                 'mask': train_mask,
                 'known_values': train_known_values,
                 'tgm_f': train_tgm_f,
                 'tgm_b': train_tgm_b}

        val = {'data': data_val,
               'mask': val_mask,
               'known_values': val_known_values,
               'tgm_f': val_tgm_f,
               'tgm_b': val_tgm_b}

        test = {'data': data_test,
                'mask': test_mask,
                'known_values': test_known_values,
                'tgm_f': test_tgm_f,
                'tgm_b': test_tgm_b}

        return train, val, test, self.data.shape


class AQICustomSplitter(Splitter):
    def __init__(self, base_data=None, test_months=None, name_time_col=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_data = base_data
        self.test_months = test_months if test_months is not None else [6, 9, 12, 3]

        self.name_time_col = name_time_col

        self.selected_periods = self.define_selected_periods()
        self.selected_indexes = self.define_selected_indexes()

    def define_selected_periods(self):
        range_months = []
        for month in self.test_months:
            month_end = month + 1 if month != 12 else 1
            start_year = 2014 if month >= 5 else 2015
            end_year = 2014 if month > 5 and month != 12 else 2015
            range_months.append(
                pd.date_range(start=f'{start_year}-{month}-01', end=f'{end_year}-{month_end}-01', freq='H',
                              inclusive='left')
            )

        return range_months

    def define_selected_indexes(self):
        df = self.base_data.dataframe().copy()
        df = df.reset_index().set_index(np.arange(len(df)))
        test_sets_index = []
        for range_index in self.selected_periods:
            test_index = df.loc[df[self.name_time_col].isin(range_index)].index
            test_sets_index.append(test_index)

        return np.concatenate(test_sets_index)


class AQICustomInSampleSplitter(AQICustomSplitter):

    def split(self):
        self.create_windows()

        train = {'data': self.data,
                 'mask': self.mask,
                 'known_values': self.known_values,
                 'tgm_f': self.tgm_f,
                 'tgm_b': self.tgm_b}

        test = {'data': self.data[self.selected_indexes],
                'mask': self.mask[self.selected_indexes],
                'known_values': self.known_values[self.selected_indexes],
                'tgm_f': self.tgm_f[self.selected_indexes],
                'tgm_b': self.tgm_b[self.selected_indexes]}

        val = test.copy()

        return train, val, test, self.data.shape


class AQICustomOutSampleSplitter(AQICustomSplitter):

    def create_windows_iterative(self, indexes_set):
        data, mask, known_values, tgm_f, tgm_b = [], [], [], [], []
        for index_set in indexes_set:
            data_i, mask_i, known_values_i, tgm_f_i, tgm_b_i = create_windows_from_sequence(
                self.data.iloc[index_set],
                self.mask[index_set],
                self.known_values[index_set],
                self.tgm_f[index_set],
                self.tgm_b[index_set],
                window_len=self.windows_len,
                stride=self.stride
            )
            data.append(data_i)
            mask.append(mask_i)
            known_values.append(known_values_i)
            tgm_f.append(tgm_f_i)
            tgm_b.append(tgm_b_i)

        res = {'data': np.concatenate(data),
               'mask': np.concatenate(mask),
               'known_values': np.concatenate(known_values),
               'tgm_f': np.concatenate(tgm_f),
               'tgm_b': np.concatenate(tgm_b)}

        return res

    def split(self):
        train_indexes = np.setdiff1d(np.arange(len(self.data)), self.selected_indexes)

        stop_positions_train = np.where(np.diff(train_indexes) != 1)[0] + 1
        stop_positions_test = np.where(np.diff(self.selected_indexes) != 1)[0] + 1

        train_sets_indexes = np.split(train_indexes, stop_positions_train)
        test_sets_indexes = np.split(self.selected_indexes, stop_positions_test)

        train = self.create_windows_iterative(train_sets_indexes)
        test = self.create_windows_iterative(test_sets_indexes)
        val = test.copy()

        return train, val, test, train['data'].shape
