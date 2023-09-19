import pandas as pd
import numpy as np

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
