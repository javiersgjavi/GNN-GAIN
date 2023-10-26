import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple
from tsl.ops.imputation import add_missing_values

from sklearn.preprocessing import MinMaxScaler


def init_weights_xavier(m: nn.Module) -> None:
    """
    Initialize the weights of the neural network module using the Xavier initialization method.

    Args:
        m (nn.Module): Neural network module

    Returns:
        None
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  # Este if lo he añadido a posteriori
            m.bias.data.fill_(0.01)


def create_windows_from_sequence(data, mask, known_values, time_gap_matrix_f, time_gap_matrix_b, window_len=12,
                                 stride=1, exog_time=None):
    """
    Create windows from a sequence.

    Args:
        data (np.ndarray): Sequence data
        windows_len (int): Length of the windows

    Returns:
        np.ndarray: Windows
    """
    windows = []
    windows_mask = []
    windows_known_values = []
    windows_time_gap_matrix_f = []
    windows_time_gap_matrix_b = []
    exog_windows = []

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
        known_values = known_values[:, :, 0]

    for i in range(0, data.shape[0] - window_len + 1, stride):
        window_pos = slice(i, i + window_len)
        windows.append(data[window_pos])
        windows_mask.append(mask[window_pos])
        windows_known_values.append(known_values[window_pos])
        windows_time_gap_matrix_f.append(time_gap_matrix_f[window_pos])
        windows_time_gap_matrix_b.append(time_gap_matrix_b[window_pos])
        if exog_time is not None:
            shape = windows[i].shape
            exog_windows.append(create_exog_windows_time(exog_time, window_pos, shape))

    res = (
        np.array(windows),
        np.array(windows_mask),
        np.array(windows_known_values),
        np.array(windows_time_gap_matrix_f),
        np.array(windows_time_gap_matrix_b),
        np.array(exog_windows) if exog_time is not None else None
    )
    return res

def generate_uniform_noise(tensor_like, low=0, high=0.01):
    return torch.distributions.uniform.Uniform(low, high).sample(tensor_like.shape).to(tensor_like.device)


def mean_relative_error(x: np.array, y: np.array) -> np.array:
    """
    Compute the mean relative error between two tensors.

    Args:
        x (np.array): First tensor
        y (np.array): Second tensor

    Returns:
        np.array: Mean relative error
    """
    return np.mean(np.abs(x - y) / np.abs(y)) * 100


def count_missing_sequences(matriz, max_time_gap=24):
    rows, nodes = matriz.shape[:2]
    res = np.zeros_like(matriz).astype(np.float32)
    norm_values = {i: i / max_time_gap for i in range(max_time_gap + 1)}
    for n in tqdm(range(nodes), desc='Counting missing sequences'):
        current_sequence = 0
        for r in range(rows):
            if matriz[r, n, 0] == 0:
                if current_sequence < max_time_gap:
                    current_sequence += 1
                res[r, n, 0] = norm_values[current_sequence]
            else:
                current_sequence = 0

    return res[:, :, 0]


def load_time_gap_matrix(base_data, path):
    data_f = base_data.training_mask.astype(int)
    data_b = data_f[::-1, :, :]

    # Load forward time gap matrix
    if os.path.exists(f'{path}_f.npy'):
        time_gap_matrix_f = np.load(f'{path}_f.npy')
    else:
        time_gap_matrix_f = count_missing_sequences(data_f)
        np.save(f'{path}_f.npy', time_gap_matrix_f)

    # Load backward time gap matrix
    if os.path.exists(f'{path}_b.npy'):
        time_gap_matrix_b = np.load(f'{path}_b.npy')
    else:
        time_gap_matrix_b = count_missing_sequences(data_b)
        np.save(f'{path}_b.npy', time_gap_matrix_b)

    return time_gap_matrix_f, time_gap_matrix_b[::-1, :]


def loss_d(d_prob: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Compute the discriminator loss.

    Args:
        d_prob (torch.Tensor): Discriminator output probabilities
        m (torch.Tensor): Mask tensor indicating the location of missing values

    Returns:
        torch.Tensor: Discriminator loss
    """
    return -torch.mean(m * torch.log(d_prob + 1e-8) + (1 - m) * torch.log(1. - d_prob + 1e-8))


def loss_g(d_prob: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Compute the generator loss.

    Args:
        d_prob (torch.Tensor): Discriminator output probabilities
        m (torch.Tensor): Mask tensor indicating the location of missing values

    Returns:
        torch.Tensor: Generator loss
    """
    return -torch.mean((1 - m) * torch.log(d_prob + 1e-8))

def harmonic_month(value):
    value -=1
    frac = 2*np.pi/12
    cos = np.cos(frac*value)
    sin = np.sin(frac*value)
    return cos, sin

def get_stats_month(df):
    shape = (len(np.unique(df.index.month)), len(df.columns), 2)
    stats_months = np.zeros(shape)

    for i in range(stats_months.shape[0]):
        stats = df[df.index.month == i+1].describe().loc[['mean', 'std'], df.columns].values
        stats_months[i] = stats.transpose()

    scaler = MinMaxScaler().fit(stats_months.reshape(-1,2))

    for i in range(stats_months.shape[0]):
        stats_months[i] = scaler.transform(stats_months[i])

    np.save('stats_months.npy', stats_months)
    return stats_months

def create_exog_windows_time(exog_time, window_pos, shape):
    stats_time, dates = exog_time
    dates_window = dates[window_pos]
    exog_data = np.zeros((shape[0], shape[1], 4))
    exog_data[:, :, :2] = stats_time[dates_window.month-1]
    harm_month_data = np.array(harmonic_month(dates_window.month)).transpose()
    exog_data[:, :, 2:] = np.repeat(harm_month_data, shape[0], axis=0).reshape(shape[0], shape[1], 2)
    return exog_data
    


