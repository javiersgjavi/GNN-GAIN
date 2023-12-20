import os
import tsl
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple

from torch.nn.utils.parametrizations import spectral_norm

def round_to_nearest_divisible(x, y):
        """
        This function rounds x to the nearest number divisible by y    
        """
        return round(x / y) * y

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
                                 stride=1):
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

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
        known_values = known_values[:, :, 0]

    for i in range(0, data.shape[0] - window_len + 1, stride):
        windows.append(data[i:i + window_len])
        windows_mask.append(mask[i:i + window_len])
        windows_known_values.append(known_values[i:i + window_len])
        windows_time_gap_matrix_f.append(time_gap_matrix_f[i:i + window_len])
        windows_time_gap_matrix_b.append(time_gap_matrix_b[i:i + window_len])

    res = (
        np.array(windows),
        np.array(windows_mask),
        np.array(windows_known_values),
        np.array(windows_time_gap_matrix_f),
        np.array(windows_time_gap_matrix_b)
    )

    for i in res:
        print(i.shape)
        
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
    return np.mean(np.abs(x - y) / np.abs(y + 1e-8)) * 100


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

def add_sn(m):    
    if isinstance(m, (nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m
    
def apply_spectral_norm(m):

    class_trans_layer = tsl.nn.blocks.encoders.transformer.Transformer
    decoder = tsl.nn.blocks.decoders.GCNDecoder

    if isinstance(m, class_trans_layer) or isinstance(m, decoder):
        for name, module in m.named_modules():
            if isinstance(module, nn.Linear):
                spectral_norm(module)

    else:

        for _, module in m.named_children():
            
            if isinstance(module, nn.Linear):
                spectral_norm(module)

            elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
                print(module.__class__)
                for p in module.state_dict().keys():
                    if 'weight' in p:
                        spectral_norm(module, name=p)
    return m


def loss_controller_ws(d_loss, g_loss, opt_id, d_i):
    sel_loss = None
    it_g_train = 5
    if opt_id == 0 and d_i < it_g_train:
        sel_loss = d_loss
        d_i += 1

    elif opt_id == 1 and d_i >= it_g_train:
        sel_loss = g_loss
        d_i = 0

    else:
        sel_loss = torch.zeros_like(d_loss, requires_grad=True, device=d_loss.device)

    return sel_loss, d_i
        