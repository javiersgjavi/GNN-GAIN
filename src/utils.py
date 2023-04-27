import numpy as np
import torch
from torch import nn
from typing import Tuple
from tsl.ops.imputation import add_missing_values


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


def create_windows_from_sequence(data, mask, known_values, window_len=12, stride=1):
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

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
        known_values = known_values[:, :, 0]

    for i in range(0, data.shape[0] - window_len + 1, stride):
        windows.append(data[i:i + window_len])
        windows_mask.append(mask[i:i + window_len])
        windows_known_values.append(known_values[i:i + window_len])

    return np.array(windows), np.array(windows_mask), np.array(windows_known_values)


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
