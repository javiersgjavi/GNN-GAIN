import numpy as np
import torch
from torch import nn
from typing import Tuple


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
        m.bias.data.fill_(0.01)


def generate_missing_mask(
        x: torch.Tensor,
        input_mask: torch.Tensor,
        missing_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a binary mask for randomly missing values in the input tensor and apply it to the tensor.

    Args:
        x (torch.Tensor): Input tensor
        input_mask (torch.Tensor): Mask tensor indicating already missing values in the input tensor
        missing_rate (float): Probability of missing values

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the input tensor with missing values set to 0,
                                            and the mask tensor indicating the locations of missing values
    """
    prob_missing = torch.rand_like(x).to(x.device)
    zeros_values = torch.zeros_like(x).bool()
    input_mask_new = torch.where(prob_missing < missing_rate, zeros_values, input_mask)

    missing_mask = ~input_mask_new
    x[missing_mask] = 0.0
    return x, input_mask_new


def create_windows_from_sequence(data, mask, window_len=12, stride=0):
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
    for i in range(0, data.shape[0] - window_len + 1, stride):
        windows.append(data[i:i + window_len])
        windows_mask.append(mask[i:i + window_len])
    return np.array(windows), np.array(windows_mask)


def adapt_tensor(data, extra_data, edge_index, edge_weights):
    # print(f'''
    # data.shape: {data.shape}
    # extra_data.shape: {extra_data.shape}
    # edge_index.shape: {edge_index.shape}
    # edge_weights.shape: {edge_weights.shape}''')
    num_nodes = data.shape[2]
    expand_weights = torch.ones(num_nodes).to(edge_weights.device)
    edge_weights = torch.cat([edge_weights, expand_weights], dim=0)

    expand_index = torch.stack([
        torch.arange(num_nodes, num_nodes * 2),
        torch.arange(num_nodes)], dim=0
    ).to(edge_weights.device)
    # print(f'expand_index.shape: {expand_index.shape}')
    edge_index = torch.cat([edge_index, expand_index], dim=1)

    data = torch.cat([data, extra_data], dim=2)

    # print(f'''
    # data.shape: {data.shape}
    # edge_index.shape: {edge_index.shape}
    # edge_weights.shape: {edge_weights.shape}''')

    return data, edge_index, edge_weights


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
