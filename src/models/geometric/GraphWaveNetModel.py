import torch
from typing import Tuple
from torch import nn, Tensor
from tsl.nn.models import GraphWaveNetModel
from src.utils import init_weights_xavier, generate_uniform_noise


class GNN(nn.Module):

    def __init__(self, periods, nodes, edge_index, edge_weights, batch_size):
        super().__init__()

        self.model = GraphWaveNetModel(
            input_size=2,
            hidden_size=nodes,
            output_size=1,
            horizon=periods,
            exog_size=0,
            n_nodes=nodes
        ).apply(init_weights_xavier)

        self.edge_index = edge_index
        self.edge_weights = edge_weights

    def forward_g(self, x: torch.Tensor, input_mask: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """
        The forward pass of the generator network.

        Args:
            x (torch.Tensor): The input tensor.
            input_mask (torch.Tensor): A binary mask tensor indicating missing values.

        Returns:
            torch.Tensor: The output tensor of the generator network.
            torch.Tensor: The imputed tensor.

        """
        noise_matrix = generate_uniform_noise(tensor_like=x)

        # Concatenate the input tensor with the noise matrix
        x = input_mask * x + (1 - input_mask) * noise_matrix

        input_tensor = torch.stack([x, input_mask]).permute(1, 2, 3, 0)
        imputation = self.model(input_tensor, self.edge_index, self.edge_weights).squeeze(dim=-1)

        imputation = torch.sigmoid(imputation)

        # Concatenate the original data with the imputed data
        res = input_mask * x + (1 - input_mask) * imputation


        return res, imputation

    def forward_d(self, x: torch.Tensor, hint_matrix: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the discriminator network.

        Args:
            x (torch.Tensor): The input tensor.
            hint_matrix (torch.Tensor): A binary matrix indicating which elements of the input tensor are missing.

        Returns:
            torch.Tensor: The output tensor of the discriminator network.

        """
        input_tensor = torch.stack([x, hint_matrix]).permute(1, 2, 3, 0)
        pred = self.model(input_tensor, self.edge_index, self.edge_weights).squeeze(dim=-1)
        return torch.sigmoid(pred)
