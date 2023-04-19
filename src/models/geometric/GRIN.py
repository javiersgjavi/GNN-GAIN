import torch
from torch import nn
from torch_geometric import nn as gnn
from tsl.nn.models import GRINModel
from src.utils import init_weights_xavier, adapt_tensor


class GNN(nn.Module):

    def __init__(self, periods, nodes, edge_index, edge_weights, batch_size):
        super().__init__()

        self.model = GRINModel(
            input_size=2,
            hidden_size=12,
            n_nodes=6,
            exog_size=0,
            n_layers=1,
            embedding_size=12,
        ).apply(init_weights_xavier)

        self.edge_index = edge_index
        self.edge_weights = edge_weights

    def forward(self, x):
        x = self.gnn_layer(x, self.edge_index)
        x = self.fc_block(x)
        return x

    def forward_g(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the generator network.

        Args:
            x (torch.Tensor): The input tensor.
            input_mask (torch.Tensor): A binary mask tensor indicating missing values.

        Returns:
            torch.Tensor: The output tensor of the generator network.

        """
        noise_matrix = torch.distributions.uniform.Uniform(0, 0.01).sample(x.shape).to(x.device)

        # Concatenate the input tensor with the noise matrix
        x = input_mask * x + (1 - input_mask) * noise_matrix

        input_tensor = torch.stack([x, input_mask]).permute(1, 2, 3, 0)
        imputation = self.model(input_tensor, self.edge_index, self.edge_weights).squeeze(dim=-1)

        imputation = torch.nn.functional.sigmoid(imputation)

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
        pred = torch.nn.functional.sigmoid(pred)
        return pred
