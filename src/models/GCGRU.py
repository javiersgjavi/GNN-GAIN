import torch
from torch import nn
from torch_geometric import nn as gnn
from tsl.nn.layers import GraphConvGRUCell
from src.utils import init_weights_xavier


class GNN(nn.Module):

    def __init__(self, periods, nodes, edge_index, edge_weights, batch_size):
        super().__init__()

        self.gnn_layer = GraphConvGRUCell(input_size=24, hidden_size=12)

        self.fc_block = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.Sigmoid()
        ).apply(init_weights_xavier)

        self.edge_index = torch.IntTensor(edge_index).to(torch.int64)
        self.edge_weights = edge_weights

    def forward(self, x):
        res = torch.zeros(x.shape[0], x.shape[1], x.shape[2]//2).to(x.device)
        zeros = torch.zeros(x.shape[1], 12).to(x.device)
        for i in range(x.shape[0]):
            h = self.gnn_layer(x[i], h=zeros, edge_index=self.edge_index)
            h = torch.functional.F.relu(h)
            h = self.fc_block(h)
            res[i] = h

        return res

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

        input_tensor = torch.stack([x, input_mask]).permute(1, 3, 2, 0)
        input_tensor = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        imputation = self.forward(input_tensor).permute(0, 2, 1)

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
        input_tensor = torch.stack([x, hint_matrix]).permute(1, 3, 2, 0)
        input_tensor = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        pred = self.forward(input_tensor).permute(0, 2, 1)
        return pred