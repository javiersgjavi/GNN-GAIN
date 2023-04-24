import torch
from typing import Tuple
from torch import nn, Tensor
from src.utils import init_weights_xavier, generate_uniform_noise


class RNN(nn.Module):

    def __init__(self, periods, nodes):
        super().__init__()

        self.gru1 = nn.GRU(input_size=periods, hidden_size=nodes, batch_first=True, dropout=0.0, num_layers=1, bidirectional=True).apply(init_weights_xavier)
        self.gru2 = nn.GRU(input_size=periods, hidden_size=nodes, batch_first=True, dropout=0.0, num_layers=1).apply(init_weights_xavier)
        self.fc = nn.Linear(6, 6).apply(init_weights_xavier)

    def forward(self, x):
        h = nn.functional.relu(self.gru1(x)[0])
        h = nn.functional.relu(self.gru2(h)[0])
        y = nn.functional.sigmoid(self.fc(h))
        return y

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

        input_tensor = torch.cat([x, input_mask], dim=-1)
        imputation = self.forward(input_tensor)

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
        input_tensor = torch.cat([x, hint_matrix], dim=-1)
        pred = self.forward(input_tensor)
        return pred
