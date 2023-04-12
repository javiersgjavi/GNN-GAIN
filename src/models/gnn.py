import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric_temporal.nn.recurrent import MPNNLSTM, A3TGCN2, GConvGRU
from src.utils import init_weights_xavier


class GNN(nn.Module):
    """
    The simple multi-layer perceptron architecture from the original code with a configurable input size.

    Args:
        input_size (int): The size of the input tensor.

    """

    def __init__(self, input_size: int = None, edge_index: torch.Tensor = None, edge_weights: torch.Tensor = None):
        super().__init__()

        # self.gated1 = MPNNLSTM(in_channels=2, hidden_size=1, num_nodes=207, window=12, dropout=0)
        self.gated1 = GConvGRU(in_channels=2, out_channels=1, K=3)
        # self.gated2 = MPNNLSTM(in_channels=414, hidden_size=2, num_nodes=207, window=12, dropout=0)

        # self.gnn_block = gnn.Sequential('x, edge_index, edge_weight', [
        #    (A3TGCN2(in_channels=2, out_channels=1, periods=12, batch_size=128), 'x, edge_index, edge_weight -> x'),
        #   nn.ReLU(),
        #  (A3TGCN2(in_channels=1, out_channels=1, periods=12, batch_size=128), 'x, edge_index, edge_weight -> x'),
        #   #nn.ReLU(),
        #   ]).apply(init_weights_xavier)



        self.edge_index = torch.LongTensor(edge_index)#.to(torch.device('cuda'))
        self.edge_weights = torch.Tensor(edge_weights)#.to(torch.device('cuda'))

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

        #input_tensor = torch.stack([x, input_mask]).permute(1, 2, 3, 0)
        input_tensor = x.reshape(x.shape[0], -1)

        print(f'''
        x: {x.shape}
        input_tensor: {input_tensor.shape}
        ''')

        h = self.gated1(input_tensor, self.edge_index, self.edge_weights)
        print(f'h: {h.shape}')
        imputation = self.fc_block(h)

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
        input_tensor = torch.stack([x, hint_matrix]).reshape(x.shape[0], -1)
        pred = self.d_block(input_tensor)
        pred = pred.reshape(x.shape[0], x.shape[1], -1)
        return pred