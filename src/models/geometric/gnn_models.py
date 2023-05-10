import torch
from typing import Tuple
from torch import nn, Tensor
from tsl.nn.models import GatedGraphNetworkModel, RNNEncGCNDecModel, GRUGCNModel, STCNModel
from src.utils import init_weights_xavier, generate_uniform_noise


class BaseGNN(nn.Module):
    def __init__(self, edge_index, edge_weights):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.model = None

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


class STCN(BaseGNN):

    def __init__(self, args):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.model = STCNModel(
            exog_size=0,
            input_size=2,
            output_size=1,
            hidden_size=args['periods'],
            horizon=args['periods'],
            activation=args['activation'],
            ff_size=int(args['nodes'] * args['hidden_size']),
            temporal_kernel_size=args['temporal_kernel_size'],
            spatial_kernel_size=args['spatial_kernel_size'],
            n_layers=args['n_layers'],
        ).apply(init_weights_xavier)


class GRUGCN(BaseGNN):

    def __init__(self, args):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.model = GRUGCNModel(
            exog_size=0,
            input_size=2,
            output_size=1,
            hidden_size=int(args['periods'] * args['hidden_size']),
            horizon=args['periods'],
            activation=args['activation'],
            enc_layers=args['enc_layers'],
            gcn_layers=args['gcn_layers'],
            norm=args['norm'],
        ).apply(init_weights_xavier)


class RNNEncGCNDec(BaseGNN):

    def __init__(self, args):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.model = RNNEncGCNDecModel(
            exog_size=0,
            input_size=2,
            output_size=1,
            hidden_size=int(args['periods'] * args['hidden_size']),
            horizon=args['periods'],
            rnn_layers=args['rnn_layers'],
            gcn_layers=args['gcn_layers'],
            rnn_dropout=args['rnn_dropout'],
            gcn_dropout=args['gcn_dropout'],
            activation=args['activation'],
            cell_type=args['cell_type'],
        ).apply(init_weights_xavier)


class GatedGraphNetwork(BaseGNN):

    def __init__(self, args):
        """AVISO: Este modelo no usa los pesos, simplemente recibe los vertices"""
        super().__init__(edge_index=args['edge_index'], edge_weights=None)

        self.model = GatedGraphNetworkModel(
            exog_size=0,
            input_size=2,
            output_size=1,
            hidden_size=int(args['periods'] * args['hidden_size']),
            input_window_size=args['periods'],
            horizon=args['periods'],
            n_nodes=args['nodes'],
            enc_layers=args['enc_layers'],
            gnn_layers=args['gnn_layers'],
            full_graph=args['full_graph']
        ).apply(init_weights_xavier)
