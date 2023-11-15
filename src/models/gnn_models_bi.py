import torch
from typing import Tuple
from torch import nn, Tensor
from tsl.nn.models import RNNEncGCNDecModel, GRUGCNModel
from src.utils import init_weights_xavier, generate_uniform_noise, add_sn


class BaseGNN(nn.Module):
    def __init__(self, edge_index, edge_weights):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weights = edge_weights

    def clip_weights(self, clip_value=0.01):
        for p in self.parameters():
            p.data.clamp_(-clip_value, clip_value)

    def define_mlp_encoder(self, mlp_layers):
        self.decoder_mlp = nn.Sequential()
        input_size = self.output_size*2

        for i, l in enumerate(range(mlp_layers, 1, -1)):
            output_size = int(((l - 1) *  (self.output_size*2)/ mlp_layers) + 1)
            output_size = output_size if output_size > 0 else 1
            self.decoder_mlp.add_module(
                f'linear_{i}',
                nn.Linear(input_size, output_size)
                )
            self.decoder_mlp.add_module(
                f'activation_{i}',
                nn.LeakyReLU()
            )
            input_size = output_size

        self.decoder_mlp.add_module(f'final_linear', nn.Linear(input_size, 1))
        if self.gen:
            self.decoder_mlp.add_module(f'final_activation', nn.Sigmoid())

        self.decoder_mlp.apply(init_weights_xavier)
   

    def bi_forward(self, input_tensor_f, input_tensor_b, edges, weights):
        f_representation = self.model_f(input_tensor_f, edges, weights)
        b_representation = self.model_b(input_tensor_b, edges, weights)

        h = torch.cat([f_representation, torch.flip(b_representation, dims=[1])], dim=-1)
        output = self.decoder_mlp(h)
        return output

    def forward_g(self, x: torch.Tensor, input_mask: torch.Tensor, time_gap_matrix: torch.Tensor) -> Tuple[
        Tensor, Tensor]:
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

        tensors_to_stack_f = [x, input_mask] if not self.time_gap_matrix else [x, input_mask,
                                                                               time_gap_matrix['forward']]
        input_tensor_f = torch.stack(tensors_to_stack_f).permute(1, 2, 3, 0)
        input_tensor_b = torch.flip(input_tensor_f, dims=[1])
        
        imputation = self.bi_forward(input_tensor_f, input_tensor_b, self.edge_index, self.edge_weights).squeeze(dim=-1)

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
        input_tensor_f = torch.stack([x, hint_matrix]).permute(1, 2, 3, 0)
        input_tensor_b = torch.flip(input_tensor_f, dims=[1])
        pred = self.bi_forward(input_tensor_f, input_tensor_b, self.edge_index, self.edge_weights).squeeze(dim=-1)
        return pred

class GRUGCNBI(BaseGNN):

    def __init__(self, args, time_gap_matrix=False):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.time_gap_matrix = time_gap_matrix
        self.hidden_size = int(args['periods'] * args['hidden_size'])
        self.output_size = self.hidden_size//2

        self.model_f = GRUGCNModel(
            exog_size=0,
            input_size=2 if not self.time_gap_matrix else 3,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            horizon=args['periods'],
            activation=args['activation'],
            enc_layers=args['enc_layers'],
            gcn_layers=args['gcn_layers'],
            norm=args['norm'],
        ).apply(init_weights_xavier)

        self.model_b = GRUGCNModel(
            exog_size=0,
            output_size=self.hidden_size//2,
            hidden_size=self.hidden_size,
            horizon=args['periods'],
            activation=args['activation'],
            enc_layers=args['enc_layers'],
            gcn_layers=args['gcn_layers'],
            norm=args['norm'],
        ).apply(init_weights_xavier)

        self.define_mlp_encoder(args['mlp_layers'])

        print(self.decoder_mlp)


class RNNEncGCNDecBI(BaseGNN):

    def __init__(self, args, time_gap_matrix=False, gen=False):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.time_gap_matrix = time_gap_matrix
        self.gen = gen
        self.hidden_size = int(args['periods'] * args['hidden_size'])
        self.output_size = self.hidden_size//2

        self.model_f = RNNEncGCNDecModel(
            exog_size=0,
            input_size=2 if not self.time_gap_matrix else 3,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            horizon=args['periods'],
            rnn_layers=args['rnn_layers'],
            gcn_layers=args['gcn_layers'],
            rnn_dropout=args['rnn_dropout'],
            gcn_dropout=args['gcn_dropout'],
            activation=args['activation'],
            cell_type=args['cell_type']
            ).apply(init_weights_xavier)

        self.model_b = RNNEncGCNDecModel(
            exog_size=0,
            input_size=2 if not self.time_gap_matrix else 3,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            horizon=args['periods'],
            rnn_layers=args['rnn_layers'],
            gcn_layers=args['gcn_layers'],
            rnn_dropout=args['rnn_dropout'],
            gcn_dropout=args['gcn_dropout'],
            activation=args['activation'],
            cell_type=args['cell_type'],
            ).apply(init_weights_xavier)

        self.define_mlp_encoder(args['mlp_layers'])

        self.decoder_mlp.apply(init_weights_xavier)

        print(self.decoder_mlp)