import torch
from typing import Tuple
from torch import nn, Tensor
from tsl.nn.models import RNNEncGCNDecModel, GRUGCNModel
from tsl.nn.blocks.encoders import ConditionalBlock
from src.utils import init_weights_xavier, generate_uniform_noise


class BaseGNN(nn.Module):
    def __init__(self, edge_index, edge_weights):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.model = None

    def define_mlp_stationary(self):
        self.mlp_stationary = nn.Sequential()
        self.mlp_stationary.add_module(f'linear_1', nn.Linear(5, 3))
        self.mlp_stationary.add_module(f'activation_1', nn.ReLU())

        self.mlp_stationary.add_module(f'linear_2', nn.Linear(3, 1))
        self.mlp_stationary.add_module(f'activation_2', nn.Sigmoid())

        self.mlp_stationary.apply(init_weights_xavier)

    def define_mlp_encoder(self, mlp_layers, periods):
        self.decoder_mlp = nn.Sequential()
        input_size = 4

        self.decoder_mlp.add_module(f'linear_1', nn.Linear(4, 2))
        self.decoder_mlp.add_module(f'activation_1', nn.ReLU())

        self.decoder_mlp.add_module(f'linear_2', nn.Linear(2, 1))
        self.decoder_mlp.add_module(f'activation_2', nn.Sigmoid())

        self.decoder_mlp.apply(init_weights_xavier)

    def define_mlp_encoder2(self, mlp_layers, periods):
        self.decoder_mlp = nn.Sequential()
        input_size = 4

        self.decoder_mlp.add_module(f'linear_1', nn.Linear(8, 4))
        self.decoder_mlp.add_module(f'activation_1', nn.ReLU())

        self.decoder_mlp.add_module(f'linear_2', nn.Linear(4, 1))
        self.decoder_mlp.add_module(f'activation_2', nn.Sigmoid())

        self.decoder_mlp.apply(init_weights_xavier)

    '''def define_mlp_encoder(self, mlp_layers, periods):
        self.decoder_mlp = nn.Sequential()
        input_size = 4#int(periods * 2)
        #print(input_size)

        mlp_layers= 2
        for i, l in enumerate(range(mlp_layers, 1, -1)):
            output_size = int(((l - 1) * periods / mlp_layers) + periods)
            output_size = output_size if output_size > 0 else 1
            self.decoder_mlp.add_module(
                f'linear_{i}',
                nn.Linear(input_size, output_size)
            )
            self.decoder_mlp.add_module(
                f'activation_{i}',
                nn.ReLU()
            )
            input_size = output_size

        self.decoder_mlp.add_module(f'final_linear', nn.Linear(input_size, 1))
        self.decoder_mlp.add_module(f'final_activation', nn.Sigmoid())

        self.decoder_mlp.apply(init_weights_xavier)'''


    def bi_forward(self, input_tensor_f, input_tensor_b, edges, weights, exog=None):
        f_representation = self.model_f(input_tensor_f, edges, weights)
        b_representation = self.model_b(input_tensor_b, edges, weights)

        #print(f_representation.shape)

        h = torch.cat([f_representation, b_representation], dim=-1)
        #print(h.shape)
        output = self.decoder_mlp(h)
        #print(output.shape)
        return output
    
    def bi_forward_2(self, input_tensor_f, input_tensor_b, edges, weights, exog=None):
        f_representation = self.model_f.forward(input_tensor_f, edges, weights)
        b_representation = self.model_b.forward(input_tensor_b, edges, weights)

        #print(f_representation.shape)

        h = torch.cat([f_representation, b_representation.flip(dims=[1]), exog], dim=-1)
     
        output = self.decoder_mlp(h)
        #print(output.shape)
        return output
    

    def bi_forward_exog(self, input_tensor_f, input_tensor_b, edges, weights, exog=None):

        f_representation = self.model_f(input_tensor_f, edges, weights).squeeze(dim=-1).permute(0, 2, 1)
        b_representation = self.model_b(input_tensor_b, edges, weights).squeeze(dim=-1).permute(0, 2, 1)

        h = torch.cat([f_representation, b_representation], dim=-1)
        output = self.decoder_mlp(h).unsqueeze(dim=-1)

        if not exog is None:
            #print('output', output.shape)
            #print('exog', exog.shape)
            h2 = torch.cat([output, exog.permute(0,2,1,3)], dim=3)
            #print('h2', h2.shape)
            output = self.mlp_stationary(h2).squeeze(dim=-1).permute(0, 2, 1)
            #print(output.shape)


        return output

    def prepare_inputs(self, x, input_mask, time_gap_matrix):
        tensors_to_stack_f = [x, input_mask] if not self.time_gap_matrix else [x, input_mask,
                                                                               time_gap_matrix['forward']]
        input_tensor_f = torch.stack(tensors_to_stack_f).permute(1, 2, 3, 0)

        tensors_to_stack_b = [x, input_mask] if not self.time_gap_matrix else [x, input_mask,
                                                                               time_gap_matrix['backward']]
        for i, tensor in enumerate(tensors_to_stack_b):
            tensors_to_stack_b[i] = torch.flip(tensor, dims=[1])
        input_tensor_b = torch.stack(tensors_to_stack_b).permute(1, 2, 3, 0)

        return input_tensor_f, input_tensor_b

    def forward_g(self, x: torch.Tensor, input_mask: torch.Tensor, time_gap_matrix: torch.Tensor, exog=None) -> Tuple[
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

        #print(x.shape)
        input_tensor_f, input_tensor_b = self.prepare_inputs(x, input_mask, time_gap_matrix)
        imputation = self.bi_forward_2(input_tensor_f, input_tensor_b, self.edge_index, self.edge_weights, exog=exog).squeeze(dim=-1)

        # Concatenate the original data with the imputed data
        res = input_mask * x + (1 - input_mask) * imputation

        return res, imputation

    def forward_d(self, x: torch.Tensor, hint_matrix: torch.Tensor, exog=None) -> torch.Tensor:
        """
        The forward pass of the discriminator network.

        Args:
            x (torch.Tensor): The input tensor.
            hint_matrix (torch.Tensor): A binary matrix indicating which elements of the input tensor are missing.

        Returns:
            torch.Tensor: The output tensor of the discriminator network.

        """
        input_tensor_f = torch.stack([x, hint_matrix]).permute(1, 2, 3, 0)
        input_tensor_b = torch.stack([torch.flip(x, dims=[1]), torch.flip(hint_matrix, dims=[1])]).permute(1, 2, 3, 0)

        input_tensor_f = torch.concat([input_tensor_f, exog], axis=-1) if not exog is None else input_tensor_f
        input_tensor_b = torch.concat([input_tensor_b, torch.flip(exog, dims=[1])], axis=-1) if not exog is None else input_tensor_f
        
        pred = self.bi_forward(input_tensor_f, input_tensor_b, self.edge_index, self.edge_weights).squeeze(dim=-1)
        return pred

class GRUGCNBI(BaseGNN):

    def __init__(self, args, time_gap_matrix=False):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.time_gap_matrix = time_gap_matrix

        self.model_f = GRUGCNModel(
            exog_size=0,
            input_size=2 if not self.time_gap_matrix else 3,
            output_size=1,
            hidden_size=int(args['periods'] * args['hidden_size']),
            horizon=args['periods'],
            activation=args['activation'],
            enc_layers=args['enc_layers'],
            gcn_layers=args['gcn_layers'],
            norm=args['norm'],
        ).apply(init_weights_xavier)

        self.model_b = GRUGCNModel(
            exog_size=0,
            input_size=2 if not self.time_gap_matrix else 3,
            output_size=1,
            hidden_size=int(args['periods'] * args['hidden_size']),
            horizon=args['periods'],
            activation=args['activation'],
            enc_layers=args['enc_layers'],
            gcn_layers=args['gcn_layers'],
            norm=args['norm'],
        ).apply(init_weights_xavier)

        self.define_mlp_encoder(args['mlp_layers'], args['periods'])

        print(self.decoder_mlp)


class RNNEncGCNDecBI(BaseGNN):

    def __init__(self, args, time_gap_matrix=False, exog=False):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.time_gap_matrix = time_gap_matrix
        self.exog = exog

        input_size = 2
        input_size = input_size + 1 if self.time_gap_matrix else input_size
        #input_size = input_size + 4 if self.exog else input_size

        self.model_f = RNNEncGCNDecModel(
            exog_size=0,
            input_size=input_size,
            output_size=2,
            hidden_size=int(args['periods'] * args['hidden_size']),
            horizon=args['periods'],
            rnn_layers=args['rnn_layers'],
            gcn_layers=args['gcn_layers'],
            rnn_dropout=args['rnn_dropout'],
            gcn_dropout=args['gcn_dropout'],
            activation=args['activation'],
            cell_type=args['cell_type'],
        ).apply(init_weights_xavier)

        self.model_b = RNNEncGCNDecModel(
            exog_size=0,
            input_size=input_size,
            output_size=2,
            hidden_size=int(args['periods'] * args['hidden_size']),
            horizon=args['periods'],
            rnn_layers=args['rnn_layers'],
            gcn_layers=args['gcn_layers'],
            rnn_dropout=args['rnn_dropout'],
            gcn_dropout=args['gcn_dropout'],
            activation=args['activation'],
            cell_type=args['cell_type'],
        ).apply(init_weights_xavier)

        self.define_mlp_encoder(args['mlp_layers'], args['periods'])

        if exog:
            self.define_mlp_encoder2(args['mlp_layers'], args['periods'])
        else:
            self.define_mlp_encoder(args['mlp_layers'], args['periods'])


        print(self.decoder_mlp)