import torch
from torch import nn
from typing import Tuple

from tsl.nn.blocks.encoders import TemporalConvNet, SpatioTemporalConvNet, Transformer, SpatioTemporalTransformerLayer
from tsl.nn.blocks.encoders.recurrent import RNN, MultiRNN, GraphConvRNN
from tsl.nn.blocks.decoders import GCNDecoder

from gnn_models_bi import BaseGNN
from src.utils import init_weights_xavier, generate_uniform_noise

activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
}

encoders = {
    'rnn': RNN,
    'mrnn': MultiRNN,
    'tcn': TemporalConvNet,
    'stcn': SpatioTemporalConvNet,
    'transformer': Transformer,
    'stransformer': SpatioTemporalTransformerLayer,
    'gcrnn': GraphConvRNN
}

class UniModel(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()

        self.encoder = encoders[hyperparameters['encoder_name']](**hyperparameters['encoder'])
        
        self.decoder = GCNDecoder(**hyperparameters['decoder'])

        self.encoder.apply(init_weights_xavier)
        self.decoder.apply(init_weights_xavier)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class BiModel(BaseGNN):
    def __init__(self, args, time_gap_matrix=False):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.time_gap_matrix = time_gap_matrix
        self.output_size_decoder = int(args['periods'] * args['mlp']['hidden_size'])//2

        args['encoder']['input_size'] = 2 if not self.time_gap_matrix else 3
        args['encoder']['exog_size'] = 0
        args['encoder']['hidden_size'] = int(args['periods'] * args['encoder']['hidden_size'])
        args['encoder']['output_size'] = int(args['periods']* args['encoder']['output_size'])

        
        args['decoder']['input_size'] = args['encoder']['output_size']
        args['decoder']['exog_size'] = 0
        args['decoder']['hidden_size'] = int(args['periods'] * args['decoder']['hidden_size'])
        args['decoder']['output_size'] = self.output_size_decoder

        args['mlp']['input_size'] = self.output_size_decoder*2


        self.model_f = UniModel(args)
        self.model_b = UniModel(args)
        self.define_mlp_decoder(args['mlp'])

    def define_mlp_decoder(self, mlp_params):
        self.decoder_mlp = nn.Sequential()
        mlp_layers = mlp_params['n_layers']
        input_size = mlp_params['input_size']
        activation_fnc = activations[mlp_params['activation']]

        for i, l in enumerate(range(mlp_layers, 1, -1)):
            output_size = int(((l - 1) *  (self.output_size)/ mlp_layers) + 1)
            output_size = output_size if output_size > 0 else 1
            self.decoder_mlp.add_module(
                f'linear_{i}',
                nn.Linear(input_size, output_size)
                )
            self.decoder_mlp.add_module(
                f'activation_{i}',
                activation_fnc()
            )
            input_size = output_size

        self.decoder_mlp.add_module(f'final_linear', nn.Linear(input_size, 1))
        if self.gen:
            self.decoder_mlp.add_module(f'final_activation', activations['sigmoid']())

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
