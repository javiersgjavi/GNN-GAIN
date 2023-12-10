import torch
from torch import nn
from typing import Tuple

from tsl.nn.blocks.encoders import TemporalConvNet, SpatioTemporalConvNet, Transformer, SpatioTemporalTransformerLayer
from tsl.nn.blocks.encoders.recurrent import RNN, GraphConvRNN#, MultiRNN
from tsl.nn.blocks.decoders import GCNDecoder

from src.models.gnn_models_bi import BaseGNN
from src.utils import init_weights_xavier, generate_uniform_noise, round_to_nearest_divisible, apply_spectral_norm

activations = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'silu': nn.SiLU,
    'selu': nn.SELU,
    'leaky_relu': nn.LeakyReLU,
}

encoders = {
    'rnn': RNN,
    #'mrnn': MultiRNN,
    'tcn': TemporalConvNet, 
    'stcn': SpatioTemporalConvNet, #falla
    'transformer': Transformer,
    'stransformer': SpatioTemporalTransformerLayer, #falla
    'gcrnn': GraphConvRNN 
}

class UniModel(nn.Module):
    def __init__(self, hyperparameters, d = False):
        super().__init__()
        self.d = d
        self.name = hyperparameters['encoder_name']

        structure = hyperparameters['discriminator' if self.d else 'generator']
        self.encoder = encoders[hyperparameters['encoder_name']](**structure['encoder'])
        
        self.decoder = GCNDecoder(**structure['decoder'])

        self.encoder.apply(init_weights_xavier)
        self.decoder.apply(init_weights_xavier)

        if self.d and hyperparameters['loss_fn'] == 'ws':
            self.encoder = apply_spectral_norm(self.encoder)
            self.decoder = apply_spectral_norm(self.decoder)



    def forward(self, x, edges, weights):

        #print(f'input shape: {x.shape}')
        x = self.encoder(x) if self.name != 'stcn' else self.encoder(x, edges, weights)
        #print(f'encoder output shape: {x.shape}')
        x = self.decoder(x, edges, weights)
        #print(f'decoder output shape: {x.shape}\n')
        return x

class BiModel(BaseGNN):
    def __init__(self, args, time_gap_matrix=False, d=False):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.args = args
        print(self.args)
        self.d = d
        self.loss = self.args['loss_fn']
        self.critic = self.d and self.loss in ['ws', 'ls']
        self.time_gap_matrix = time_gap_matrix
        self.output_size_decoder = int(args['periods'] * args['generator']['mlp']['hidden_size'])//2
        
        self.param_cleaner('generator')
        self.param_cleaner('discriminator')

        self.model_f = UniModel(self.args, self.d)
        self.model_b = UniModel(self.args, self.d)

        self.define_mlp_decoder(self.args['generator']['mlp'])

        print(self.model_f)
        print(self.decoder_mlp)

    def param_cleaner(self, model):
        encoder_name = self.args['encoder_name']
        in_features = 2 if not self.time_gap_matrix else 3
        
        self.args[model]['decoder']['hidden_size'] = int(self.args['periods'] * self.args[model]['decoder']['hidden_size'])
        self.args[model]['decoder']['output_size'] = self.output_size_decoder
        self.args[model]['decoder']['horizon'] = self.args['periods']

        self.args[model]['mlp']['input_size'] = self.output_size_decoder*2

        if encoder_name == 'rnn':
            self.args[model]['encoder']['input_size'] = in_features
            self.args[model]['encoder']['hidden_size'] = int(self.args['periods'] * self.args[model]['encoder']['hidden_size'])
            self.args[model]['encoder']['output_size'] = int(self.args['periods'] * self.args[model]['encoder']['output_size'])
            self.args[model]['encoder']['exog_size'] = 0

            self.args[model]['decoder']['input_size'] = self.args[model]['encoder']['output_size']

        elif encoder_name == 'tcn':
            self.args[model]['encoder']['input_channels'] = in_features
            self.args[model]['encoder']['hidden_channels'] = int(self.args['periods'] * self.args[model]['encoder']['hidden_channels'])
            self.args[model]['encoder']['output_channels'] = int(self.args['periods'] * self.args[model]['encoder']['output_channels'])

            self.args[model]['decoder']['input_size'] = self.args[model]['encoder']['output_channels']

        elif encoder_name == 'stcn':
            self.args[model]['encoder']['input_size'] = in_features
            self.args[model]['encoder']['output_size'] = int(self.args['periods'] * self.args[model]['encoder']['output_size'])

            self.args[model]['decoder']['input_size'] = self.args[model]['encoder']['output_size']

        elif encoder_name == 'transformer':
            self.args[model]['encoder']['input_size'] = in_features
            hidden_size = int(self.args['periods'] * self.args[model]['encoder']['hidden_size'])
            self.args[model]['encoder']['hidden_size'] = round_to_nearest_divisible(hidden_size, self.args[model]['encoder']['n_heads'])

            self.args[model]['encoder']['ff_size'] = int(self.args['periods'] * self.args['encoder']['ff_size'])
            self.args[model]['encoder']['output_size'] = int(self.args['periods'] * self.args['encoder']['output_size'])

            self.args[model]['decoder']['input_size'] = self.args[model]['encoder']['output_size']

        print('------- FINAL ARGS -------')
        for k, v in self.args.items():
            print(f'{k}: {v}')

    def define_mlp_decoder(self, mlp_params):
        self.decoder_mlp = nn.Sequential()
        mlp_layers = mlp_params['n_layers']
        input_size = mlp_params['input_size']
        activation_fnc = activations[mlp_params['activation']]
        ouput_size_final = 24

        for i, l in enumerate(range(mlp_layers, 1, -1)):
            output_size = int(((l - 1) *  (ouput_size_final)/ mlp_layers) + 1)
            output_size = output_size if output_size > 0 else 1
            self.decoder_mlp.add_module(
                f'linear_{i}',
                nn.Linear(input_size, output_size)
                )
            self.decoder_mlp.add_module(
                f'activation_{i}',
                activation_fnc()
            )
            self.decoder_mlp.add_module(
                f'dropout_{i}',
                nn.Dropout(mlp_params['dropout'])
            )
            input_size = output_size

        self.decoder_mlp.add_module(f'final_linear', nn.Linear(input_size, 1))

        if not self.critic:
            self.decoder_mlp.add_module(f'final_activation', activations['sigmoid']())

        self.decoder_mlp.apply(init_weights_xavier)

        if self.d and self.loss == 'ws':
            self.decoder_mlp = apply_spectral_norm(self.decoder_mlp)
   

    def bi_forward(self, input_tensor_f, input_tensor_b, edges, weights):
        #print('Forward --------------------------\n')
        f_representation = self.model_f(input_tensor_f, edges, weights)
        b_representation = self.model_b(input_tensor_b, edges, weights)

        h = torch.cat([f_representation, torch.flip(b_representation, dims=[1])], dim=-1)
        #print(f'concatenated shape: {h.shape}')
        output = self.decoder_mlp(h)
        #print(f'output shape: {output.shape}\n')
        return output
    
    def forward_g(self, x: torch.Tensor, input_mask: torch.Tensor, time_gap_matrix: torch.Tensor):
        """
        The forward pass of the generator network.

        Args:
            x (torch.Tensor): The input tensor.
            input_mask (torch.Tensor): A binary mask tensor indicating missing values.

        Returns:
            torch.Tensor: The output tensor of the generator network.
            torch.Tensor: The imputed tensor.

        """
        #print('GENERATOR --------------------------\n')
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
        #print('DISCRIMINATOR --------------------------\n')
        input_tensor_f = torch.stack([x, hint_matrix]).permute(1, 2, 3, 0)
        input_tensor_b = torch.flip(input_tensor_f, dims=[1])
        pred = self.bi_forward(input_tensor_f, input_tensor_b, self.edge_index, self.edge_weights).squeeze(dim=-1)
        return pred
