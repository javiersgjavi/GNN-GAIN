from torch import nn

from tsl.nn.blocks.encoders import TemporalConvNet, SpatioTemporalConvNet, Transformer, SpatioTemporalTransformerLayer
from tsl.nn.blocks.encoders.recurrent import RNN, MultiRNN, GraphConvRNN
from tsl.nn.blocks.decoders import GCNDecoder

from gnn_models_bi import BaseGNN
from src.utils import init_weights_xavier


encoders = {
    'rnn': RNN,
    'mrnn': MultiRNN,
    'tcn': TemporalConvNet,
    'stcn': SpatioTemporalConvNet,
    'transformer': Transformer,
    'stransformer': SpatioTemporalTransformerLayer,
    'gcrnn': GraphConvRNN
}

def rnn_encoder(hyperparameters):
    return RNN(**hyperparameters)

def multi_rnn_encoder(hyperparameters):
    return MultiRNN(
        input_size='input_size',
        hidden_size='hidden_size',
        output_size='hidden_size',
        n_layers='rnn_layers',
        dropout='rnn_dropout',
        bidirectional='bidirectional',
        activation='activation',
    )

def tcn_encoder(hyperparameters):
    return TemporalConvNet(
        input_size='input_size',
        hidden_size='hidden_size',
        output_size='hidden_size',
        n_layers='tcn_layers',
        dropout='tcn_dropout',
        activation='activation',
    )

def stcn_encoder(hyperparameters):
    return SpatioTemporalConvNet(
        input_size='input_size',
        hidden_size='hidden_size',
        output_size='hidden_size',
        n_layers='stcn_layers',
        dropout='stcn_dropout',
        activation='activation',
    )

def transformer_encoder(hyperparameters):
    return Transformer()

def stransformer_encoder(hyperparameters):
    return SpatioTemporalTransformerLayer()

def gcrnn_encoder(hyperparameters):
    return GraphConvRNN(
        input_size='input_size',
        hidden_size='hidden_size',
        output_size='hidden_size',
        n_layers='gcrnn_layers',
        dropout='gcrnn_dropout',
        activation='activation',
    )


def define_encoder(hyperparameters):

    encoder_type = hyperparameters['encoder']
    if encoder_type == 'rnn':
        encoder = rnn_encoder(hyperparameters)

    elif encoder_type == 'mrnn':
        encoder = multi_rnn_encoder(hyperparameters)

    elif encoder_type == 'tcn':
        encoder = tcn_encoder(hyperparameters)

    elif encoder_type == 'stcn':
        encoder = stcn_encoder(hyperparameters)

    elif encoder_type == 'transformer':
        encoder = transformer_encoder(hyperparameters)

    elif encoder_type == 'stransformer':
        encoder = stransformer_encoder(hyperparameters)

    elif encoder_type == 'gcrnn':
        encoder = gcrnn_encoder(hyperparameters)

    return encoder

class Model(nn.Module):
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


    