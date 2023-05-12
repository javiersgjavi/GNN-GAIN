import json
from numpy.random import randint, uniform, choice


def randint_close_interval(low, high, size=None):
    return randint(low, high + 1, size=size)


class RandomSearchLoader:
    def __init__(self, model_name, n_iter):
        self.n_iter = n_iter
        self.model_name = model_name

        with open('src/experiment/base_params.json') as f:
            self.params_grid = json.load(f)

        self.random_params = self.load_params_grid(n_iter)

    def load_params_grid(self, n_iter):
        params_dict = {
            'batch_size': [self.params_grid['batch_size'] for _ in range(n_iter)],
            'learning_rate': 10 ** uniform(*self.params_grid['log_learning_rate'], size=n_iter),
            'activation': choice(self.params_grid['activation'], size=n_iter),
            'hidden_size': uniform(*self.params_grid['hidden_size'], size=n_iter),
        }

        params_grid_model = self.params_grid[self.model_name]

        if self.model_name == 'ggn':
            params_dict['enc_layers'] = randint_close_interval(*params_grid_model['enc_layers'], size=n_iter)
            params_dict['gnn_layers'] = randint_close_interval(*params_grid_model['gnn_layers'], size=n_iter)

            params_dict['full_graph'] = choice(params_grid_model['full_graph'], size=n_iter)

        elif self.model_name == 'rnngcn':
            params_dict['rnn_layers'] = randint_close_interval(*params_grid_model['rnn_layers'], size=n_iter)
            params_dict['gcn_layers'] = randint_close_interval(*params_grid_model['gcn_layers'], size=n_iter)

            params_dict['rnn_dropout'] = uniform(*params_grid_model['rnn_dropout'], size=n_iter)
            params_dict['gcn_dropout'] = uniform(*params_grid_model['gcn_dropout'], size=n_iter)

            params_dict['cell_type'] = choice(params_grid_model['cell_type'], size=n_iter)

        elif self.model_name == 'stcn':

            params_dict['temporal_kernel_size'] = randint_close_interval(*params_grid_model['temporal_kernel_size'],
                                                                         size=n_iter)
            params_dict['spatial_kernel_size'] = randint_close_interval(*params_grid_model['spatial_kernel_size'],
                                                                        size=n_iter)
            params_dict['n_layers'] = randint_close_interval(*params_grid_model['n_layers'], size=n_iter)

        elif self.model_name == 'grugcn':

            params_dict['enc_layers'] = randint_close_interval(*params_grid_model['enc_layers'], size=n_iter)
            params_dict['gcn_layers'] = randint_close_interval(*params_grid_model['gcn_layers'], size=n_iter)
            params_dict['norm'] = choice(params_grid_model['norm'], size=n_iter)

        return params_dict

    def get_params(self, i):
        if i == self.n_iter:
            raise StopIteration
        params_iteration = {k: v[i] for k, v in self.random_params.items()}
        return params_iteration
