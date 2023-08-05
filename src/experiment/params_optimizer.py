import json
import itertools
import pandas as pd
from numpy.random import randint, uniform, choice
from src.experiment.experiment import RandomSearchExperiment


def randint_close_interval(low, high, size=None):
    return randint(low, high + 1, size=size)


class RandomSearchLoader:
    def __init__(self, model_name, n_iter, bi=False):
        self.n_iter = n_iter
        self.model_name = model_name
        self.bi = bi

        file = 'params_random_search.json'

        with open(f'src/experiment/{file}') as f:
            self.params_grid = json.load(f)

        self.random_params = self.load_params_grid(n_iter)

    def load_params_grid(self, n_iter):
        params_dict = {
            'batch_size': [self.params_grid['batch_size'] for _ in range(n_iter)],
            'learning_rate': 10 ** uniform(*self.params_grid['log_learning_rate'], size=n_iter),
            'activation': choice(self.params_grid['activation'], size=n_iter),
            'hidden_size': uniform(*self.params_grid['hidden_size'], size=n_iter),
            'alpha': uniform(*self.params_grid['alpha'], size=n_iter).astype(int),
        }

        if self.bi:
            params_dict['mlp_layers'] = randint_close_interval(*self.params_grid['mlp_layers'], size=n_iter)

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

        elif self.model_name == 'dcrnn':
            params_dict['kernel_size'] = randint_close_interval(*params_grid_model['enc_layers'], size=n_iter)
            params_dict['n_layers'] = randint_close_interval(*params_grid_model['gcn_layers'], size=n_iter)
            params_dict['dropout'] = uniform(*params_grid_model['rnn_dropout'], size=n_iter)

        return params_dict

    def get_params(self, i):
        if i == self.n_iter:
            raise StopIteration
        params_iteration = {k: v[i] for k, v in self.random_params.items()}
        return params_iteration


class RandomSearch:

    def __init__(self, models=None, datasets=None, iterations=100, gpu='auto', max_iter_train=5000, bi=None,
                 time_gap=None, folder='results'):

        self.columns = [
            'mae',
            'mse',
            'rmse',
            'denorm_mae',
            'denorm_mse',
            'denorm_mre',
            'denorm_rmse',
            'params'
        ]

        self.models = models
        self.datasets = datasets
        self.iterations = iterations

        self.folder = folder
        self.gpu = gpu
        self.max_iter_train = max_iter_train
        self.bi = bi
        self.time_gap = time_gap

    def make_summary_dataset(self, datasets, models):
        columns = ['model'] + self.columns

        for dataset in datasets:
            results_path = f'./{self.folder}/{dataset}/'
            result_file = pd.DataFrame(columns=columns)
            for model in models:
                results_model = pd.read_csv(f'{results_path}{model}_results.csv')
                best_result = results_model.iloc[results_model['mae'].idxmin()]
                row = [
                    model,
                    best_result['mse'],
                    best_result['mae'],
                    best_result['rmse'],
                    best_result['denorm_mae'],
                    best_result['denorm_mse'],
                    best_result['denorm_mre'],
                    best_result['denorm_rmse'],
                    best_result['params']
                ]
                result_file.loc[len(result_file)] = row

            result_file.to_csv(f'{results_path}/results.csv', index=False)

    def make_summary_general(self, datasets):
        columns = ['dataset', 'model'] + self.columns
        result_file = pd.DataFrame(columns=columns)

        for dataset in datasets:
            results_dataset_path = f'./{self.folder}/{dataset}/results.csv'
            results_dataset = pd.read_csv(results_dataset_path)

            best_result = results_dataset.iloc[results_dataset['mae'].idxmin()]
            row = [
                dataset,
                best_result['model'],
                best_result['mae'],
                best_result['mse'],
                best_result['rmse'],
                best_result['denorm_mae'],
                best_result['denorm_mse'],
                best_result['denorm_mre'],
                best_result['denorm_rmse'],
                best_result['params']
            ]
            result_file.loc[len(result_file)] = row

        result_file.to_csv(f'./{self.folder}/results.csv', index=False)

    def run(self):
        for dataset, model in itertools.product(self.datasets, self.models):
            results_path = f'./{self.folder}/{dataset}/'

            param_loader = RandomSearchLoader(model, self.iterations, self.bi)
            
            random_search = RandomSearchExperiment(
                model=model,
                dataset=dataset,
                iterations=self.iterations,
                results_path=results_path,
                gpu=self.gpu,
                max_iter_train=self.max_iter_train,
                bi=self.bi,
                time_gap=self.time_gap,
                save_file=f'{model}_results',
                param_loader=param_loader
            )

            random_search.run()
        self.make_summary_dataset(self.datasets, self.models)
        self.make_summary_general(self.datasets)
