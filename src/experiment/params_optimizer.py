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

        file = 'params_random_search_2.json'

        with open(f'src/experiment/{file}') as f:
            self.params_grid = json.load(f)

        self.random_params = self.load_params_grid(n_iter)

    def load_params_grid(self, n_iter):
        params_dict = {
            'learning_rate': 10 ** uniform(*self.params_grid['log_learning_rate'], size=n_iter),
            'alpha': [100 for _ in range(n_iter)], #'alpha': uniform(*self.params_grid['alpha'], size=n_iter).astype(int),
            'encoder': {
                'hidden_size': uniform(*self.params_grid['encoder_hidden_size'], size=n_iter),
                'activation': choice(self.params_grid['encoder_activation'], size=n_iter),
                'n_layers': randint_close_interval(*self.params_grid['encoder_layers'], size=n_iter),
                'dropout': uniform(*self.params_grid['encoder_dropout'], size=n_iter),
                'output': uniform(*self.params_grid['encoder_output'], size=n_iter),
            },
            'decoder': {
                'input_size': uniform(*self.params_grid['encoder_output'], size=n_iter),
                'hidden_size': uniform(*self.params_grid['decoder_hidden_size'], size=n_iter),
                'activation': choice(self.params_grid['decoder_activation'], size=n_iter),
                'n_layers': randint_close_interval(*self.params_grid['encoder_layers'], size=n_iter),
                'dropout': uniform(*self.params_grid['encoder_dropout'], size=n_iter),
                'cat_states_layers': choice(self.params_grid['gcrnn']['cat_states_layers'], size=n_iter),
            },
            'mlp':{
                'hidden_size': uniform(*self.params_grid['mlp_hidden_size'], size=n_iter),
                'n_layers': randint_close_interval(*self.params_grid['mlp_layers'], size=n_iter),
                'dropout': uniform(*self.params_grid['mlp_dropout'], size=n_iter),
                'activation': choice(self.params_grid['mlp_activation'], size=n_iter),
            },
            
        }

        if self.bi:
            params_dict['mlp_layers'] = randint_close_interval(*self.params_grid['mlp_layers'], size=n_iter)

        params_grid_model = self.params_grid[self.model_name]

        if self.model_name == 'rnn':
            params_dict['encoder_type'] = ['rnn' for _ in range(n_iter)]
            params_dict['encoder']['cell'] = choice(self.params_grid_model['rnn']['cell'], size=n_iter)

        elif self.model_name == 'mrnn':
            params_dict['encoder_type'] = ['mrnn' for _ in range(n_iter)]
            params_dict['encoder']['cell'] = choice(self.params_grid_model['mrnn']['cell'], size=n_iter)

        elif self.model_name == 'tcn':
            params_dict['encoder_type'] = ['tcn' for _ in range(n_iter)]
            params_dict['encoder']['kernel_size'] = randint_close_interval(*self.params_grid_model['tcn']['kernel_size'], size=n_iter)
            params_dict['encoder']['dilation'] = randint_close_interval(*self.params_grid_model['tcn']['dilation'], size=n_iter)
            params_dict['encoder']['stride'] = randint_close_interval(*self.params_grid_model['tcn']['stride'], size=n_iter)

        elif self.model_name == 'stcn':
            params_dict['encoder_type'] = ['stcn' for _ in range(n_iter)]
            params_dict['encoder']['temporal_kernel_size'] = randint_close_interval(*self.params_grid_model['stcn']['temporal_kernel_size'], size=n_iter)
            params_dict['encoder']['spatial_kernel_size'] = randint_close_interval(*self.params_grid_model['stcn']['spatial_kernel_size'], size=n_iter)
            params_dict['encoder']['dilation'] = randint_close_interval(*self.params_grid_model['stcn']['dilation'], size=n_iter)
            params_dict['encoder']['stride'] = randint_close_interval(*self.params_grid_model['stcn']['stride'], size=n_iter)

        elif self.model_name == 'transformer':
            params_dict['encoder_type'] = ['transformer' for _ in range(n_iter)]
            params_dict['encoder']['n_heads'] = randint_close_interval(*self.params_grid_model['transformer']['n_heads'], size=n_iter)
            params_dict['encoder']['axis'] = choice(self.params_grid_model['transformer']['axis'], size=n_iter)
            params_dict['encoder']['casual'] = choice(self.params_grid_model['transformer']['casual'], size=n_iter)
            params_dict['encoder']['ff_size'] = randint_close_interval(*self.params_grid_model['transformer']['cell'], size=n_iter)

        elif self.model_name == 'stransformer':
            params_dict['encoder_type'] = ['transformer' for _ in range(n_iter)]
            params_dict['encoder']['n_heads'] = randint_close_interval(*self.params_grid_model['transformer']['n_heads'], size=n_iter)
            params_dict['encoder']['axis'] = choice(self.params_grid_model['transformer']['axis'], size=n_iter)
            params_dict['encoder']['casual'] = choice(self.params_grid_model['transformer']['casual'], size=n_iter)
            params_dict['encoder']['ff_size'] = randint_close_interval(*self.params_grid_model['transformer']['cell'], size=n_iter)

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
