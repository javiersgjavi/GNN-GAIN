import json
import itertools
import pandas as pd
from tqdm import tqdm
from numpy.random import randint, uniform, choice
from src.experiment.base_experiment import BaseExperiment, print_dict


def randint_close_interval(low, high, size=None):
    return randint(low, high + 1, size=size)


class RandomSearchParamLoader:
    def __init__(self, model_name, n_iter, bi=False, loss_fn=None):
        self.n_iter = n_iter
        self.model_name = model_name
        self.bi = bi
        self.loss_fn = loss_fn

        file = 'params_random_search_2.json'

        with open(f'src/experiment/{file}') as f:
            self.params_grid = json.load(f)

        self.random_params_g = self.load_params_grid(n_iter)
        self.random_params_d = self.load_params_grid(n_iter)

    def load_params_grid(self, n_iter):
        params_dict = {
            'loss_fn': [self.loss_fn if self.loss_fn is not None else choice(self.params_grid['loss']) for _ in range(n_iter)],
            'learning_rate': 10 ** uniform(*self.params_grid['log_learning_rate'], size=n_iter),
            'alpha': [100 for _ in range(n_iter)], #'alpha': uniform(*self.params_grid['alpha'], size=n_iter).astype(int),
            'encoder': {
                'activation': choice(self.params_grid['encoder_activation'], size=n_iter),
                'dropout': uniform(*self.params_grid['encoder_dropout'], size=n_iter)
            },
            'decoder': {
                'hidden_size': uniform(*self.params_grid['decoder_hidden_size'], size=n_iter),
                'activation': choice(self.params_grid['decoder_activation'], size=n_iter),
                'n_layers': randint_close_interval(*self.params_grid['decoder_layers'], size=n_iter),
                'dropout': uniform(*self.params_grid['encoder_dropout'], size=n_iter),
                #'cat_states_layers': choice(self.params_grid['gcrnn']['cat_states_layers'], size=n_iter),
            },
            'mlp':{
                'hidden_size': uniform(*self.params_grid['mlp_hidden_size'], size=n_iter),
                'n_layers': randint_close_interval(*self.params_grid['mlp_layers'], size=n_iter),
                'dropout': uniform(*self.params_grid['mlp_dropout'], size=n_iter),
                'activation': choice(self.params_grid['mlp_activation'], size=n_iter),
            },   
        }

        if self.model_name == 'rnn':
            params_dict['encoder_type'] = ['rnn' for _ in range(n_iter)]
            params_dict['encoder']['n_layers'] = randint_close_interval(*self.params_grid['rnn']['encoder_layers'], size=n_iter)
            params_dict['encoder']['cell'] = choice(self.params_grid['rnn']['cell'], size=n_iter)
            params_dict['encoder']['hidden_size'] = uniform(*self.params_grid['rnn']['hidden_size'], size=n_iter)
            params_dict['encoder']['output_size'] = uniform(*self.params_grid['encoder_output'], size=n_iter)
    
        elif self.model_name == 'mrnn':
            params_dict['encoder_type'] = ['mrnn' for _ in range(n_iter)]
            params_dict['encoder']['n_layers'] = randint_close_interval(*self.params_grid['mrnn']['encoder_layers'], size=n_iter)
            params_dict['encoder']['cell'] = choice(self.params_grid['mrnn']['cell'], size=n_iter)
            params_dict['encoder']['hidden_size'] = uniform(*self.params_grid['mrnn']['hidden_size'], size=n_iter)
            params_dict['encoder']['output_size'] = uniform(*self.params_grid['encoder_output'], size=n_iter)

        elif self.model_name == 'tcn':
            params_dict['encoder_type'] = ['tcn' for _ in range(n_iter)]
            params_dict['encoder']['n_layers'] = randint_close_interval(*self.params_grid['tcn']['encoder_layers'], size=n_iter)
            params_dict['encoder']['kernel_size'] = randint_close_interval(*self.params_grid['tcn']['kernel_size'], size=n_iter)
            params_dict['encoder']['dilation'] = randint_close_interval(*self.params_grid['tcn']['dilation'], size=n_iter)
            params_dict['encoder']['hidden_channels'] = uniform(*self.params_grid['tcn']['encoder_channels'], size=n_iter)
            params_dict['encoder']['output_channels'] = uniform(*self.params_grid['encoder_output'], size=n_iter)
            params_dict['encoder']['weight_norm'] = choice(self.params_grid['tcn']['weight_norm'], size=n_iter)

        elif self.model_name == 'stcn':
            params_dict['encoder_type'] = ['stcn' for _ in range(n_iter)]
            params_dict['encoder']['temporal_convs'] = choice(self.params_grid['stcn']['temporal_convs'], size=n_iter)
            params_dict['encoder']['temporal_kernel_size'] = randint_close_interval(*self.params_grid['stcn']['temporal_kernel_size'], size=n_iter)
            params_dict['encoder']['spatial_convs'] = choice(self.params_grid['stcn']['spatial_convs'], size=n_iter)
            params_dict['encoder']['spatial_kernel_size'] = randint_close_interval(*self.params_grid['stcn']['spatial_kernel_size'], size=n_iter)
            params_dict['encoder']['dilation'] = randint_close_interval(*self.params_grid['stcn']['dilation'], size=n_iter)
            params_dict['encoder']['output_size'] = uniform(*self.params_grid['encoder_output'], size=n_iter)
            params_dict['encoder']['gated'] = choice(self.params_grid['stcn']['gated'], size=n_iter)

        elif self.model_name == 'transformer':
            params_dict['encoder_type'] = ['transformer' for _ in range(n_iter)]
            params_dict['encoder']['n_heads'] = randint_close_interval(*self.params_grid['transformer']['n_heads'], size=n_iter)
            params_dict['encoder']['causal'] = choice(self.params_grid['transformer']['causal'], size=n_iter)
            params_dict['encoder']['ff_size'] = randint_close_interval(*self.params_grid['transformer']['ff_size'], size=n_iter)
            params_dict['encoder']['hidden_size'] = uniform(*self.params_grid['transformer']['hidden_size'], size=n_iter)
            params_dict['encoder']['output_size'] = uniform(*self.params_grid['encoder_output'], size=n_iter)
            params_dict['encoder']['activation'] = choice(self.params_grid['encoder_activation'], size=n_iter)
            params_dict['encoder']['n_layers'] = randint_close_interval(*self.params_grid['transformer']['n_layers'], size=n_iter)
            params_dict['encoder']['axis'] = ['time' for _ in range(n_iter)]

        elif self.model_name == 'stransformer':
            params_dict['encoder_type'] = ['transformer' for _ in range(n_iter)]
            params_dict['encoder']['n_heads'] = randint_close_interval(*self.params_grid['transformer']['n_heads'], size=n_iter)
            params_dict['encoder']['axis'] = choice(self.params_grid['transformer']['axis'], size=n_iter)
            params_dict['encoder']['casual'] = choice(self.params_grid['transformer']['casual'], size=n_iter)
            params_dict['encoder']['ff_size'] = randint_close_interval(*self.params_grid['transformer']['cell'], size=n_iter)

        return params_dict

    def get_params(self, i):
        if i == self.n_iter:
            raise StopIteration
        
        params_iteration = {
            'loss_fn': self.random_params_g['loss_fn'][i],
            'learning_rate': self.random_params_g['learning_rate'][i],
            'alpha': self.random_params_g['alpha'][i],
            'generator':{
                'encoder': {k: v[i] for k, v in self.random_params_g['encoder'].items()},
                'decoder': {k: v[i] for k, v in self.random_params_g['decoder'].items()},
                'mlp': {k: v[i] for k, v in self.random_params_g['mlp'].items()},
                },
            'discriminator':{
                'encoder': {k: v[i] for k, v in self.random_params_d['encoder'].items()},
                'decoder': {k: v[i] for k, v in self.random_params_d['decoder'].items()},
                'mlp': {k: v[i] for k, v in self.random_params_d['mlp'].items()},
                },
        }
        return params_iteration

class RandomSearchExperiment(BaseExperiment):
    def __init__(self, bi=False, param_loader=None, *args, **kwargs):
        self.bi = bi
        self.params_loader = param_loader

        super().__init__( *args, **kwargs)

        self.exp_name = 'Random search experiment'

    def train_test(self, hyperparameters):
        hyperparameters['use_time_gap_matrix'] = self.time_gap
        hyperparameters['bi'] = self.bi
        return super().train_test(hyperparameters)

    def run(self):
        remaining_iterations = self.iterations - self.results_file.shape[0]
        i = 0
        with tqdm(total=remaining_iterations, desc=f'{self.exp_name} with {self.model_name} in {self.dataset}') as pbar:
            while i < remaining_iterations:
                try:
                    hyperparameters = self.params_loader.get_params(i)
                    results = self.train_test(hyperparameters)
                    self.save_results_file(results, hyperparameters)
                    i += 1
                    pbar.update(1)
                    
                except KeyboardInterrupt as e:
                    raise e

                except Exception as e:
                    print(e)
                    
        pbar.close()

        for i in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'{self.exp_name} with {self.model_name} in {self.dataset}'):
            hyperparameters = self.params_loader.get_params(i)
            print_dict(hyperparameters, self.max_iter_train)
            results = self.train_test(hyperparameters)
            self.save_results_file(results, hyperparameters)

class RandomSearch:

    def __init__(self, models=None, datasets=None, iterations=100, gpu='auto', max_iter_train=5000, bi=None,
                 time_gap=None, folder='results', loss_fn=None):

        self.columns = [
            'mae',
            'mse',
            'rmse',
            'denorm_mae',
            'denorm_mse',
            'denorm_mre',
            'denorm_rmse',
            'time',
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
        self.loss_fn = loss_fn

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
                    best_result['time'],
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
                best_result['time'],
                best_result['params']
            ]
            result_file.loc[len(result_file)] = row

        result_file.to_csv(f'./{self.folder}/results.csv', index=False)

    def run(self):
        for dataset, model in itertools.product(self.datasets, self.models):
            results_path = f'./{self.folder}/{dataset}/'

            param_loader = RandomSearchParamLoader(model, self.iterations*5, self.bi, self.loss_fn)
            
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


