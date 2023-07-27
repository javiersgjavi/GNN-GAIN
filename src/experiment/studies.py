import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from src.data.datasets import DataModule
from src.experiment.experiment import Experiment, ExperimentAblation, VirtualSensingExperiment

class AverageResults:
    def __init__(self, iterations=5, gpu='auto', max_iter_train=5000, folder='results', input_file=None):
        columns = [
            'mae',
            'mse',
            'rmse',
            'denorm_mae',
            'denorm_mse',
            'denorm_mre',
            'denorm_rmse',
        ]

        self.input_file_path = input_file
        self.input_file = pd.read_csv(input_file)
        self.iterations = iterations

        self.folder = folder
        self.gpu = gpu
        self.max_iter_train = max_iter_train

        columns = list(itertools.product(columns, ['mean', 'std']))
        self.columns = [f'{variable}-{suffix}' for variable, suffix in columns] + ['params']

    def get_row_and_name(self, results_path):
        name_dataset = results_path.split('/')[-1].split('.')[0]
        original_row = self.input_file.loc[self.input_file['dataset'] == name_dataset]

        return original_row, name_dataset
        
    def extract_results(self, results_path):
        results_model = pd.read_csv(f'{results_path}')
        original_row, name_dataset = self.get_row_and_name(results_path)

        model = original_row['model'].values[0]
        res = [name_dataset, model]
        for column in self.columns[:-1]:
            variable, suffix = column.split('-')
            if suffix == 'mean':
                value = results_model[variable].mean()
            else:
                value = results_model[variable].std()
            res.append(value)
        res.append(results_model['params'].values[0])
        return res

    def make_summary_dataset(self):
        columns = ['dataset', 'model'] + self.columns
        result_file = pd.DataFrame(columns=columns)

        for file in np.sort(os.listdir(self.folder)):
            if file != 'results.csv' and file.endswith('.csv'):
                results_path = f'./{self.folder}/{file}'
                row = self.extract_results(results_path)
                result_file.loc[len(result_file)] = row

        result_file.to_csv(f'{self.folder}/results.csv', index=False)

    def run(self):
        results_path = f'./{self.folder}'

        for i in range(len(self.input_file)):
            row = self.input_file.iloc[i]
            model = row['model']
            dataset = row['dataset']
            hyperparameters = row['params']
            experiment = Experiment(
                model=model,
                dataset=dataset,
                iterations=self.iterations,
                results_path=results_path,
                gpu=self.gpu,
                max_iter_train=self.max_iter_train,
                default_hyperparameters=hyperparameters,
                save_file=dataset
            )
            experiment.run()

        self.make_summary_dataset()

class AblationStudy(AverageResults):

    def get_row_and_name(self, results_path):
        original_name_dataset = results_path.split('/')[-1].split('.')[0]
        original_name_dataset, ablation  = original_name_dataset.split('_ab_')
        ablation = ablation.split('.')[0]
        name_dataset = f'{original_name_dataset}_{ablation}'
        original_row = self.input_file.loc[self.input_file['dataset'] == original_name_dataset]

        return original_row, name_dataset

    def run(self):
        results_path = f'./{self.folder}'
        for i in range(len(self.input_file)):
            row = self.input_file.iloc[i]
            model = row['model']
            dataset = row['dataset']
            hyperparameters = row['params']

            for ablation in ['no_bi', 'no_tg', 'no_bi_no_tg','fc', 'nc']:
                experiment = ExperimentAblation(
                    model=model,
                    dataset=dataset,
                    iterations=self.iterations,
                    results_path=results_path,
                    gpu=self.gpu,
                    max_iter_train=self.max_iter_train,
                    default_hyperparameters=hyperparameters,
                    save_file=dataset,
                    suffix=f'ab_{ablation}',
                    ablation=ablation
                )
                experiment.run()

        self.make_summary_dataset()        
                
class SensitivityAnalysis(AverageResults):
    def __init__(self, dataset_name=None, missing_percentages=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.missing_percentages = missing_percentages

    def run(self):

        results_path = f'./{self.folder}'
        datasets = self.input_file['dataset'].unique()
        for dataset in datasets:
            if self.dataset_name in dataset:
                row = self.input_file.loc[self.input_file['dataset'] == dataset]
                self.dataset_name = dataset.split('_')[0]

        datasets_to_test = [f'{self.dataset_name}_{missing_percentage / 10}_point' for missing_percentage in
                            self.missing_percentages]
        datasets_to_test.append(f'{self.dataset_name}_0.25_point')
        model = row['model'].values[0]
        hyperparameters = row['params'].values[0]

        for dataset in datasets_to_test:
            experiment = Experiment(
                model=model,
                dataset=dataset,
                iterations=self.iterations,
                results_path=results_path,
                gpu=self.gpu,
                max_iter_train=self.max_iter_train,
                default_hyperparameters=hyperparameters,
                save_file=dataset
            )
            experiment.run()

        self.make_summary_dataset()
        self.create_plot()

    def extract_results(self, results_path):
        results_model = pd.read_csv(f'{results_path}')
        name_dataset = results_path.split('/')[-1].split('.csv')[0]
        res = [name_dataset, self.input_file['model'].values[0]]
        for column in self.columns[:-1]:
            variable, suffix = column.split('-')
            if suffix == 'mean':
                value = results_model[variable].mean()
            else:
                value = results_model[variable].std()
            res.append(value)
        res.append(results_model['params'].values[0])
        return res

    def create_plot(self):
        results = pd.read_csv(f'{self.folder}/results.csv')[['dataset', 'mae-mean']]
        results['dataset'] = results['dataset'].apply(lambda x: x.split('_')[1])
        results['dataset'] = results['dataset'].astype(float)

        sns.set_theme()
        ax = sns.lineplot(x="dataset", y="mae-mean", data=results)
        ax.set(xlabel='Missing percentage', ylabel='MAE')
        plt.savefig(f'{self.folder}/sensitivity_analysis.png')

class VirtualSensingStudy(AverageResults):
    def __init__(self, masked=None, dataset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.masked = masked

    def run(self):
        results_path = f'./{self.folder}'

        row = self.input_file.loc[self.input_file['dataset'] == self.dataset]
        model = row['model'].values[0]
        hyperparameters = row['params'].values[0]

        experiment = VirtualSensingExperiment(
            model=model,
            dataset=self.dataset,
            iterations=self.iterations,
            results_path=results_path,
            gpu=self.gpu,
            max_iter_train=self.max_iter_train,
            default_hyperparameters=hyperparameters,
            save_file=self.dataset,
            masked=self.masked
        )
        predictions = experiment.run()
        model = experiment.model
        dm_v = experiment.dm

        mae_dict = self.obtain_virtual_sensing_errors(predictions, model, dm_v)
    
        self.save_results(mae_dict)

    def save_results(self, mae_dict):
        df = pd.DataFrame(columns = ['masked', 'mae-mean', 'mae-std'])
        for i, column in enumerate(self.masked):
            mean = np.mean(mae_dict[column])
            std = np.std(mae_dict[column])
            df.loc[i] = [column, mean, std]
        df.to_csv(f'{self.folder}/results.csv')

    def obtain_virtual_sensing_errors(self, predictions, model, dm_v):
        real_denorms = []
        n_nodes = model.nodes
        normalizer = model.normalizer
        iterations = predictions.keys()
        id_masked = dm_v.id_to_mask
        batch_size = dm_v.batch_size
        use_time_gap = dm_v.use_time_gap_matrix
        dm_original = DataModule(dataset=self.dataset, batch_size=batch_size, use_time_gap_matrix=use_time_gap)
        dm_original.setup()
        maes_it = {i: [] for i in self.masked}
        mae_column = {i: [] for i in self.masked}
        

        for _, x_real, _, _, _, _ in dm_original.predict_dataloader():

            x_real_denorm = normalizer.inverse_transform(x_real.reshape(-1, n_nodes).detach().cpu())
            real_denorms.append(x_real_denorm)

        for i in iterations:
            batches_iteration = predictions[i]
            for b , prediction in enumerate(batches_iteration):

                for j, column_name in enumerate(self.masked):
                    column_id = id_masked[j]
                    pred = prediction[:, column_id]
                    real= real_denorms[b][:, column_id]

                    mae = mean_absolute_error(real, pred)

                    maes_it[column_name].append(mae)

            for key in maes_it.keys():
                mae_column[key].append(np.mean(maes_it[key]))

        return mae_column





        

