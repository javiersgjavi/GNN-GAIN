import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from src.data.datasets import DataModule
from src.experiment.experiment import Experiment, ExperimentAblation, VirtualSensingExperiment, MissingDataSensitivityExperiment

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
        
    def extract_results(self, results_path, model=None):
        results_model = pd.read_csv(f'{results_path}')
        original_row, name_dataset = self.get_row_and_name(results_path)

        model = original_row['model'].values[0] if model is None else model
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

    def make_summary_dataset(self, model=None):
        columns = ['dataset', 'model'] + self.columns
        result_file = pd.DataFrame(columns=columns)

        for file in np.sort(os.listdir(self.folder)):
            if file != 'results.csv' and file.endswith('.csv'):
                results_path = f'./{self.folder}/{file}'
                row = self.extract_results(results_path, model)
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

    def make_table_ab(self, name_table, suffix_exp, datasets, problems):

        df = pd.read_csv(f'{self.folder}/results.csv', index_col='dataset')
        ab_table = pd.DataFrame(columns = df.columns)

        for problem, dataset in itertools.product(problems, datasets):
            family_exp = f'{dataset}_{problem}_'
            for row in df.index:
                    if family_exp in row and row.split(family_exp)[1] in suffix_exp:
                        #print(row)
                        ab_table.loc[row] = df.loc[row]

        ab_table.to_csv(f'{self.folder}/results_{name_table}.csv')

    def make_tables_ab(self):
        df = pd.read_csv(f'{self.folder}/results.csv', index_col='dataset')

        datasets = np.unique([row.split('_')[0] for row in df.index])
        problems = np.unique([row.split('_')[1] for row in df.index])

        tables_to_make ={
            'arch': ['no_bi', 'no_tg', 'no_bi_no_tg', 'no_loop', 'no_loop_no_bi'],
            'graph': ['fc', 'nc'],
            'loss': ['no_gan', 'no_reconstruction']
        }

        for table_name in tables_to_make.keys():
            self.make_table_ab(table_name, tables_to_make[table_name], datasets, problems)
        

    def run(self):
        results_path = f'./{self.folder}'
        for i in range(len(self.input_file)):
            row = self.input_file.iloc[i]
            model = row['model']
            dataset = row['dataset']
            hyperparameters = row['params']

            for ablation in ['no_gan', 'no_reconstruction', 'no_bi', 'no_tg', 'no_loop', 'no_loop_no_bi', 'no_bi_no_tg','fc', 'nc']:
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
        self.make_tables_ab()
       
          
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
            print(column, mean, std)
            print(mae_dict[column])
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
    
      
class MissingDataSensitivityStudy(AverageResults):
    def __init__(self, dataset_name=None, p_noises=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.p_noises = p_noises

    def run(self):
        results_path = f'./{self.folder}'

        row = self.input_file.loc[self.input_file['dataset'] == self.dataset_name]
        model = row['model'].values[0]
        hyperparameters = row['params'].values[0]

        experiment = MissingDataSensitivityExperiment(
            model=model,
            dataset=self.dataset_name,
            iterations=self.iterations,
            results_path=results_path,
            gpu=self.gpu,
            max_iter_train=self.max_iter_train,
            default_hyperparameters=hyperparameters,
            save_file=self.dataset_name,
            base_noise=self.p_noises[5]
        )
        experiment.train_model()

        for p_noise in self.p_noises:
            experiment.run_test(p_noise)

        self.make_summary_dataset(model) 
        self.create_plot()
        self.create_plot_top()

    def create_plot(self):
        df = pd.read_csv(f'{self.folder}/results.csv')
        res = pd.DataFrame(columns = ['model']+[i for i in range(10, 100, 10)]).set_index('model')
        res.loc['TG-GAIN'] = df['denorm_mae-mean'].values
        res.loc['GRIN'] = [1.87, 1.9, 1.94, 1.98, 2.04, 2.11, 2.22, 2.40, 2.84]
        res.loc['BRITS'] = [2.32, 2.34, 2.36, 2.40, 2.47, 2.57, 2.76, 3.08, 4.02]

        sns.set_theme()
        sns.lineplot(x=res.columns, y=res.loc['TG-GAIN'], label='TG-GAIN')
        sns.lineplot(x=res.columns, y=res.loc['GRIN'], label='GRIN')
        sns.lineplot(x=res.columns, y=res.loc['BRITS'], label='BRITS')
        plt.xlabel('Missing percentage')
        plt.ylabel('MAE')
        plt.savefig(f'{self.folder}/sensitivity_analysis.png', dpi=300)


    def create_plot_top(self):

        df = pd.read_csv(f'{self.folder}/results.csv')
        best_results = []
        for file in np.sort(os.listdir(self.folder)):
            if file.endswith('.csv') and not file == 'results.csv':
                print(file)
                data_file = pd.read_csv(f'{self.folder}/{file}')
                best_results.append(data_file['denorm_mae'].min())

        res = pd.DataFrame(columns = ['model']+[i for i in range(10, 100, 10)]).set_index('model')
        res.loc['TG-GAIN'] = best_results
        res.loc['GRIN'] = [1.87, 1.9, 1.94, 1.98, 2.04, 2.11, 2.22, 2.40, 2.84]
        res.loc['BRITS'] = [2.32, 2.34, 2.36, 2.40, 2.47, 2.57, 2.76, 3.08, 4.02]

        sns.set_theme()
        plt.figure()
        sns.lineplot(x=res.columns, y=res.loc['TG-GAIN'], label='TG-GAIN')
        sns.lineplot(x=res.columns, y=res.loc['GRIN'], label='GRIN')
        sns.lineplot(x=res.columns, y=res.loc['BRITS'], label='BRITS')
        plt.xlabel('Missing percentage')
        plt.ylabel('MAE')
        plt.savefig(f'{self.folder}/sensitivity_analysis_top.png', dpi=300)
