import os
import torch
import time
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.experiment.base_experiment import BaseExperiment, AverageResults
from src.models.g_tigre import GTIGRE, GTIGRE_DYNAMIC
from src.data.traffic import MetrLADataset, PemsBayDataset
from src.data.mimic_iii import MIMICIIIDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class MissingDataSensitivityExperiment(BaseExperiment):
    def __init__(self, base_noise, trainning_threshold, *args, **kwargs):
        self.base_noise = base_noise
        self.trainning_threshold = trainning_threshold
        super().__init__(*args, **kwargs)
        self.edge_index, self.edge_weights = self.dm.get_connectivity()
        self.normalizer = self.dm.get_normalizer()
        self.dm.setup()
        self.exp_name = 'Sensitivity experiment'

        if self.accelerator == 'gpu':
            self.edge_index = torch.from_numpy(self.edge_index).to(f'cuda:{self.selected_gpu[0]}')
            self.edge_weights = torch.from_numpy(self.edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        #return self.dm, edge_index, edge_weights, normalizer
    
    def train_model(self):
        print(f'[INFO] starting trainning base model')
        candidates = []
        best_candidate = None
        best_denorm_mae = 100000
        results_candidates = []
        for i in tqdm(range(1)):
            self.model = GTIGRE_DYNAMIC(
                model_type=self.model_name,
                input_size=self.dm.input_size(),
                edge_index=self.edge_index,
                edge_weights=self.edge_weights,
                normalizer=self.normalizer,
                params=self.default_hyperparameters,
                alpha=self.default_hyperparameters['alpha'] if 'alpha' in self.default_hyperparameters.keys() else None,
            )

            self.model.set_threshold(self.trainning_threshold)

            early_stopping = EarlyStopping(monitor='denorm_mse', patience=1, mode='min')
            self.trainer = Trainer(
                max_steps=self.max_iter_train,
                default_root_dir='reports/logs_experiments',
                accelerator=self.accelerator,
                devices=self.selected_gpu,
                gradient_clip_val=5.,
                gradient_clip_algorithm='norm',
                callbacks=[early_stopping],
            )
            start_time = time.time()
            self.trainer.fit(self.model, datamodule=self.dm)
            elapsed_time = time.time() - start_time

            results = self.trainer.test(self.model, datamodule=self.dm)[0]
            results['time'] = elapsed_time

            candidates.append(self.model)
            results_candidates.append(results['denorm_mae'])
            if results['denorm_mae'] < best_denorm_mae:
                best_candidate = i
                best_denorm_mae = results['denorm_mae']

            print(f'[INFO] candidates until now: {[np.round(r_c, 2) for r_c in results_candidates]}')

        self.model = candidates[best_candidate]
        print(f'[INFO] error model selected in mae: {results_candidates[best_candidate]}')


    def run_test(self, percentage, threshold):
        if self.save_file.endswith('0.csv'):
            self.save_file = self.save_file.replace(self.save_file[-7:], f'_{int(round(percentage, -1))}.csv')
        else:
            self.save_file = self.save_file.replace('.csv', f'_{int(round(percentage, -1))}.csv')

        self.results_file = self.load_file()

        self.model.set_threshold(threshold)
        
        dm = MetrLADataset(p_noise=threshold, point=True)
        dm.setup()

        for _ in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'{self.exp_name} with {self.model_name} in {self.dataset}'):
            results = self.trainer.test(self.model, datamodule=dm)[0]
            results['time'] = 0
            self.save_results_file(results, self.default_hyperparameters)

class MissingDataSensitivityStudy(AverageResults):
    def __init__(self, dataset_name=None, p_noise=None, thresholds=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thresholds = thresholds
        self.dataset_name = dataset_name
        self.p_noise = p_noise

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
            base_noise=self.p_noise,
            trainning_threshold=self.thresholds[5]
        )
        experiment.train_model()

        for i, p_noise in enumerate(self.thresholds):
            percentage = (i+1)*10
            experiment.run_test(percentage, p_noise)

        self.make_summary_dataset(model) 
        self.create_plot()
        self.create_plot_top()

    def create_plot(self):
        df = pd.read_csv(f'{self.folder}/results.csv')
        res = pd.DataFrame(columns = ['model']+[i for i in range(10, 100, 10)]).set_index('model')
        res.loc['G-TIGRE'] = df['denorm_mae-mean'].values
        res.loc['GRIN'] = [1.87, 1.9, 1.94, 1.98, 2.04, 2.11, 2.22, 2.40, 2.84]
        res.loc['BRITS'] = [2.32, 2.34, 2.36, 2.40, 2.47, 2.57, 2.76, 3.08, 4.02]

        sns.set_theme()
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=res.columns, y=res.loc['G-TIGRE'], label='G-TIGRE', markers=True, linestyle='--', marker='s')
        sns.lineplot(x=res.columns, y=res.loc['GRIN'], label='GRIN', markers=True, linestyle='--', marker='o')
        sns.lineplot(x=res.columns, y=res.loc['BRITS'], label='BRITS', markers=True, linestyle='--', marker='v')
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
        res.loc['G-TIGRE'] = best_results
        res.loc['GRIN'] = [1.87, 1.9, 1.94, 1.98, 2.04, 2.11, 2.22, 2.40, 2.84]
        res.loc['BRITS'] = [2.32, 2.34, 2.36, 2.40, 2.47, 2.57, 2.76, 3.08, 4.02]

        sns.set_theme()
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=res.columns, y=res.loc['G-TIGRE'], label='G-TIGRE', markers=True, linestyle='--', marker='s')
        sns.lineplot(x=res.columns, y=res.loc['GRIN'], label='GRIN', markers=True, linestyle='--', marker='o')
        sns.lineplot(x=res.columns, y=res.loc['BRITS'], label='BRITS', markers=True, linestyle='--', marker='v')
        plt.xlabel('Missing percentage')
        plt.ylabel('MAE')
        plt.savefig(f'{self.folder}/sensitivity_analysis_top.png', dpi=300)
