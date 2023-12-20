import time
import torch
import itertools
import numpy as np
import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.models.g_tigre import GTIGRE
from src.data.traffic import MetrLADataset, PemsBayDataset
from src.data.mimic_iii import MIMICIIIDataset
from src.experiment.base_experiment import BaseExperiment, AverageResults

class ExperimentAblation(BaseExperiment):
    def __init__(self, ablation=None, suffix=None, *args, **kwargs):
        self.ablation = ablation
        super().__init__(*args, **kwargs)
        self.save_file = self.save_file.replace('.csv', f'_{suffix}.csv')
        self.results_file = self.load_file()
        self.exp_name = 'Ablation experiment'

        self.make_architecture_ablation()

    def make_architecture_ablation(self):
        if 'no_bi' in self.ablation:
            self.default_hyperparameters['bi'] = False
        elif 'no_tg' in self.ablation:
            self.default_hyperparameters['use_time_gap_matrix'] = False

    def make_graph_ablation(self, edge_index, edge_weights):

        dtype_w = edge_weights.dtype
        dtype_e = edge_index.dtype

        if self.ablation == 'fc':
            edge_weights = np.ones(edge_weights.shape).astype(dtype_w)
        elif self.ablation == 'nc':
            num_nodes = edge_index.max() + 1
            edge_index_arange = np.arange(num_nodes)
            edge_index = np.array([edge_index_arange, edge_index_arange]).astype(dtype_e)
            edge_weights = np.ones(num_nodes).astype(dtype_w)

        return edge_index, edge_weights

    def prepare_data(self):
        name_dataset = self.dataset.split('_')[0]

        if name_dataset == 'la':
            dm = MetrLADataset(point=True)
        elif name_dataset == 'bay':
            dm = PemsBayDataset(point=True)
        elif name_dataset == 'mimic':
            dm = MIMICIIIDataset()
            
        edge_index, edge_weights = dm.get_connectivity()
        edge_index, edge_weights = self.make_graph_ablation(edge_index, edge_weights)
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        return dm, edge_index, edge_weights, normalizer

    def train_test(self, hyperparameters):
        self.model = GTIGRE(
            model_type=self.model_name,
            input_size=self.dm.input_size(),
            edge_index=self.edge_index,
            edge_weights=self.edge_weights,
            normalizer=self.normalizer,
            params=hyperparameters,
            alpha=hyperparameters['alpha'] if 'alpha' in hyperparameters.keys() else None,
            ablation_gan = True if self.ablation == 'no_gan' else False,
            ablation_reconstruction = True if self.ablation == 'no_reconstruction' else False,
            ablation_loop = True if 'no_loop' in self.ablation else False
        )

        early_stopping = EarlyStopping(monitor='denorm_mse', patience=2, mode='min')
        self.trainer = Trainer(
            max_steps=self.max_iter_train,
            #max_epochs=300,
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
        return results
    

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
       
          