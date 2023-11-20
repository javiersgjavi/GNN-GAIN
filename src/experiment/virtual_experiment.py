import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class VirtualSensingExperiment(BaseExperiment):
    def __init__(self, masked=None, *args, **kwargs):
        self.masked = masked
        super().__init__(*args, **kwargs)
        self.predictions = {i: None for i in range(self.iterations)}
        self.exp_name = 'Virtual sensing experiment'

    def prepare_data(self):
        dm = VirtualSensingDataModule(dataset=self.dataset, batch_size=self.batch_size,
                                      use_time_gap_matrix=self.time_gap, masked=self.masked)
        edge_index, edge_weights = dm.get_connectivity()
        normalizer = dm.get_normalizer()
        dm.setup()

        if self.accelerator == 'gpu':
            edge_index = torch.from_numpy(edge_index).to(f'cuda:{self.selected_gpu[0]}')
            edge_weights = torch.from_numpy(edge_weights).to(f'cuda:{self.selected_gpu[0]}')

        return dm, edge_index, edge_weights, normalizer

    def get_predictions(self):
        predictions = self.trainer.predict(model=self.model, datamodule=self.dm)
        return predictions

    def run(self):
        for i in tqdm(range(self.results_file.shape[0], self.iterations),
                      desc=f'{self.exp_name} with {self.model_name} in {self.dataset}'):
            _ = self.train_test(self.default_hyperparameters)
            self.predictions[i] = self.get_predictions()

        return self.predictions


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
    