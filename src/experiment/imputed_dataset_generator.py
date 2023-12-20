import torch
import pickle
import pandas as pd
from tqdm import tqdm

from src.experiment.base_experiment import AverageResults, BaseExperiment
from src.data.mimic_iii import ImputedMIMICIIIDataset, MIMICIIIDatasetToImpute

class ImputeDatasetExperiment(BaseExperiment):
    def run(self):
        models = {} # mae: model
        trainers = {}
        # este for es por cada iteración que hagamos
        for _ in tqdm(range(self.iterations)):
            results = self.train_test(self.default_hyperparameters)
            models[results['mae']] = self.model
            trainers[results['mae']] = self.trainer

        print(f'Best model: {min(models.keys())}')

        return models[min(models.keys())], trainers[min(models.keys())]


class ImputeDatasetStudy:
    def __init__(self, input_file, output_folder, gpu, iterations=5, train_iterations=5000):
        self.input_file = pd.read_csv(input_file)
        self.output_folder = output_folder
        self.train_iterations = train_iterations
        self.iterations = iterations
        self.gpu = gpu

    def run(self):

        # este for es por cada linea del results
        for i in range(len(self.input_file)):
            row = self.input_file.iloc[i]
            experiment = ImputeDatasetExperiment(
                model = row['model'],
                dataset = row['dataset'],
                iterations=self.iterations,
                gpu=[self.gpu],
                max_iter_train=self.train_iterations,
                default_hyperparameters=row['params'],
                results_path=self.output_folder,
            )
            best_model, trainer= experiment.run()

            if 'mimic' in row['dataset']:

                dm = MIMICIIIDatasetToImpute()

                predicted_data = trainer.predict(best_model, datamodule=dm)

                imputed_dataset = ImputedMIMICIIIDataset(
                    predicted_data,
                    self.output_folder
                )

            imputed_dataset.save()

