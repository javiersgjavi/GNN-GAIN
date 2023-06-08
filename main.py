import os
import json
import glob
import torch
import argparse
from src.models.gain import GAIN
from src.data.datasets import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(args):
    # Parse arguments
    dataset = args.dataset
    model = args.model_name
    miss_rate = args.miss_rate
    iterations = args.iterations
    imputation_problem = args.imputation_problem
    early_stopping = args.early_stopping
    dataset = f'{dataset}_{imputation_problem}'
    use_time_gap_matrix = True

    accelerator = 'gpu'

    # Load hyperparameters
    with open('base_params.json') as f:
        params_dict = json.load(f)

    hyperparameters = {'use_time_gap_matrix': use_time_gap_matrix}
    for key in ['batch_size', 'learning_rate', 'activation', 'hidden_size']:
        hyperparameters[key] = params_dict[key]

    hyperparameters = {**hyperparameters, **params_dict[model]}

    # Load data
    dm = DataModule(
        dataset=dataset,
        batch_size=hyperparameters['batch_size'],
        prop_missing=miss_rate,
        use_time_gap_matrix=use_time_gap_matrix
    )

    edge_index, edge_weights = dm.get_connectivity()
    normalizer = dm.get_normalizer()
    dm.setup()

    if accelerator == 'gpu':
        edge_index = torch.from_numpy(edge_index).cuda()
        edge_weights = torch.from_numpy(edge_weights).cuda()

    if early_stopping != 0:
        callbacks = [EarlyStopping(monitor='denorm_mse', patience=early_stopping, mode='min')]
    else:
        callbacks = None

    # Load model
    model = GAIN(
        model_type=model,
        input_size=dm.input_size(),
        edge_index=edge_index,
        edge_weights=edge_weights,
        normalizer=normalizer,
        params=hyperparameters,
    )

    print(f'''
        -------------------------- Experiment --------------------------
        Dataset: {dataset}
        Model: {model}
        Missing rate: {miss_rate}
        Iterations: {iterations}
        Hyperparameters: {hyperparameters}

        Generator: {model.generator}
        Discriminator: {model.discriminator}
        ---------------------------------------------------------------
        ''')

    # Train model
    exp_logger = TensorBoardLogger('reports/logs_experiments', name=dataset)
    trainer = Trainer(
        max_steps=iterations,
        default_root_dir='reports/logs_experiments',
        logger=exp_logger,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)

    # Save model
    files = glob.glob(f'./reports/logs_experiments/{dataset}/*')
    newest = max(files, key=os.path.getctime)
    torch.save(model.state_dict(), f'{newest}/model.pt')


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=['la', 'electric', 'air', 'air-36', 'bay'],
        default='la',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.25,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training iterations',
        default=10000,
        type=int)
    parser.add_argument(
        '--imputation_problem',
        help='type of imputation problem',
        choices=['point', 'block'],
        default='point',
        type=str)
    parser.add_argument(
        '--model',
        help='type of model',
        choices=['grugcn', 'rnngcn', 'stcn', 'ggn', 'dcrnn'],
        default='rnngcn',
        type=str)
    parser.add_argument(
        '--early_stopping',
        help='early stopping',
        default=0,
        type=int
    )

    args = parser.parse_args()

    main(args)
