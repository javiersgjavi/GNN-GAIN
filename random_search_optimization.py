import itertools
import pandas as pd
import argparse
from src.experiment.experiment import RandomSearchExperiment


def make_summary_dataset(datasets, models):
    columns = [
        'model',
        'mse',
        'mae',
        'rmse',
        'denorm_mse',
        'denorm_mae',
        'denorm_rmse',
        'denorm_mre',
        'params'
    ]

    for dataset in datasets:
        results_path = f'./results/{dataset}/'
        result_file = pd.DataFrame(columns=columns)
        for model in models:
            results_model = pd.read_csv(f'{results_path}{model}_results.csv')
            best_result = results_model.iloc[results_model['mae'].idxmin()]
            row = [
                model,
                best_result['mse'],
                best_result['mae'],
                best_result['rmse'],
                best_result['denorm_mse'],
                best_result['denorm_mae'],
                best_result['denorm_rmse'],
                best_result['denorm_mre'],
                best_result['params']
            ]
            result_file.loc[len(result_file)] = row

        result_file.to_csv(f'{results_path}/results.csv', index=False)


def make_summary_general(datasets):
    columns = [
        'dataset',
        'model',
        'mae',
        'mse',
        'rmse',
        'denorm_mae',
        'denorm_mse',
        'denorm_mre',
        'denorm_rmse',
        'params'
    ]
    result_file = pd.DataFrame(columns=columns)

    for dataset in datasets:
        results_dataset_path = f'./results/{dataset}/results.csv'
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

    result_file.to_csv(f'./results/results.csv', index=False)


def main(args):
    datasets = args.datasets.split(',')
    models = args.models.split(',')
    iterations = args.iterations
    imputation_problem = args.imputation_problem.split(',')
    gpu = args.gpu
    bi = bool(args.bi)
    datasets = [f'{dataset}_{imputation}' for dataset, imputation in itertools.product(datasets, imputation_problem)]

    for dataset, model in itertools.product(datasets, models):
        results_path = f'./results/{dataset}/'
        random_search = RandomSearchExperiment(
            model=model,
            dataset=dataset,
            iterations=iterations,
            results_path=results_path,
            gpu=[gpu],
            max_iter_train=5000,
            bi=bi
        )

        random_search.run()

    make_summary_dataset(datasets, models)
    make_summary_general(datasets)


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        default='la,air,air-36,bay',
        type=str)
    parser.add_argument(
        '--models',
        help='models to optimize',
        default='grugcn,rnngcn,stcn,ggn,dcrnn',
        type=str)
    parser.add_argument(
        '--iterations',
        help='number of iterations for the random search',
        default=300,
        type=int)
    parser.add_argument(
        '--imputation_problem',
        help='type of imputation problem',
        choices=['point', 'block', 'point,block'],
        default='point',
        type=str)
    parser.add_argument(
        '--gpu',
        help='gpu to use',
        default='0',
        type=int
    )
    parser.add_argument(
        '--bi',
        help='If the model is bidirectional',
        choices=[0, 1],
        default='0',
        type=int
    )

    args = parser.parse_args()

    main(args)
