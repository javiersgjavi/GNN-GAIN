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
            results_model = pd.read_csv(f'{results_path}/{model}_results.csv')
            best_result = results_model.iloc[results_model['mse'].idxmin()]
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

        result_file.to_csv(f'{results_path}/{dataset}/results.csv', index=False)


def make_summary_general(datasets):
    columns = [
        'dataset',
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
    results_path = f'./results/results.csv'
    result_file = pd.DataFrame(columns=columns)

    for dataset in datasets:
        results_dataset_path = f'./results/{dataset}/results.csv'
        results_dataset = pd.read_csv(results_dataset_path, index_col='Unnamed: 0')

        best_result = results_dataset.iloc[results_dataset['mse'].idxmin()]
        row = [
            dataset,
            best_result['model'],
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


def main(args):
    datasets = args.datasets.split(',')
    models = args.models.split(',')
    iterations = args.iterations
    imputation_problem = args.imputation_problem
    datasets = [f'{dataset}_{imputation_problem}' for dataset in datasets]

    for dataset, model in zip(datasets, models):
        results_path = f'./results/{dataset}/'
        random_search = RandomSearchExperiment(
            model=model,
            dataset=dataset,
            iterations=iterations,
            results_path=results_path
        )

        random_search.run()

    make_summary_dataset(datasets, models)
    make_summary_general(datasets)


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        choices=['la', 'electric', 'air', 'air-36', 'bay'],
        default='la,air,air-36,bay',
        type=str)
    parser.add_argument(
        '--models',
        help='models to optimize',
        default='stcn,grugcn,rnngcn,ggn',
        type=str)
    parser.add_argument(
        '--iterations',
        help='number of training iterations',
        default=1000,
        type=int)
    parser.add_argument(
        '--imputation_problem',
        help='type of imputation problem',
        choices=['point', 'block'],
        default='point',
        type=str)

    args = parser.parse_args()

    main(args)
