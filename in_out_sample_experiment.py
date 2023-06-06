import itertools
import pandas as pd
import argparse
from src.experiment.experiment import InOutSampleExperiment


def make_summary_dataset(datasets, models):
    columns = [
        'model'
        'category'
        'mae_in',
        'mse_in',
        'rmse_in',
        'denorm_mae_in',
        'denorm_mse_in',
        'denorm_mre_in',
        'denorm_rmse_in',
        'mae_out',
        'mse_out',
        'rmse_out',
        'denorm_mae_out',
        'denorm_mse_out',
        'denorm_mre_out',
        'denorm_rmse_out',
        'params'
    ]

    for dataset in datasets:
        results_path = f'./results/{dataset}/'
        result_file = pd.DataFrame(columns=columns)
        for model in models:
            results_model = pd.read_csv(f'{results_path}{model}_results.csv')
            best_result_in = results_model.iloc[results_model['mae_in'].idxmin()]
            best_result_out = results_model.iloc[results_model['mae_out'].idxmin()]

            row_in = [
                model,
                'in_sample',
                best_result_in['mse_in'],
                best_result_in['mae_in'],
                best_result_in['rmse_in'],
                best_result_in['denorm_mse_in'],
                best_result_in['denorm_mae_in'],
                best_result_in['denorm_rmse_in'],
                best_result_in['denorm_mre_in'],
                best_result_in['mse_out'],
                best_result_in['mae_out'],
                best_result_in['rmse_out'],
                best_result_in['denorm_mse_out'],
                best_result_in['denorm_mae_out'],
                best_result_in['denorm_rmse_out'],
                best_result_in['denorm_mre_out'],
                best_result_in['params']
            ]

            row_out = [
                model,
                'out_sample',
                best_result_out['mse_in'],
                best_result_out['mae_in'],
                best_result_out['rmse_in'],
                best_result_out['denorm_mse_in'],
                best_result_out['denorm_mae_in'],
                best_result_out['denorm_rmse_in'],
                best_result_out['denorm_mre_in'],
                best_result_out['mse_out'],
                best_result_out['mae_out'],
                best_result_out['rmse_out'],
                best_result_out['denorm_mse_out'],
                best_result_out['denorm_mae_out'],
                best_result_out['denorm_rmse_out'],
                best_result_out['denorm_mre_out'],
                best_result_out['params']
            ]
            result_file.loc[len(result_file)] = row_in
            result_file.loc[len(result_file)] = row_out

        result_file.to_csv(f'{results_path}/results.csv', index=False)


def make_summary_general(datasets):
    columns = [
        'dataset',
        'category',
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

        for category in ['in_sample', 'out_sample']:
            category_label = "out" if "out" in category else "in"
            results_category = results_dataset.loc[results_dataset['category'] == category]
            best_result_category = results_category.iloc[
                results_dataset[f'mae_{category_label}'].idxmin()]
            row = [
                dataset,
                category,
                best_result['model'],
                best_result[f'mae_{category_label}'],
                best_result[f'mse_{category_label}'],
                best_result[f'rmse_{category_label}'],
                best_result[f'denorm_mae_{category_label}'],
                best_result[f'denorm_mse_{category_label}'],
                best_result[f'denorm_mre_{category_label}'],
                best_result[f'denorm_rmse_{category_label}'],
                best_result['params']
            ]
            result_file.loc[len(result_file)] = row

    result_file.to_csv(f'./results/results.csv', index=False)


def main(args):
    datasets = args.datasets.split(',')
    models = args.models.split(',')
    iterations = args.iterations
    gpu = args.gpu
    bi = bool(args.bi)
    time_gap = bool(args.time_gap)
    datasets = [f'{dataset}' for dataset in datasets]

    for dataset, model in itertools.product(datasets, models):
        results_path = f'./results_in_out/{dataset}/'
        random_search = InOutSampleExperiment(
            model=model,
            dataset=dataset,
            iterations=iterations,
            results_path=results_path,
            gpu=[gpu],
            max_iter_train=5000,
            bi=bi,
            time_gap=time_gap
        )

        random_search.run()

    make_summary_dataset(datasets, models)
    make_summary_general(datasets)


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        default='air, air-36',
        type=str)
    parser.add_argument(
        '--models',
        help='models to optimize',
        default='grugcn,rnngcn',
        type=str)
    parser.add_argument(
        '--iterations',
        help='number of iterations for the random search',
        default=100,
        type=int)

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
    parser.add_argument(
        '--time_gap',
        help='If the model uses the time_gap matrix',
        choices=[0, 1],
        default='0',
        type=int
    )

    args = parser.parse_args()

    main(args)
