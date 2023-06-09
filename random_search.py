import argparse
import itertools
from src.experiment.experiment import RandomSearch


def main(args):
    datasets = args.datasets.split(',')
    models = args.models.split(',')
    iterations = args.iterations
    gpu = args.gpu
    folder = args.folder
    bi = bool(args.bi)
    time_gap = bool(args.time_gap)

    imputation_problem = args.imputation_problem
    scenario = args.scenario

    if (imputation_problem is not None) and (scenario is not None):
        raise ValueError('Scenario or imputation problem must be specified, not both')

    elif imputation_problem is not None:
        folder = f'{folder}_point_block'
        labels = imputation_problem.split(',')
        if 'air' in datasets or 'air-36' in datasets:
            raise ValueError('Only la and bay datasets are available for point or block missing experiments')

    else:
        folder = f'{folder}_in_out'
        labels = scenario.split(',')
        if 'la' in datasets or 'bay' in datasets:
            raise ValueError('Only air and air-36 datasets are available for in and out sample experiments')

    datasets = [f'{dataset}_{label}' for dataset, label in itertools.product(datasets, labels)]

    random_search = RandomSearch(
        models=models,
        datasets=datasets,
        iterations=iterations,
        gpu=[gpu],
        max_iter_train=5000,
        bi=bi,
        time_gap=time_gap,
        folder=folder
    )

    random_search.run()


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        help='datasets to optimize separated by commas, e.g. la,bay,air,air-36',
        default='la,bay',
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
        '--imputation_problem',
        help='type of imputation problem',
        choices=['point', 'block', 'point,block', None],
        default=None,
        type=str)
    parser.add_argument(
        '--scenario',
        help='type of imputation scenario',
        choices=['in', 'out', 'in,out', None],
        default=None,
        type=str)
    parser.add_argument(
        '--gpu',
        help='gpu to use',
        default='0',
        type=int)
    parser.add_argument(
        '--bi',
        help='If the model is bidirectional',
        choices=[0, 1],
        default='0',
        type=int)
    parser.add_argument(
        '--time_gap',
        help='If the model uses the time_gap matrix',
        choices=[0, 1],
        default='0',
        type=int)
    parser.add_argument(
        '--folder',
        help='Path to save the results',
        default='results', )

    args = parser.parse_args()

    main(args)
