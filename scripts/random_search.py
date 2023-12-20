import sys

sys.path.append('./')
import argparse
import itertools
from src.experiment.params_optimizer_experiment import RandomSearch


def main(args):
    datasets = args.datasets.split(',')
    models = args.models.split(',')
    iterations = args.iterations
    gpu = args.gpu
    folder = args.folder
    bi = bool(args.bi)
    time_gap = bool(args.time_gap)
    prop_missing = args.prop_missing

    imputation_problem = args.imputation_problem
    scenario = args.scenario

    loss_fn = args.loss

    if (imputation_problem is not None) and (scenario is not None):
        raise ValueError('Scenario or imputation problem must be specified, not both')

    if imputation_problem is None and scenario is None:
        folder = f'{folder}_electric'
        labels = prop_missing.split(',')

    elif imputation_problem is not None:
        folder = f'{folder}_point_block'
        labels = imputation_problem.split(',')
        if 'air' in datasets or 'air-36' in datasets:
            raise ValueError('Only la and bay datasets are available for point or block missing experiments')

    elif scenario is not None:
        folder = f'{folder}_in_out'
        labels = scenario.split(',')
        if 'la' in datasets or 'bay' in datasets:
            raise ValueError('Only air and air-36 datasets are available for in and out sample experiments')

    datasets = [f'{dataset}_{label}' for dataset, label in itertools.product(datasets, labels)]

    folder = f'./results/{folder}'
    random_search = RandomSearch(
        models=models,
        datasets=datasets,
        iterations=iterations,
        gpu=[gpu],
        max_iter_train=5000,
        bi=bi,
        time_gap=time_gap,
        folder=folder,
        loss_fn=loss_fn
    )

    random_search.run()


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        help='datasets to optimize separated by commas, e.g. la,bay,air,air-36,electric{missing_prop}',
        default='la,bay,mimic',
        type=str)
    parser.add_argument(
        '--models',
        help='models to optimize',
        default='rnn,mrnn,tcn,stcn,transformer,stransformer,gcrnn',
        type=str)
    parser.add_argument(
        '--iterations',
        help='number of iterations for the random search',
        default=100,
        type=int)
    parser.add_argument(
        '--imputation_problem',
        help='type of imputation problem',
        choices=['point', 'block', 'point,block', 'block,point', None],
        default=None,
        type=str)
    parser.add_argument(
        '--scenario',
        help='type of imputation scenario',
        choices=['in', 'out', 'in,out', 'out,in', None],
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
        default='1',
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
    parser.add_argument(
        '--prop_missing',
        help='Proportion of missing values to test with electric dataset',
        default='0.25', )
    parser.add_argument(
        '--loss',
        help='Loss function to use',
        choices=['base', 'ls', 'ws', None],
        default=None,
    )

    args = parser.parse_args()

    main(args)
