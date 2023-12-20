import sys
sys.path.append('./')
import argparse
from src.experiment.sensitivity_experiment import MissingDataSensitivityStudy


def main(args):
    iterations = args.iterations
    gpu = args.gpu
    input_file = args.input_file
    dataset = args.dataset

    name_input_file = input_file.split('/')[-2]
    folder = f'results/missing_{dataset}_{name_input_file}'

    if dataset == 'la_point':
        thresholds = [0.02, 0.12, 0.23, 0.34, 0.44, 0.56, 0.66, 0.78, 0.89]
        start_p_noises = 0.001

    random_search = MissingDataSensitivityStudy(
        dataset_name=dataset,
        iterations=iterations,
        gpu=[gpu],
        max_iter_train=5000,
        folder=folder,
        input_file=input_file,
        p_noise=start_p_noises,
        thresholds=thresholds,
    )

    random_search.run()


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--iterations',
        help='number of iterations for the random search',
        default=5,
        type=int)
    parser.add_argument(
        '--gpu',
        help='gpu to use',
        default='0',
        type=int)
    parser.add_argument(
        '--input_file',
        help='Path of the file to find the hyperparameters',
        default=None)
    parser.add_argument(
        '--dataset',
        help='Dataset to run the sensitivity analysis',
        default='la_point',
    )

    args = parser.parse_args()

    main(args)
