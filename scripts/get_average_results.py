import sys
sys.path.append('./')
import argparse
from src.experiment.base_experiment import AverageResults


def main(args):
    iterations = args.iterations
    gpu = args.gpu
    input_file = args.input_file

    name_input_file = input_file.split('/')[-2]
    folder = f'./results/average_{name_input_file}'

    random_search = AverageResults(
        iterations=iterations,
        gpu=[gpu],
        max_iter_train=5000,
        folder=folder,
        input_file=input_file
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

    args = parser.parse_args()

    main(args)