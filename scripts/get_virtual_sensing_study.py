import sys
sys.path.append('./')
import argparse
import numpy as np
from src.experiment.studies import VirtualSensingStudy

np.random.seed(2023)


def main(args):
    iterations = args.iterations
    gpu = args.gpu
    input_file = args.input_file
    dataset = args.dataset

    name_input_file = input_file.split('/')[-2]
    folder = f'./results/virtual_{dataset}_{name_input_file}'


    if 'air-36' in dataset:
        masked = [1014, 1031]
    elif 'la' in dataset:
        masked = np.random.randint(207, size=2).tolist()

    random_search = VirtualSensingStudy(
        iterations=iterations,
        gpu=[gpu],
        max_iter_train=5000,
        folder=folder,
        input_file=input_file,
        dataset=dataset,
        masked=masked
    )

    random_search.run()


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='Dataset to use',
        default='la_point',
        type=str)
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
