import argparse
from src.experiment.experiment import SensitivityAnalysis


def main(args):
    iterations = args.iterations
    gpu = args.gpu
    input_file = args.input_file
    dataset = args.dataset

    name_input_file = input_file.split('/')[-2]
    folder = f'sensitivity_{dataset}_{name_input_file}'

    random_search = SensitivityAnalysis(
        dataset_name=dataset,
        iterations=iterations,
        gpu=[gpu],
        max_iter_train=5000,
        folder=folder,
        input_file=input_file,
        missing_percentages=range(1, 10)
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
        default='electric',
    )

    args = parser.parse_args()

    main(args)
