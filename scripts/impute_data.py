import sys

sys.path.append('./')
import argparse
import itertools
from src.experiment.imputed_dataset_generator import ImputeDatasetStudy 

def main(args):
    input_file = args.input_file
    gpu = args.gpu
    iterations = args.iterations

    output_folder = f'./imputed_data/'

    impute_experiment = ImputeDatasetStudy(
        input_file=input_file,
        output_folder=output_folder,
        gpu=gpu,
        iterations=iterations
    )

    impute_experiment.run()
    

if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        help='gpu to use',
        default='0',
        type=int)
    parser.add_argument(
        '--input_file',
        help='Path to save the results',
        default='results', ) 
    parser.add_argument(
        '--iterations',
        help='Path to save the results',
        default=5,
        type=int)

    args = parser.parse_args()

    main(args)