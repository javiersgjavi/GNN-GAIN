# How to execute the code

1. Give execution permissions to setup.sh: ```chmod +x setup.sh```
2. Launch setup.sh: ```./setup.sh```
3. Run parameter optimization: ```python ./scripts/random_search.py --folder experiment```
4. Get average results of the best hyperparameters found: ```python ./scripts/get_average_results.py --input_file ./resutls/experiment/results.csv```
5. Obtain ablation results with the best hyperparameters found: ```python ./scripts/get_ablation_study.py --input_file ./resutls/experiment/results.csv```
6. Make sensitivity analysis with the best hyperparameters found: ```python ./scripts/get_missing_rate_sensitivity_study.py --input_file ./resutls/experiment/results.csv```
