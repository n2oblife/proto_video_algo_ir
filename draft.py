from itertools import product

# Define the parameters and their possible values
# params = {
#     'param1': [0.1, 0.01, 0.001, 0.0001, 0.00001],
#     'param2': [1, 2, 3],
#     'param3': [True, False]
# }
params = {}
# Generate all combinations of parameter values
param_combinations = list(product(*params.values()))

# Print all combinations
for combination in param_combinations:
    current_params = dict(zip(params.keys(), combination))
    param_str = '_'.join(f"{k}_{v}" for k, v in current_params.items())
    print(f"combination : {current_params} \nparam_str : {param_str}")
