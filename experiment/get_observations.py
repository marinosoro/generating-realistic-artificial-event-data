import ast

from definitions import *


def representation_model(experiment_path):
    experiment_iterations_path = create_path(experiment_path, 'iterations')
    iteration_paths = glob.glob(experiment_iterations_path + '/*')

    all_observations = []
    for iteration in iteration_paths:
        iteration_params = create_path(iteration, 'representation_model_parameters.csv')
        iteration_params = read_csv(iteration_params)

        results_dir = create_path(iteration, 'results')
        result_files = glob.glob(results_dir + '/*')

        for observation in result_files:
            observation_dict = read_csv(observation)
            representation_length = int(observation.split('size_')[1].split('.')[0])
            all_observations.append({
                'accuracy': float(observation_dict['ACCURACY']),
                'representation_length': representation_length,
                'mode': int(iteration_params['MODE']),
                'logs_per_tree': int(iteration_params['LOGS_PER_TREE'])
            })

    return all_observations
