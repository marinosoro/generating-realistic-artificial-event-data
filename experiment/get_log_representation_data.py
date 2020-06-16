import ast
import random

from definitions import *
from experiment.config import *


def main(experiment_path):
    experiment_iterations_path = create_path(experiment_path, 'iterations')
    iteration_paths = glob.glob(experiment_iterations_path + '/*')

    log_representation_data_for_lengths = {}

    representation_lengths = REPRESENTATION_MODEL_PARAMETER_SETUP['REPRESENTATION_LENGTH']['value']
    for length in representation_lengths:
        log_representation_data_for_lengths[length] = []

    for iteration_path in iteration_paths:
        iteration_parameters = read_csv(create_path(iteration_path, "representation_model_parameters.csv"))
        iteration_log_threshold = int(iteration_parameters['TREES_PER_POPULATION']) * int(iteration_parameters['LOGS_PER_TREE'])
        prediction_parameter = 'max_repeat'
        iteration_prediction_value = float(read_csv(create_path(iteration_path, "populations", "with_loops", "parameters.csv"))[prediction_parameter])

        representation_lengths = ast.literal_eval(iteration_parameters['REPRESENTATION_LENGTH'])
        for representation_length in representation_lengths:
            log_representations = parse_log_vectors(create_path(iteration_path, "models", "representation_model_size_{}__log_representation_matrix.txt".format(representation_length)))
            for log_id in range(len(log_representations)):
                has_loops = log_id + 1 <= iteration_log_threshold
                log_representation_data = {
                    "representation": log_representations[log_id],
                    "has_loops": has_loops,
                    "max_repeat_index": indexed_value(prediction_parameter, iteration_prediction_value) if has_loops else -1
                }
                log_representation_data_for_lengths[representation_length].append(log_representation_data)
    return log_representation_data_for_lengths


def get_baseline(experiment_path, parameter, include_zero=True):
    log_representation_data = main(experiment_path)[32]
    parameter_range = POPULATION_PARAMETER_SETUP[parameter]['value'] if POPULATION_PARAMETER_SETUP[parameter]['type'] == 'range' else None

    if parameter == 'max_repeat':
        occurrences = []

        if parameter_range is not None:
            occurrences_range = parameter_range[1] - parameter_range[0]
            if include_zero:
                occurrences_range += 1
            occurrences_range = range(occurrences_range)
            for i in occurrences_range:
                occurrences.append(0)

        for data in log_representation_data:
            index = int(data['max_repeat_index'])
            if index >= 0:
                occurrences[index] += 1

        for index in range(len(occurrences)):
            print('index: {} - occurrences: {}'.format(index, occurrences[index]))

        largest_occurrence = max(occurrences)
        largest_occurrence_index = occurrences.index(largest_occurrence)
        print('max repeat largest occurrence: [{}: {}]'.format(largest_occurrence_index, largest_occurrence))
        print('sum occurrencs: ', sum(occurrences))
        baseline = largest_occurrence / sum(occurrences)

        return baseline

    if parameter == 'loop':
        occurrences = [0, 0]

        for data in log_representation_data:
            has_loops = data['has_loops']
            if has_loops:
                occurrences[1] += 1
            else:
                occurrences[0] += 1

        largest_occurrence = max(occurrences)
        baseline = largest_occurrence / sum(occurrences)

        return baseline


def for_iteration(iteration_path):
    log_representations = {}

    iteration_parameters = read_csv(create_path(iteration_path, "representation_model_parameters.csv"))
    representation_lengths = ast.literal_eval(iteration_parameters['REPRESENTATION_LENGTH'])

    for length in representation_lengths:
        log_representations[length] = []

    for length in representation_lengths:
        log_representations[length] = parse_log_vectors(create_path(iteration_path, "models", "representation_model_size_{}__log_representation_matrix.txt".format(length)))

    return log_representations


def normalized_value(parameter, original_value):
    parameter_range = POPULATION_PARAMETER_SETUP[parameter]['value'] if POPULATION_PARAMETER_SETUP[parameter]['type'] == 'range' else None
    if parameter_range is not None:
        normalized = ((original_value - parameter_range[0]) / (parameter_range[1] - parameter_range[0]))
        return normalized
    return original_value


def indexed_value(parameter, original_value):
    parameter_range = POPULATION_PARAMETER_SETUP[parameter]['value'] if POPULATION_PARAMETER_SETUP[parameter]['type'] == 'range' else None
    if parameter_range is not None:
        indexed = original_value - parameter_range[0]
        return indexed
    return original_value


def parse_log_vectors(from_file):
    vectors = []
    for row in open(from_file):
        row = row[1:-2]  # Drop '[' and ']\n'
        row.strip()
        row = row.split(', ')
        row = [float(el) for el in row if el != '']
        vectors.append(row)

    return vectors


def scrambled(log_representation_data):
    dest = log_representation_data
    random.shuffle(dest)
    return dest


def get_training_test_data__loop(log_representation_data):
    scrambled_data = scrambled(log_representation_data)
    training_test_threshold = int(len(scrambled_data)*PREDICTION_TRAINING_PERCENTAGE/100)
    training_section = scrambled_data[:training_test_threshold]
    test_section = scrambled_data[training_test_threshold:]

    training = {
        'input': [],
        'output': []
    }
    test = {
        'input': [],
        'output': [],
    }

    for representation_object in training_section:
        training['input'].append(representation_object['representation'])
        training['output'].append(representation_object['has_loops'])
    for representation_object in test_section:
        test['input'].append(representation_object['representation'])
        test['output'].append(representation_object['has_loops'])

    return training, test


def get_training_test_data__max_repeat(log_representation_data):
    scrambled_data = scrambled(log_representation_data)
    training_test_threshold = int(len(scrambled_data)*PREDICTION_TRAINING_PERCENTAGE/100)
    training_section = scrambled_data[:training_test_threshold]
    test_section = scrambled_data[training_test_threshold:]

    training = {
        'input': [],
        'output': []
    }
    test = {
        'input': [],
        'output': [],
    }

    for representation_object in training_section:
        if representation_object['max_repeat_index'] >= 0:
            training['input'].append(representation_object['representation'])
            training['output'].append(representation_object['max_repeat_index'])
    for representation_object in test_section:
        if representation_object['max_repeat_index'] >= 0:
            test['input'].append(representation_object['representation'])
            test['output'].append(representation_object['max_repeat_index'])

    return training, test
