import random
from definitions import *
from experiment.config import *


def main():
    set_path = prepare_representation_mode_set()

    iterations_path = create_path(set_path, 'iterations')
    for iteration in range(ITERATION_COUNT):
        corpora = []
        iteration_path = prepare_iteration_set(iterations_path)
        # Define population for iteration
        population_parameters = {}
        for parameter in POPULATION_PARAMETER_SETUP.keys():
            if POPULATION_PARAMETER_SETUP[parameter]['type'] == 'range':
                RANGE = POPULATION_PARAMETER_SETUP[parameter]['value']
                LOWER_BOUND = RANGE[0] if not isinstance(RANGE[0], str) else population_parameters[RANGE[0]]
                UPPER_BOUND = RANGE[1] if not isinstance(RANGE[1], str) else population_parameters[RANGE[1]]
                population_parameters[parameter] = random.randrange(LOWER_BOUND, UPPER_BOUND, 1) if not isinstance(LOWER_BOUND, float) else round(random.uniform( LOWER_BOUND, LOWER_BOUND), 1)
            elif POPULATION_PARAMETER_SETUP[parameter]['type'] == 'choice':
                CHOICE = POPULATION_PARAMETER_SETUP[parameter]['value']
                population_parameters[parameter] = CHOICE[random.randrange(0, len(CHOICE), 1)]
            elif POPULATION_PARAMETER_SETUP[parameter]['type'] == 'fixed':
                population_parameters[parameter] = POPULATION_PARAMETER_SETUP[parameter]['value']

        representation_model_params = {}
        for parameter in REPRESENTATION_MODEL_PARAMETER_SETUP.keys():
            if REPRESENTATION_MODEL_PARAMETER_SETUP[parameter]['type'] == 'range':
                RANGE = REPRESENTATION_MODEL_PARAMETER_SETUP[parameter]['value']
                LOWER_BOUND = RANGE[0] if not isinstance(RANGE[0], str) else representation_model_params[RANGE[0]]
                UPPER_BOUND = RANGE[1] if not isinstance(RANGE[1], str) else representation_model_params[RANGE[1]]
                representation_model_params[parameter] = random.randrange(LOWER_BOUND, UPPER_BOUND, 1) if not isinstance(LOWER_BOUND, float) else round(random.uniform( LOWER_BOUND, LOWER_BOUND), 1)
            elif REPRESENTATION_MODEL_PARAMETER_SETUP[parameter]['type'] == 'choice':
                CHOICE = REPRESENTATION_MODEL_PARAMETER_SETUP[parameter]['value']
                representation_model_params[parameter] = CHOICE[random.randrange(0, len(CHOICE), 1)]
            elif REPRESENTATION_MODEL_PARAMETER_SETUP[parameter]['type'] == 'fixed':
                representation_model_params[parameter] = REPRESENTATION_MODEL_PARAMETER_SETUP[parameter]['value']
            elif REPRESENTATION_MODEL_PARAMETER_SETUP[parameter]['type'] == 'each':
                representation_model_params[parameter] = REPRESENTATION_MODEL_PARAMETER_SETUP[parameter]['value']

        population_parameters['no_models'] = representation_model_params['TREES_PER_POPULATION']

        representation_model_params_csv = create_path(iteration_path, 'representation_model_parameters.csv')
        write_csv(representation_model_params_csv, {
            **representation_model_params,
            'MODE': population_parameters['mode']
        })

        populations = [
            {
                'name': 'without_loops',
                'params': {
                    **population_parameters,
                    'loop': 0
                }
            },
            {
                'name': 'with_loops',
                'params': {
                    **population_parameters,
                    'loop': round(random.uniform(0.1, 0.5), 1)
                }
            }
        ]

        starting_log_index = 1
        for population in populations:
            print('population: {}'.format(population['name']))
            print('paramaters: {}'.format(population['params']))
            (corpus, next_index) = generate_representation_model_data(iteration_path,
                                                                      population,
                                                                      representation_model_params,
                                                                      starting_log_index)
            corpora.append(corpus)
            starting_log_index = next_index

        general_corpus = [trace_info for corpus in corpora for trace_info in corpus]
        general_corpus_path = create_path(iteration_path, 'general_corpus.json')
        write_json(general_corpus_path, general_corpus)

        write_training_test_data(iteration_path, general_corpus, TRAINING_PERCENTAGE, WINDOW_SIZE)

    print('Data generated for {} iterations, using a {}% training percentage and window size of {}'.format(ITERATION_COUNT, TRAINING_PERCENTAGE, WINDOW_SIZE))

    return set_path
