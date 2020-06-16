WINDOW_SIZE = 6
EPOCHS = 25

MODELS_PER_POPULATION = 3
LOGS_PER_MODEL = 50
TRACES_PER_LOG = 100

ACTIVITY_HIDDEN_LAYER_LENGTH = 32
LOG_HIDDEN_LAYER_LENGTH = 11

SHOULD_ALWAYS_TRAIN = False

ITERATION_COUNT = 100
TRAINING_PERCENTAGE = 80
PREDICTION_TRAINING_PERCENTAGE = 80

INDIVIDUAL_P_THRESHOLD = 0.0125
# INDIVIDUAL_P_THRESHOLD = 0.01
MODE_THRESHOLD = 10
LOGS_PER_TREE_THRESHOLD = 75

CLUSTER_ITERATION_COUNT = 10

REPRESENTATION_MODEL_PARAMETER_SETUP = {
    'TREES_PER_POPULATION': {
        'type': 'fixed',
        'value': 1
    },
    'LOGS_PER_TREE': {
        'type': 'range',
        'value': [50, 100]
    },
    'TRACES_PER_LOG': {
        'type': 'fixed',
        'value': 1000
    },
    'REPRESENTATION_LENGTH': {
        'type': 'each',
        'value': [32, 64]
    }
}

POPULATION_PARAMETER_SETUP = {
    'min': {
        'type': 'range',
        'value': [5, 10]
    },
    'max': {
        'type': 'range',
        'value': [10, 15]
    },
    'mode': {
        'type': 'range',
        'value': ['min', 'max']
    },
    'parallel': {
        'type': 'range',
        'value': [0.0, 0.5]
    },
    'silent': {
        'type': 'range',
        'value': [0.0, 0.3]
    },
    'duplicate': {
        'type': 'choice',
        'value': [0, 1]
    },
    'lt_dependency': {
        'type': 'range',
        'value': [0.0, 0.1]
    },
    'infrequent': {
        'type': 'range',
        'value': [0.0, 0.5]
    },
    'max_repeat': {
        'type': 'range',
        'value': [2, 10]
    },
    'sequence': {
        'type': 'fixed',
        'value': 0.2
    },
    'choice': {
        'type': 'fixed',
        'value': 0.2
    },
    'or': {
        'type': 'fixed',
        'value': 0.2
    },
    'unfold': {
        'type': 'fixed',
        'value': 1
    },
    'loop': {
        'type': 'each',
        'value': [
            {
                'type': 'fixed',
                'value': 0
            },
            {
                'type': 'range',
                'value': [0.1, 0.5]
            },
        ]
    },
}
