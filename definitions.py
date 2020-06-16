# -*- coding: utf-8 -*-

import sys
import os
from datetime import date
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import progressbar
import time
from itertools import groupby
from config import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Root of the project

sys.path.insert(0, ROOT_DIR)

PLUGINS_DIR = os.path.join(ROOT_DIR, 'PTandLogGenerator', 'plugins')
DATA_DIR = os.path.join(ROOT_DIR, 'PTandLogGenerator', 'data')

LOGS_DIR = os.path.join(DATA_DIR, 'logs')
TREES_DIR = os.path.join(DATA_DIR, 'trees')
INPUT_DIR = os.path.join(DATA_DIR, 'parameter_files')

TRAINING_SETS_DIR = os.path.join(ROOT_DIR, 'training-sets')
POPULATIONS_DIR = os.path.join(ROOT_DIR, 'populations')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

def create_path(*argv):
    path_components = []
    for arg in argv:
        path_components.append(arg)

    return os.path.join(*path_components)


def generate_trees(destDir, input_file):
    os.system('python {}/generate_newick_trees.py {} --g=True'.format(PLUGINS_DIR, input_file))
    os.rename(TREES_DIR, destDir)
    os.mkdir(TREES_DIR) # Recreate the moved trees directory


def generate_logs(destDir, treesDir, traces, noise):
    os.system('python {}/generate_logs.py {} {} --f=csv --i={}/ --l={}'.format(PLUGINS_DIR, traces, noise, treesDir, int(LOGS_PER_MODEL)))
    os.rename(LOGS_DIR, destDir)
    os.mkdir(LOGS_DIR) # Recreate the moved logs directory


def generate_logs2(destDir, treesDir, traces, logs_per_model, noise):
    os.system('python {}/generate_logs.py {} {} --f=csv --i={}/ --l={}'.format(PLUGINS_DIR, traces, noise, treesDir, logs_per_model))
    os.rename(LOGS_DIR, destDir)
    os.mkdir(LOGS_DIR) # Recreate the moved logs directory


def write_csv(destDir, dict):
    import csv
    with open(destDir, 'w') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(dict.keys())
        writer.writerow(dict.values())


def read_csv(file, direction='horizontal'):
    import csv
    reader = csv.reader(open(file), delimiter=';')

    result = {}
    if direction == 'horizontal':
        result = {i[0]: x for i in zip(*list(reader)) for x in i[1:]}
    elif direction == 'vertical':
        for row in reader:
            key = row[0]
            if key in result:
                # implement your duplicate row handling here
                pass
            result[key] = row[1:]
    return result


def generate_new_population(training_set_dir, name, pop_params, number_of_traces, noise_level=0):
    # Create new directory for population
    population_dir = create_path(training_set_dir, 'populations', name)
    os.mkdir(population_dir)

    # Create input parameters file
    input_file = create_path(population_dir, 'parameters.csv')
    write_csv(input_file, pop_params)

    # Generate trees for population
    tree_dir = create_path(population_dir, 'trees')
    generate_trees(tree_dir, input_file)

    # Generate logs for trees in population
    log_dir = create_path(population_dir, 'logs')
    generate_logs(log_dir, tree_dir, number_of_traces, noise_level)

    return population_dir


def generate_new_population2(training_set_dir, name, pop_params, model_params, noise_level=0):
    # Create new directory for population
    population_dir = create_path(training_set_dir, 'populations', name)
    os.mkdir(population_dir)

    # Create input parameters file
    input_file = create_path(population_dir, 'parameters.csv')
    write_csv(input_file, pop_params)

    # Generate trees for population
    print('START GENERATING TREES NOW')
    tree_dir = create_path(population_dir, 'trees')
    generate_trees(tree_dir, input_file)

    # Generate logs for trees in population
    log_dir = create_path(population_dir, 'logs')
    generate_logs2(log_dir, tree_dir, model_params['TRACES_PER_LOG'], model_params['LOGS_PER_TREE'], noise_level)

    return population_dir


# corpus = [
#     {
#         'log_id': Number,
#         'trace': String[],
#     }
# ]
def generate_corpus(log_files, starting_log_index=1, population_params=None):
    corpus = []
    log_index = starting_log_index
    for log in log_files:
        current_trace = []
        prev_trace_id = None

        import csv
        with open(log, 'r') as log_file:
            reader = csv.reader(log_file)
            header = next(reader)
            for row in reader:
                if not prev_trace_id or prev_trace_id == row[0]:
                    current_trace.append(row[1])
                else:
                    corpus.append({
                        'log_id': log_index,
                        'trace_id': int(row[0])-1,
                        'trace': current_trace,
                        'population_params': get_normalized_population_params(population_params)
                    })
                    current_trace = [row[1]]
                prev_trace_id = row[0]
            corpus.append({
                'log_id': log_index,
                'trace_id': int(row[0]) - 1,
                'trace': current_trace,
                'population_params': get_normalized_population_params(population_params)
            })
        log_index += 1
    return corpus


# corpus = [
#     {
#         'log_id': Number,
#         'trace': String[],
#     }
# ]
def generate_corpus2(log_files, starting_log_index, population_params=None):
    corpus = []
    log_index = starting_log_index
    for log in log_files:
        current_trace = []
        prev_trace_id = None

        import csv
        with open(log, 'r') as log_file:
            reader = csv.reader(log_file)
            header = next(reader)
            for row in reader:
                if not prev_trace_id or prev_trace_id == row[0]:
                    current_trace.append(row[1])
                else:
                    corpus.append({
                        'log_id': log_index,
                        'trace_id': int(row[0])-1,
                        'trace': current_trace,
                        'population_params': get_normalized_population_params2(population_params)
                    })
                    current_trace = [row[1]]
                prev_trace_id = row[0]
            corpus.append({
                'log_id': log_index,
                'trace_id': int(row[0]) - 1,
                'trace': current_trace,
                'population_params': get_normalized_population_params2(population_params)
            })
        log_index += 1
    return corpus, log_index


def get_normalized_population_params(population_params):
    if population_params is None:
        return None
    return [
        population_params['sequence'],
        population_params['choice'],
        population_params['parallel'],
        population_params['loop'],
        population_params['or'],
        population_params['silent'],
        population_params['duplicate'],
        population_params['lt_dependency'],
        population_params['infrequent'],
        population_params['unfold'],
        population_params['max_repeat']
    ]


def get_normalized_population_params2(population_params):
    if population_params is None:
        return None
    return [
        population_params['SEQUENCE'],
        population_params['CHOICE'],
        population_params['PARALLEL'],
        population_params['LOOP'],
        population_params['OR'],
        population_params['SILENT'],
        population_params['DUPLICATE'],
        population_params['LT_DEPENDENCY'],
        population_params['INFREQUENT'],
        population_params['UNFOLD'],
        population_params['MAX_REPEAT']
    ]


def generate_training_data(from_corpus, window_size=WINDOW_SIZE):
    np.set_printoptions(linewidth=np.inf)
    activity_inputs = []
    trace_input = []
    log_input = []
    prediction_output = []
    activity_vocab = get_vocabulary(from_corpus)
    trace_vocab = index_list(get_trace_count(from_corpus))
    log_vocab = index_list(get_log_count(from_corpus))
    for window_index in range(window_size-1):
        activity_inputs.append([])
    for traceDict in from_corpus:
        trace = traceDict['trace']
        while len(trace) >= window_size:
            index = 0
            while index < window_size-1:
                activity = trace[index]
                encoding = get_one_hot_encoding(activity, activity_vocab)
                activity_inputs[index].append(encoding)
                index += 1
            trace_encoding = get_one_hot_encoding(traceDict['trace_id'], trace_vocab)
            trace_input.append(trace_encoding)
            log_encoding = get_one_hot_encoding(traceDict['log_id'], log_vocab)
            log_input.append(log_encoding)
            prediction_encoding = get_one_hot_encoding(trace[window_size-1], activity_vocab)
            prediction_output.append(convert_to_index(prediction_encoding))
            trace.pop(0)

    activity_inputs = (np.array(el) for el in activity_inputs)
    trace_input = np.array(trace_input)
    log_input = np.array(log_input)
    prediction_output = np.array(prediction_output)
    # activity_inputs = np.array2string(activity_inputs)
    # trace_input = np.array2string(trace_input)
    # log_input = np.array2string(log_input)
    # prediction_output = np.array2string(prediction_output)
    print('corpus length: {}'.format(len(from_corpus)))
    print('trace_input length: {}'.format(len(trace_input)))
    print('log_input length: {}'.format(len(log_input)))
    print('prediction_output length: {}'.format(len(prediction_output)))
    return (input for input in activity_inputs), trace_input, log_input, prediction_output


def write_json(path, dict):
    import json
    json = json.dumps(dict)
    f = open(path, "w")
    f.write(json)
    f.close()


def write_tuple(path, *tuple_to_write):
    for x in tuple_to_write:
        print('tuple item length: {}'.format(len(x)))

    array_to_write = [','.join([str(elem) for elem in tuple_el]) for tuple_el in tuple_to_write]
    print('array_to_write length: {}'.format(len(array_to_write)))
    with open(path, 'w') as fp:
        fp.write('\n'.join(array_to_write))


def read_json(path):
    import json
    with open(path, 'r') as f:
        dict = json.load(f)
        return dict


def read_tuple(path, cast_to_np_array=False):
    tuple_elements = []
    with open(path, 'r') as f:
        lines = f.readlines()
        print('Number of lines: {}'.format(len(lines)))
        for line in lines:
            tuple_elements.append(line.split(','))
            print('line length: {}'.format(len(line.split(','))))
    if cast_to_np_array:
        tuple_elements = [np.array(el) for el in tuple_elements]
    print('Number of tuple elements: {}'.format(len(tuple_elements)))
    return (el for el in tuple_elements)


def unique(list):
    x = np.array(list)
    return np.unique(x).tolist()


def get_vocabulary(for_corpus):
    all_traces = []
    for traceInfo in for_corpus:
        all_traces.append(traceInfo['trace'])
    all_activities = [activity for trace in all_traces for activity in trace]
    print('LENGTH CORPUS: ', len(for_corpus))
    print('GET VOCABULARY: ', unique(all_activities))
    return unique(all_activities)


def get_vocabulary_length(general_corpus):
    return len(get_vocabulary(general_corpus))


def get_trace_count(general_corpus):
    trace_count = 0
    for log_id in range(get_log_count(general_corpus)):
        trace_count += len(unique([item['trace_id'] for item in general_corpus if item['log_id'] == log_id+1]))
    return trace_count


def get_log_count(general_corpus):
    return len(unique([item['log_id'] for item in general_corpus]))


def prepare_representation_mode_set():
    if not os.path.exists(TRAINING_SETS_DIR):
        os.mkdir(TRAINING_SETS_DIR)

    # Create new directory for population
    set_name = 'REPRESENTATION_DATA--{}'.format(date.today())
    set_dir = create_path(TRAINING_SETS_DIR, set_name)
    duplicate_name_count = 0
    for item in os.listdir(TRAINING_SETS_DIR):
        if set_name in item:
            duplicate_name_count += 1
    if duplicate_name_count > 0:
        set_dir += '--{}'.format(duplicate_name_count+1)
    os.mkdir(set_dir)
    iterations_dir = create_path(set_dir, 'iterations')
    os.mkdir(iterations_dir)

    return set_dir


def prepare_iteration_set(iterations_path):
    count = len(os.listdir(iterations_path))
    iteration_path = create_path(iterations_path, '{}'.format(count+1))
    os.mkdir(iteration_path)
    os.mkdir(create_path(iteration_path, 'populations'))
    os.mkdir(create_path(iteration_path, 'models'))
    os.mkdir(create_path(iteration_path, 'results'))
    return iteration_path


def generate_representation_model_data(iteration_dir, population, representation_model_params, starting_log_index):
    population_dir = generate_new_population2(
        iteration_dir,
        population['name'],
        population['params'],
        representation_model_params
    )
    logs_path = create_path(population_dir, 'logs')

    log_files = glob.glob(logs_path + "/*.csv")

    corpus, next_index = generate_corpus2(log_files, starting_log_index)
    write_json(create_path(population_dir, 'corpus.json'), corpus)
    return corpus, next_index


def generate_training_set(populations, network_type='A', log_count=50):
    if not os.path.exists(TRAINING_SETS_DIR):
        os.mkdir(TRAINING_SETS_DIR)

    # Create new directory for population
    set_name = 'TRAINING-SET--{}'.format(date.today())
    set_dir = create_path(TRAINING_SETS_DIR, set_name)
    duplicate_name_count = 0
    for item in os.listdir(TRAINING_SETS_DIR):
        if set_name in item:
            duplicate_name_count += 1
    if duplicate_name_count > 0:
        set_dir += '--{}'.format(duplicate_name_count+1)
    os.mkdir(set_dir)
    os.mkdir(create_path(set_dir, 'populations'))

    corpora = []
    log_index = 1
    max_progress = len(populations) * log_count
    for population in populations:
        population_dir = generate_new_population(set_dir, population['name'], population['params'], TRACES_PER_LOG, 0)
        logs_path = create_path(population_dir, 'logs')

        log_files = glob.glob(logs_path + "/*.csv")

        if network_type == 'A':
            corpus = generate_corpus(log_files, log_index)
        elif network_type == 'B':
            corpus = generate_corpus(log_files, log_index, population['params'])
        write_json(create_path(population_dir, 'corpus.json'), corpus)

        log_index += len(log_files)
        corpora.append(corpus)

    general_corpus = [trace_info for corpus in corpora for trace_info in corpus]
    general_corpus_path = create_path(set_dir, 'corpus.json')
    write_json(general_corpus_path, general_corpus)

    write_training_test_data(set_dir, general_corpus)

    return set_dir


def write_training_test_data(from_set_dir, from_corpus, training_percentage=70, window_size=WINDOW_SIZE):
    np.set_printoptions(linewidth=np.inf)

    start_time = time.time()
    bar = progressbar.ProgressBar(maxval=len(from_corpus), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), progressbar.ETA(), progressbar.Timer()])
    bar.start()

    # Define data paths
    training_data_path = create_path(from_set_dir, 'training_data')
    test_data_path = create_path(from_set_dir, 'test_data')

    # Make sure paths exist
    if not os.path.exists(training_data_path):
        os.mkdir(training_data_path)
    if not os.path.exists(test_data_path):
        os.mkdir(test_data_path)

    # Define data files
    training_files = {
        'activities': [create_path(training_data_path, 'activity_{}.txt'.format(i+1)) for i in range(window_size)],
        'log': create_path(training_data_path, 'log.txt'),
        'output': create_path(training_data_path, 'output.txt')
    }
    test_files = {
        'activities': [create_path(test_data_path, 'activity_{}.txt'.format(i+1)) for i in range(window_size)],
        'log': create_path(test_data_path, 'log.txt'),
        'output': create_path(test_data_path, 'output.txt')
    }

    # Open all files
    training_files['activities'] = [open(filepath, 'a') for filepath in training_files['activities']]
    training_files['log'] = open(training_files['log'], 'a')
    training_files['output'] = open(training_files['output'], 'a')

    test_files['activities'] = [open(filepath, 'a') for filepath in test_files['activities']]
    test_files['log'] = open(test_files['log'], 'a')
    test_files['output'] = open(test_files['output'], 'a')

    # Define vocabularies
    activity_vocab = get_vocabulary(from_corpus)
    log_vocab = index_list(get_log_count(from_corpus))

    logs = groupby(from_corpus, lambda x: x['log_id'])
    current_progress = 1.0
    max_progress = float(len(from_corpus))
    for log_id, log_dict in logs:
        current_log = [traceDict for traceDict in log_dict]
        current_log_progress = 1.0
        max_log_progress = float(len(current_log))

        # Write training data
        for traceDict in current_log:
            bar.update(current_progress)
            current_log_progress_percentage = current_log_progress / max_log_progress * 100

            trace = traceDict['trace']
            population_params = traceDict['population_params']

            # Write data to files
            while len(trace) >= 2: # at least 1 input should be an actual activity
                index = len(trace) - 1 # Last element of the trace -> to be predicted

                ## PREDICTION
                prediction_encoding = activity_vocab.index(trace[index])
                # TODO: Write to file with/without ,
                if current_log_progress_percentage < training_percentage:
                    if population_params != None:
                        training_files['output'].write(str(population_params) + '\n')
                    else:
                        training_files['output'].write(str(prediction_encoding) + '\n')
                else:
                    if population_params != None:
                        test_files['output'].write(str(population_params) + '\n')
                    else:
                        test_files['output'].write(str(prediction_encoding) + '\n')


                ## INPUTS
                # Activities
                activities_count = 0
                while activities_count < window_size - 1: # We need to add window_size-1 activities as an input
                    index = len(trace) - 1 - activities_count - 1
                    if index >= 0:
                        activity = trace[index]
                    else:
                        activity = 'DUMMY'
                    # TODO: Write to file with/without ,
                    if current_log_progress_percentage < training_percentage:
                        training_files['activities'][window_size - 1 - activities_count - 1].write(str(activity) + '\n')
                    else:
                        test_files['activities'][window_size - 1 - activities_count - 1].write(str(activity) + '\n')
                    activities_count += 1

                # Log ID
                # TODO: Write to file with/without ,
                if current_log_progress_percentage < training_percentage:
                    training_files['log'].write(str(traceDict['log_id']) + '\n')
                else:
                    test_files['log'].write(str(traceDict['log_id']) + '\n')

                trace.pop()

            current_log_progress += 1
            current_progress += 1

    # print('corpus length: {}'.format(len(from_corpus)))
    bar.finish()
    end_time = time.time()
    print('Elapsed time: {}'.format(end_time - start_time))
    return


def split_into_training_testing(all_data, training_percentage = .8):
    (activity_1_input, activity_2_input, activity_3_input, trace_input, log_input, prediction_output) = all_data
    total_size = len(prediction_output)
    training_size = int(total_size * training_percentage)

    training_activity_1_input = activity_1_input[0:training_size]
    training_activity_2_input = activity_2_input[0:training_size]
    training_activity_3_input = activity_3_input[0:training_size]
    training_trace_input = trace_input[0:training_size]
    training_log_input = log_input[0:training_size]
    training_output = prediction_output[0:training_size]

    test_activity_1_input = activity_1_input[training_size+1:total_size-1]
    test_activity_2_input = activity_2_input[training_size+1:total_size-1]
    test_activity_3_input = activity_3_input[training_size+1:total_size-1]
    test_trace_input = trace_input[training_size+1:total_size-1]
    test_log_input = log_input[training_size+1:total_size-1]
    test_output = prediction_output[training_size+1:total_size-1]

    return (training_activity_1_input, training_activity_2_input, training_activity_3_input, training_trace_input, training_log_input, training_output), \
           (test_activity_1_input, test_activity_2_input, test_activity_3_input, test_trace_input, test_log_input, test_output)


def index_list(size):
    retval = []
    i = 1
    while i <= size:
        retval.append(i)
        i += 1
    return retval


def get_one_hot_encoding(item, vocab):
    # print('item: {}'.format(item))
    # print('vocab: {}'.format(vocab))
    index = vocab.index(item)
    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    vocab = np.array(vocab)
    vocab = vocab.reshape(len(vocab), 1)
    one_hot_encoded = one_hot_encoder.fit_transform(vocab)

    return one_hot_encoded[index]


def prepare_for_neural_network(corpus, training_data, index):
    transformed = {}
    iteration = training_data[index]

    act_vocab = get_vocabulary(corpus)
    trace_vocab = index_list(get_trace_count(corpus))
    log_vocab = index_list(get_log_count(corpus))

    for key in iteration.keys():
        if key == 'input':
            activities = []
            for activity in iteration[key]['activities']:
                activities.append(get_one_hot_encoding(activity, act_vocab))
            transformed[key] = {
                'activities': activities,
                'trace_id': get_one_hot_encoding(iteration[key]['trace_id'], trace_vocab),
                'log_id': get_one_hot_encoding(iteration[key]['log_id'], log_vocab),
            }
        elif key == 'output':
            output = [get_one_hot_encoding(activity, act_vocab) for activity in iteration[key]]
            transformed[key] = output
    return transformed


def print_separator():
    print('========================================')


def load_model_data(corpus, training_data):
    training_percentage = 0.8

    train_activity_1 = []
    train_activity_2 = []
    train_activity_3 = []
    train_trace = []
    train_log = []
    train_output = []

    test_activity_1 = []
    test_activity_2 = []
    test_activity_3 = []
    test_trace = []
    test_log = []
    test_output = []

    for data_index in range(len(training_data)):
        prepared_data = prepare_for_neural_network(corpus, training_data, data_index)
        if data_index < len(training_data) * training_percentage:
            train_activity_1.append(np.array(prepared_data['input']['activities'][0]))
            train_activity_2.append(np.array(prepared_data['input']['activities'][1]))
            train_activity_3.append(np.array(prepared_data['input']['activities'][2]))
            train_trace.append(np.array(prepared_data['input']['trace_id']))
            train_log.append(np.array(prepared_data['input']['log_id']))
            train_output.append(np.array(prepared_data['output'][0]))
        else:
            test_activity_1.append(np.array(prepared_data['input']['activities'][0]))
            test_activity_2.append(np.array(prepared_data['input']['activities'][1]))
            test_activity_3.append(np.array(prepared_data['input']['activities'][2]))
            test_trace.append(np.array(prepared_data['input']['trace_id']))
            test_log.append(np.array(prepared_data['input']['log_id']))
            test_output.append(np.array(prepared_data['output'][0]))

    train_activity_1 = np.array(train_activity_1)
    train_activity_2 = np.array(train_activity_2)
    train_activity_3 = np.array(train_activity_3)
    train_trace = np.array(train_trace)
    train_log = np.array(train_log)
    train_output = np.array(train_output)
    test_activity_1 = np.array(test_activity_1)
    test_activity_2 = np.array(test_activity_2)
    test_activity_3 = np.array(test_activity_3)
    test_trace = np.array(test_trace)
    test_log = np.array(test_log)
    test_output = np.array(test_output)

    train_output = np.array([convert_to_index(encoding) for encoding in train_output])
    test_output = np.array([convert_to_index(encoding) for encoding in test_output])

    return (train_activity_1, train_activity_2, train_activity_3, train_trace, train_log, train_output), \
           (test_activity_1, test_activity_2, test_activity_3, test_trace, test_log, test_output)


def convert_to_index(one_hot_encoding):
    retval = 0
    for index in range(len(one_hot_encoding)):
        if one_hot_encoding[index] == 1:
            retval = index
            break
    return retval


def prepared_activity_data2(path, activity_vocab):
    f = open(path)
    raw_data = f.readlines()
    raw_data = [el[:-1] for el in raw_data]
    for char in reversed(activity_vocab):
        raw_data.insert(0, char)
    raw_data.insert(0, 'DUMMY')
    classnames, indices = np.unique(raw_data, return_inverse=True)
    indices = indices[len(activity_vocab)+1:]
    n_values = len(activity_vocab)+1
    template = np.delete(np.eye(n_values, k=-1, dtype=int), n_values-1, 1)
    return template[indices]


def prepared_network_data(path, vocab, data_type='activity'):
    f = open(path)
    raw_data = f.readlines()
    raw_data = [el[:-1] for el in raw_data]
    if data_type == 'activity':
        for char in reversed(vocab):
            raw_data.insert(0, char)
        raw_data.insert(0, 'DUMMY')
        classnames, indices = np.unique(raw_data, return_inverse=True)
        indices = indices[len(vocab)+1:]
        n_values = len(vocab)+1
        template = np.delete(np.eye(n_values, k=-1, dtype=int), n_values-1, 1)
        return template[indices]
    elif data_type == 'log':
        classnames, indices = np.unique(raw_data, return_inverse=True)
        n_values = len(vocab)
        template = np.eye(n_values, k=0, dtype=int)
        return template[indices]
    elif data_type == 'output':
        template = np.array(range(len(vocab)))
        indices = [int(i) for i in raw_data]
        return template[indices]
