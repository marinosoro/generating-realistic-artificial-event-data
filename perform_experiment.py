from experiment.generate_data import main as generate_data
from experiment.config import *
from experiment.get_representation_model import main as get_representation_model
from experiment.train_representation_model import main as train_representation_model
from experiment.train_representation_model import get_prepared_data as get_prepared_data_representation
from experiment.perform_tests import main as perform_tests
from experiment.export_log_representations import main as export_log_representations
from experiment.get_log_representation_data import main as get_log_representation_data
from experiment.get_log_representation_data import get_training_test_data as training_test_data_prediction
from experiment.get_prediction_model import main as get_prediction_model
from experiment.train_prediction_model import main as train_prediction_model
from definitions import *

experiment_path = generate_data()
experiment_iterations_path = create_path(experiment_path, 'iterations')
experiment_models_path = create_path(experiment_path, 'models')
iteration_paths = glob.glob(experiment_iterations_path + '/*')
print('iteration paths: ', ','.join(iteration_paths))


for iteration in iteration_paths:
    iteration_models_path = create_path(iteration, 'models')
    general_corpus = read_json(create_path(iteration, 'general_corpus.json'))
    activity_vocab = get_vocabulary(general_corpus)
    activity_count = len(activity_vocab)
    log_vocab = index_list(get_log_count(general_corpus))

    models = []
    for size in REPRESENTATION_MODEL_PARAMETER_SETUP['REPRESENTATION_LENGTH']['value']:
        log_count = get_log_count(general_corpus)
        models.append({
            'name': 'representation_model_size_{}'.format(size),
            'model': get_representation_model(activity_count, log_count, size)
        })

    training_data_dir = create_path(iteration, 'training_data')
    training_activities, training_log, training_output = get_prepared_data_representation(training_data_dir, activity_vocab, log_vocab)
    test_data_dir = create_path(iteration, 'test_data')
    test_activities, test_log, test_output = get_prepared_data_representation(test_data_dir, activity_vocab, log_vocab)
    for model_index in range(len(models)):
        print('Training model: ', models[model_index]['name'])
        models[model_index]['model'] = train_representation_model(models[model_index]['model'], training_activities, training_log, training_output)
        models[model_index]['model'].save(create_path(iteration_models_path, '{}.h5'.format(models[model_index]['name'])))
        # Validate model and save accuracy to file.
        test_loss, test_accuracy = models[model_index]['model'].evaluate([*test_activities, test_log], test_output)
        write_csv(create_path(iteration, 'results', '{}.csv'.format(models[model_index]['name'])), {
            'ACCURACY': test_accuracy,
        })

    print('experiment_path: {}'.format(experiment_iterations_path))
    print('models: ', ','.join([model['name'] for model in models]))

# TODO: After all iterations are complete, all observations (accuracies) need to be grouped by testing parameters \
#  and the effect of these parameters needs to be found according to a t-test

perform_tests(experiment_path)

# Start second part of the experiment
export_log_representations()
log_representation_data = get_log_representation_data(experiment_path)

prediction_models = []
for length in log_representation_data.keys():
    prediction_models.append({
        'name': 'prediction_model_length_{}'.format(length),
        'model': get_prediction_model(length)
    })

for model_index in range(len(prediction_models)):
    print('Training model: ', prediction_models[model_index]['name'])
    model_name_list = prediction_models[model_index]['name'].split("_")
    print("model name list: ", model_name_list)
    representation_length_for_model = int(model_name_list[len(model_name_list)-1])

    training_data, test_data = training_test_data_prediction(log_representation_data[representation_length_for_model])
    prediction_input = training_data['input']
    prediction_outputs = {
        'prediction_boolean': training_data['output_boolean'],
        'prediction_value': training_data['output_value'],
    }

    test_input = test_data['input']
    test_outputs = {
        'prediction_boolean': test_data['output_boolean'],
        'prediction_value': test_data['output_value'],
    }

    prediction_models[model_index]['model'], training_results = train_prediction_model(prediction_models[model_index]['model'], prediction_input, prediction_outputs, test_input, test_outputs)
    prediction_models[model_index]['model'].save(create_path(experiment_path, "prediction_models", '{}.h5'.format(prediction_models[model_index]['name'])))

    print("training results: ", training_results.history['val_prediction_boolean_accuracy'])

    val_prediction_boolean_accuracy_history = training_results.history['val_prediction_boolean_accuracy']
    val_prediction_value_accuracy_history = training_results.history['val_prediction_value_accuracy']
    prediction_boolean_accuracy = val_prediction_boolean_accuracy_history[len(val_prediction_boolean_accuracy_history)-1]
    prediction_value_accuracy = val_prediction_value_accuracy_history[len(val_prediction_value_accuracy_history)-1]

    write_csv(create_path(experiment_path, 'prediction_results', '{}.csv'.format(prediction_models[model_index]['name'])), {
        'BOOLEAN_ACCURACY': prediction_boolean_accuracy,
        'VALUE_ACCURACY': prediction_value_accuracy,
    })



