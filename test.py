from definitions import *
from experiment.perform_tests import main as perform_tests
from experiment.export_log_representations import main as export_log_representations
from experiment.get_log_representation_data import main as get_log_representation_data
# from experiment.get_prediction_model import main as get_prediction_model
from scipy.stats import ttest_ind
from experiment.config import *
from experiment.export_log_representations import main as export_log_representations
from experiment.get_log_representation_data import main as get_log_representation_data
from experiment.get_log_representation_data import get_training_test_data__loop as training_test_data_prediction__loop
from experiment.get_log_representation_data import get_training_test_data__max_repeat as training_test_data_prediction__max_repeat
from experiment.get_log_representation_data import get_baseline as get_baseline
import experiment.get_prediction_model as get_prediction_model
from experiment.get_prediction_model import alternative as get_prediction_model_alternative
from experiment.train_prediction_model import main as train_prediction_model

experiment_path = '/Users/marinosoro/Codecetera/UHasselt/masterproef/laboratory/training-sets/REPRESENTATION_DATA--2020-05-22/'

# perform_tests(experiment_path)
# export_log_representations(experiment_path)


# log_representation_data = get_log_representation_data(experiment_path)
# print("log representation data size; ", len(log_representation_data[32]))
#
# prediction_models = []
# for length in log_representation_data.keys():
#     prediction_models.append({
#         'name': 'prediction_model_loop_categorical_length_{}'.format(length),
#         'type': 'loop',
#         'model': get_prediction_model.loops_categorical(length)
#     })
#     max_repeat_range = POPULATION_PARAMETER_SETUP['max_repeat']['value']
#     prediction_models.append({
#         'name': 'prediction_model_max_repeat_categorical_length_{}'.format(length),
#         'type': 'max_repeat',
#         'model': get_prediction_model.max_repeat_categorical(length, max_repeat_range[1] - max_repeat_range[0] + 1)
#     })
#
# for model_index in range(len(prediction_models)):
#     print('Training model: ', prediction_models[model_index]['name'])
#     model_name_list = prediction_models[model_index]['name'].split("_")
#     representation_length_for_model = int(model_name_list[len(model_name_list)-1])
#
#     prediction_input = []
#     prediction_outputs = []
#     test_input = []
#     test_outputs = []
#     if prediction_models[model_index]['type'] == 'loop':
#         training_data, test_data = training_test_data_prediction__loop(log_representation_data[representation_length_for_model])
#
#         prediction_input = training_data['input']
#         prediction_outputs = training_data['output']
#
#         test_input = test_data['input']
#         test_outputs = test_data['output']
#
#     elif prediction_models[model_index]['type'] == 'max_repeat':
#         training_data, test_data = training_test_data_prediction__max_repeat(log_representation_data[representation_length_for_model])
#
#         prediction_input = training_data['input']
#         prediction_outputs = training_data['output']
#
#         test_input = test_data['input']
#         test_outputs = test_data['output']
#
#     prediction_models[model_index]['model'], training_results = train_prediction_model(prediction_models[model_index]['model'], prediction_input, prediction_outputs, test_input, test_outputs)
#     prediction_models[model_index]['model'].save(create_path(experiment_path, "prediction_models", '{}.h5'.format(prediction_models[model_index]['name'])))
#
#     val_accuracy_history = training_results.history['val_accuracy']
#
#     val_accuracy = val_accuracy_history[len(val_accuracy_history)-1]
#
#     write_csv(create_path(experiment_path, 'prediction_results', '{}.csv'.format(prediction_models[model_index]['name'])), {
#         'ACCURACY': val_accuracy,
#     })

loop_baseline = get_baseline(experiment_path, 'loop')
max_repeat_baseline = get_baseline(experiment_path, 'max_repeat', False)

print('loop baseline: ', loop_baseline)
print('max_repeat baseline: ', max_repeat_baseline)


# a = [0.6662, 0.6730, 0.6717, 0.6714]
# b = [0.7050, 0.7146, 0.6995, 0.6970]
# statistic, p_value = ttest_ind(a, b);
# print()
# print('p_value: ', p_value)
# if p_value < INDIVIDUAL_P_THRESHOLD:
#     print('We reject the nullhypothesis at a {}% confidence level.'.format((1 - INDIVIDUAL_P_THRESHOLD) * 100))
# else:
#     print(
#         'We fail to reject the nullhypothesis at a {}% confidence level.'.format(
#             (1 - INDIVIDUAL_P_THRESHOLD) * 100))
# print()