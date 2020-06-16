from definitions import *
from experiment.perform_tests import main as perform_tests
from experiment.export_log_representations import main as export_log_representations
from experiment.get_log_representation_data import main as get_log_representation_data
# from experiment.get_prediction_model import main as get_prediction_model
from scipy.stats import ttest_ind
from experiment.config import *
from experiment.export_log_representations import main as export_log_representations
from experiment.get_log_representation_data import main as get_log_representation_data
# from experiment.get_log_representation_data import get_training_test_data as training_test_data_prediction
# from experiment.get_prediction_model import main as get_prediction_model
from experiment.get_prediction_model import alternative as get_prediction_model_alternative
from experiment.train_prediction_model import main as train_prediction_model
from experiment.perform_clustering import for_iterations as perform_clustering_for_iterations
import experiment.get_observations as get_observations
from analyse_majority_class import main as analyse_majority_class

experiment_path = '/Users/marinosoro/Codecetera/UHasselt/masterproef/laboratory/training-sets/REPRESENTATION_DATA--2020-05-22/'

# export_log_representations(experiment_path)

# perform_clustering_for_iterations(experiment_path)
analyse_majority_class(experiment_path)


# perform_tests(experiment_path)
#
# representation_model_observations = get_observations.representation_model(experiment_path)
# representation_model_accuracy_list = [obj['accuracy'] for obj in representation_model_observations]
# representation_model_mean_accuracy = sum(representation_model_accuracy_list) / len(representation_model_accuracy_list)
#
# print('Mean accuracy for representation model: ', representation_model_mean_accuracy)
