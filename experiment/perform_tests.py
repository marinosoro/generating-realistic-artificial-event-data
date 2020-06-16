from definitions import *
from experiment.config import *
import experiment.get_observations as get_observations
from scipy.stats import ttest_ind


def main(experiment_path):
    representation_model_observations = get_observations.representation_model(experiment_path)

    mode(representation_model_observations, MODE_THRESHOLD)
    logs_per_tree(representation_model_observations, LOGS_PER_TREE_THRESHOLD)
    representation_length(representation_model_observations)


def print_result(p_value):
    print()
    print('p_value: ', p_value)
    if p_value < INDIVIDUAL_P_THRESHOLD:
        print('We reject the nullhypothesis at a {}% confidence level.'.format((1 - INDIVIDUAL_P_THRESHOLD) * 100))
    else:
        print(
            'We fail to reject the nullhypothesis at a {}% confidence level.'.format((1 - INDIVIDUAL_P_THRESHOLD) * 100))
    print()


def mode(observations, threshold):
    print('Testing the statistical significance of the \'mode\' attribute')

    a = [obj['accuracy'] for obj in observations if obj['mode'] <= threshold]
    b = [obj['accuracy'] for obj in observations if obj['mode'] > threshold]

    print('a: ', a)
    print('b: ', b)
    print()
    print('Mean accuracy a: ', sum(a) / len(a))
    print('Mean accuracy b: ', sum(b) / len(b))

    statistic, p_value = ttest_ind(a, b);
    print_result(p_value)


def logs_per_tree(observations, threshold):
    print('Testing the statistical significance of the \'logs_per_tree\' attribute')

    a = [obj['accuracy'] for obj in observations if obj['logs_per_tree'] <= threshold]
    b = [obj['accuracy'] for obj in observations if obj['logs_per_tree'] > threshold]

    print('a: ', a)
    print('b: ', b)
    print()
    print('Mean accuracy a: ', sum(a) / len(a))
    print('Mean accuracy b: ', sum(b) / len(b))

    statistic, p_value = ttest_ind(a, b);
    print_result(p_value)


def representation_length(observations):
    print('Testing the statistical significance of the \'representation_length\' attribute')

    a = [obj['accuracy'] for obj in observations if obj['representation_length'] == 32]
    b = [obj['accuracy'] for obj in observations if obj['representation_length'] == 64]

    print('a: ', a)
    print('b: ', b)
    print()
    print('Mean accuracy a: ', sum(a) / len(a))
    print('Mean accuracy b: ', sum(b) / len(b))

    statistic, p_value = ttest_ind(a, b);
    print_result(p_value)
