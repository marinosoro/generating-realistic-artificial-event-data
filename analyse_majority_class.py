from definitions import *
from collections import Counter

training_set_name = 'REPRESENTATION_DATA--2020-05-22/iterations/1/'
training_set = create_path(TRAINING_SETS_DIR, training_set_name)

training_data_dir = create_path(training_set, 'training_data')
test_data_dir = create_path(training_set, 'test_data')

training_output_file = create_path(training_data_dir, 'output.txt')
test_output_file = create_path(test_data_dir, 'output.txt')

training_output = [line.rstrip('\n') for line in open(training_output_file)]
test_output = [line.rstrip('\n') for line in open(test_output_file)]

counter = Counter(test_output)
result = [(*key, counter[key]) for key in counter]
print('test_output: {}'.format(result))


def main(experiment_path):
    iterations_path = create_path(experiment_path, 'iterations')
    iterations = glob.glob(iterations_path + '/*')

    iteration_baselines = []

    for iteration in iterations:
        training_data_dir = create_path(iteration, 'training_data')
        test_data_dir = create_path(iteration, 'test_data')

        training_output_file = create_path(training_data_dir, 'output.txt')
        test_output_file = create_path(test_data_dir, 'output.txt')

        training_output = [line.rstrip('\n') for line in open(training_output_file)]
        test_output = [line.rstrip('\n') for line in open(test_output_file)]

        counter = Counter(test_output)
        result = [(*key, counter[key]) for key in counter]

        max_occurrence = max([counter[key] for key in counter])
        total_occurrence = sum([counter[key] for key in counter])

        baseline = max_occurrence / total_occurrence

        iteration_baselines.append(baseline)

    print('The mean baseline for all datasets is: ', sum(iteration_baselines) / len(iteration_baselines))