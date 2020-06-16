from definitions import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py
from scipy import cluster
from statistics import stdev
from experiment.config import *
from experiment.get_log_representation_data import for_iteration as get_log_representations_for_iteration
import matplotlib.pyplot as plt


def for_iterations(experiment_path):
    iterations_path = create_path(experiment_path, 'iterations')
    iterations = glob.glob(iterations_path + '/*')

    mean_accuracy_list = {}

    for iteration_index in range(len(iterations)):
        print_separator()
        print('Clustering for iteration: ', iteration_index)
        log_representations = get_log_representations_for_iteration(iterations[iteration_index])
        number_of_logs_per_pop = int(read_csv(create_path(iterations[iteration_index], 'representation_model_parameters.csv'))['LOGS_PER_TREE'])

        for representation_length in log_representations.keys():
            lr_for_length = log_representations[representation_length]
            number_of_logs = len(lr_for_length)

            # Loop starts here
            accuracy_list = []
            for clustering_iteration in range(CLUSTER_ITERATION_COUNT):
                centroid, k_means_array = cluster.vq.kmeans2(lr_for_length, 2)
                pop_A = k_means_array[:len(k_means_array) // 2]
                pop_B = k_means_array[len(k_means_array) // 2:]
                confusion_matrix = []

                # print('pop A: {}\n'.format(pop_A))
                # unique, counts = np.unique(pop_A, return_counts=True)
                counts = [number_of_logs_per_pop - np.count_nonzero(pop_A), np.count_nonzero(pop_A)]
                # print('{}\n'.format(dict(zip(unique, counts))))
                confusion_matrix.append(counts)

                # print('pop B: {}\n'.format(pop_B))
                # unique, counts = np.unique(pop_B, return_counts=True)
                counts = [number_of_logs_per_pop - np.count_nonzero(pop_B), np.count_nonzero(pop_B)]
                # print('{}\n'.format(dict(zip(unique, counts))))
                confusion_matrix.append(counts)

                # print('Confusion matrix: {}'.format(confusion_matrix))
                main_diagonal = [confusion_matrix[i - 1][i - 1] for i in range(len(confusion_matrix))]
                main_diagonal.reverse()

                # print('Main Diagonal: {}'.format(main_diagonal))

                initial_accuracy = sum(main_diagonal) / number_of_logs
                accuracy = max(initial_accuracy, 1 - initial_accuracy)
                # print('Accuracy of k means: {}'.format(accuracy))

                accuracy_list.append(accuracy)

                # if iteration_index == 89 and clustering_iteration == 1:
                #     w0 = np.array(lr_for_length)[k_means_array == 0]
                #     w1 = np.array(lr_for_length)[k_means_array == 1]
                #     pop_A = np.array(lr_for_length)[:len(lr_for_length)//2]
                #     pop_B = np.array(lr_for_length)[len(lr_for_length)//2:]
                #
                #     plt.plot(pop_A[:, 0], pop_A[:, 1], 'o', alpha=0.5, label='Without loops')
                #     plt.plot(pop_B[:, 0], pop_B[:, 1], 'd', alpha=0.5, label='With loops')
                #     plt.plot(centroid[:, 0], centroid[:, 1], 'k*', label='centroids')
                #     plt.axis('equal')
                #     plt.legend(shadow=True)
                #     plt.show()

            mean_accuracy = sum(accuracy_list) / len(accuracy_list)
            st_dev = stdev(accuracy_list)

            if representation_length not in mean_accuracy_list:
                mean_accuracy_list[representation_length] = []
            mean_accuracy_list[representation_length].append(mean_accuracy)

            print('LENGTH {} - Mean accuracy of k-means: {}'.format(representation_length, mean_accuracy))
            print('LENGTH {} - Standard deviation of k-means: {}'.format(representation_length, st_dev))
        print_separator()
        print()
    print('Mean accuracy for all iterations:')
    for length in mean_accuracy_list.keys():
        mean_accuracy = sum(mean_accuracy_list[length]) / len(mean_accuracy_list[length])
        print('LENGTH {} - Mean accuracy of k-means: {}'.format(length, mean_accuracy))
        print('LENGTH {} - Standard deviation of k-means: {}'.format(length, stdev(mean_accuracy_list[length])))
