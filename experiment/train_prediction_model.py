import tensorflow as tf
from definitions import *
import numpy as np
from experiment.config import WINDOW_SIZE, EPOCHS


def main(model, train_input, train_output, test_input, test_output):
    # print("train input: ", np.asarray(train_input))
    training_results = model.fit(
        np.asarray(train_input),
        np.asarray(train_output),
        validation_data=(
            np.asarray(test_input),
            np.asarray(test_output)
        ),
        epochs=EPOCHS)

    return model, training_results


def get_prepared_data(data_dir, activity_vocab, log_vocab):
    activities = [
        prepared_network_data(create_path(data_dir, 'activity_{}.txt'.format(i + 1)), activity_vocab, 'activity') for
        i in range(WINDOW_SIZE - 1)]

    log = prepared_network_data(create_path(data_dir, 'log.txt'), log_vocab, 'log')

    output = prepared_network_data(create_path(data_dir, 'output.txt'), activity_vocab, 'output')

    return activities, log, output