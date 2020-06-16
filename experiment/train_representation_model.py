import tensorflow as tf
from definitions import *
import numpy as np
from experiment.config import WINDOW_SIZE


def main(model, train_activities, train_log, train_output):
    model.fit(
        [*train_activities, train_log],
        train_output,
        epochs=EPOCHS)

    return model


def get_prepared_data(data_dir, activity_vocab, log_vocab):
    activities = [
        prepared_network_data(create_path(data_dir, 'activity_{}.txt'.format(i + 1)), activity_vocab, 'activity') for
        i in range(WINDOW_SIZE - 1)]

    log = prepared_network_data(create_path(data_dir, 'log.txt'), log_vocab, 'log')

    output = prepared_network_data(create_path(data_dir, 'output.txt'), activity_vocab, 'output')

    return activities, log, output