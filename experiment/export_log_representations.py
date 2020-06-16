from definitions import *
from config import *
import tensorflow as tf
from tensorflow import keras
import numpy as np


def main(experiment_path):
    iterations = glob.glob(create_path(experiment_path, 'iterations', '*'))
    for iteration in iterations:
        models_dir = create_path(iteration, 'models')
        model_files = glob.glob(create_path(models_dir, '*.h5'))
        for model_file in model_files:
            model = tf.keras.models.load_model(model_file)
            layers = model.layers

            log_model_layer = next((layer for layer in layers if 'log_vector' in layer.name), None)
            log_model_weights = log_model_layer.get_weights()[0]

            weights_file_path = create_path(models_dir, '{}__log_representation_matrix.txt'.format(model_file.split('.')[0]))
            if os.path.exists(weights_file_path):
                # Make sure not to just append to the existing file when rerunning this script.
                os.remove(weights_file_path)

            np.set_printoptions(linewidth=np.inf)

            number_of_logs = len(log_model_weights)
            print('number of logs: {}'.format(number_of_logs))

            with open(weights_file_path, 'a') as file:
                count = 1
                for weights_vector in log_model_weights:
                    # Apply Relu activation function
                    weights_vector = [max(0, value) for value in weights_vector]

                    # print(count)
                    file.write(str(weights_vector) + '\n')
                    count += 1
