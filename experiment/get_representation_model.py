from tensorflow import keras
from config import *


def main(activity_count, log_count, log_representation_size):
    # Model definition here
    def get_activity_model():
        inputs = keras.Input(shape=(activity_count,))
        outputs = keras.layers.Dense(ACTIVITY_HIDDEN_LAYER_LENGTH, activation='relu', name='activity_model')(inputs)
        return keras.Model(inputs, outputs)

    def get_log_model():
        inputs = keras.Input(shape=(log_count,), name='log_vector')
        outputs = keras.layers.Dense(log_representation_size, activation='relu', name='log_model')(inputs)
        return keras.Model(inputs, outputs, name='log_vector')

    def create_model():
        activity_model = get_activity_model()

        activity_inputs = [keras.Input(shape=(activity_count,), name='activity_{}'.format(i + 1)) for i in
                           range(WINDOW_SIZE - 1)]
        activity_models = [activity_model(activity_inputs[i]) for i in range(WINDOW_SIZE - 1)]
        activities_average = keras.layers.average(activity_models)

        log_input = keras.Input(shape=(log_count,), name='log_id')
        log_model = get_log_model()(log_input)

        hidden_layer = keras.layers.concatenate([activities_average, log_model])

        output = keras.layers.Dense(activity_count, activation='softmax', name='prediction')(hidden_layer)

        model = keras.Model(inputs=[*activity_inputs, log_input], outputs=output, name='test_model')
        model.summary()

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    return create_model()
