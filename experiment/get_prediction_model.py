from tensorflow import keras
from config import *


# def main(representation_length, plot_path=None):
#     input_layer = keras.Input(shape=(representation_length,), name='log_representation')
#     hidden_layer = keras.layers.Dense(128, activation='relu', name='hidden_layer_1')(input_layer)
#     hidden_layer_2 = keras.layers.Dense(128, activation='relu', name='hidden_layer_2')(hidden_layer)
#     hidden_layer_3 = keras.layers.Dense(128, activation='relu', name='hidden_layer_3')(hidden_layer_2)
#     output_layer_bool = keras.layers.Dense(2, activation='softmax', name='prediction_boolean')(hidden_layer_3)
#     output_layer_value = keras.layers.Dense(1, activation='sigmoid', name='prediction_value')(hidden_layer_3)
#
#     model = keras.Model(inputs=input_layer, outputs=[output_layer_bool, output_layer_value], name='prediction_model')
#     # model = keras.Model(inputs=input_layer, outputs=[output_layer_value], name='prediction_model')
#     model.summary()
#     if plot_path:
#         keras.utils.plot_model(model, plot_path, show_shapes=True)
#
#     model.compile(
#         optimizer='adam',
#         loss={
#             'prediction_boolean': 'sparse_categorical_crossentropy',
#             'prediction_value': 'mean_squared_error'
#         },
#         loss_weights={
#             'prediction_boolean': 1.0,
#             'prediction_value': 1.0
#         },
#         metrics=['accuracy']
#     )
#     return model


def loops_categorical(representation_length, plot_path=None):
    input_layer = keras.Input(shape=(representation_length,), name='log_representation')
    hidden_layer = keras.layers.Dense(128, activation='relu', name='hidden_layer_1')(input_layer)
    hidden_layer_2 = keras.layers.Dense(128, activation='relu', name='hidden_layer_2')(hidden_layer)
    hidden_layer_3 = keras.layers.Dense(128, activation='relu', name='hidden_layer_3')(hidden_layer_2)
    output_layer_bool = keras.layers.Dense(2, activation='softmax', name='prediction_boolean')(hidden_layer_3)

    model = keras.Model(inputs=input_layer, outputs=output_layer_bool, name='prediction_model_loops_boolean')
    model.summary()
    if plot_path:
        keras.utils.plot_model(model, plot_path, show_shapes=True)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def max_repeat_categorical(representation_length, category_count, plot_path=None):
    input_layer = keras.Input(shape=(representation_length,), name='log_representation')
    hidden_layer = keras.layers.Dense(128, activation='relu', name='hidden_layer_1')(input_layer)
    hidden_layer_2 = keras.layers.Dense(128, activation='relu', name='hidden_layer_2')(hidden_layer)
    hidden_layer_3 = keras.layers.Dense(128, activation='relu', name='hidden_layer_3')(hidden_layer_2)
    output_layer_bool = keras.layers.Dense(category_count, activation='softmax', name='prediction_categorical')(hidden_layer_3)

    model = keras.Model(inputs=input_layer, outputs=output_layer_bool, name='prediction_model_max_repeat_categorical')
    model.summary()
    if plot_path:
        keras.utils.plot_model(model, plot_path, show_shapes=True)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def alternative(representation_length, plot_path=None):
    input_layer = keras.Input(shape=(representation_length,), name='log_representation')
    hidden_layer = keras.layers.Dense(128, activation='relu', name='hidden_layer')(input_layer)
    output_layer_bool = keras.layers.Dense(2, activation='softmax', name='prediction_boolean')(hidden_layer)

    model = keras.Model(inputs=input_layer, outputs=[output_layer_bool], name='prediction_model')
    model.summary()
    if plot_path:
        keras.utils.plot_model(model, plot_path, show_shapes=True)

    model.compile(
        optimizer='adam',
        loss={
            'prediction_boolean': 'sparse_categorical_crossentropy',
        },
        loss_weights={
            'prediction_boolean': 1.0,
        },
        metrics=['accuracy']
    )
    return model