import os

# Tensorflow print errors (default is print everything)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
import tensorflow.keras as keras

#/ CREATE MODELS /#

# 3D CNN model for frame stack as input <4, 84, 84>
def create_image_processing_model(frame_size, frame_stack, n_actions, n_channels, model_loss, model_optimizer, model_metrics):
    kernel_size = (frame_stack, 4, 4)
    pool_size = (frame_stack // 2, 2, 2)
    input_shape = (frame_stack, frame_size, frame_size, n_channels)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv3D(128, kernel_size=kernel_size, activation="relu", padding="same", input_shape=input_shape))
    model.add(keras.layers.MaxPool3D(pool_size=pool_size))

    model.add(keras.layers.Conv3D(64, kernel_size=kernel_size, activation="relu", padding="same"))
    model.add(keras.layers.MaxPool3D(pool_size=pool_size))
    
    model.add(keras.layers.Conv3D(32, kernel_size=kernel_size, activation="relu", padding="same"))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation="softplus"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(n_actions, activation="softplus"))

    model.compile(loss=model_loss, optimizer=model_optimizer, metrics=model_metrics)

    print(model.summary())
    return model

# 1D CNN model for RAM as input <128>
def create_state_model(n_bytes, action_count, channel_count, model_loss, model_optimizer, model_metrics):
    kernel_size = n_bytes // 32
    pool_size = n_bytes // 64
    input_shape = (n_bytes, channel_count)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(128, kernel_size=kernel_size, activation="relu", padding="same", input_shape=input_shape))
    model.add(keras.layers.MaxPool1D(pool_size=pool_size))

    model.add(keras.layers.Conv1D(64, kernel_size=kernel_size, 
    activation="relu", padding="same"))
    model.add(keras.layers.MaxPool1D(pool_size=pool_size))
   
    model.add(keras.layers.Conv1D(32, kernel_size=kernel_size, activation="relu", padding="same"))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation="softplus", bias_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(action_count, activation="softplus", bias_initializer="he_normal"))

    model.compile(loss=model_loss, optimizer=model_optimizer, metrics=model_metrics)

    print(model.summary())
    return model