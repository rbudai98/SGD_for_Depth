import sys
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import datetime
#from sklearn import utils


img_height = 20
img_width = 15


def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

name_specific = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
profile_logs = "logs/" + name_specific

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

# MODEL


inputs = keras.Input(shape=(300,))
inputs.shape
x = layers.Dense(128, activation="sigmoid")(inputs)
x1 = layers.Dense(128, activation="sigmoid")(x)
x2 = layers.Dense(128, activation="sigmoid")(x1)
outputs = layers.Dense(1, activation="sigmoid")(x2)
model = keras.Model(inputs=inputs, outputs=outputs, name="fully_connected")

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint-20220417-150829')

model.summary()

print(model.get_weights())

#for layer in model.layers:
    #print(layer.name, layer)

for layer in model.layers[1:]:
    #print(layer.weights)
    print(keras.layers.serialize(layer))
    print(layer.bias.numpy())
    print("###################################################################")
  