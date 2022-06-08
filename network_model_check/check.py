import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import datetime
import json
# from sklearn import utils


img_height = 40
img_width = 30


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


inputs = keras.Input(shape=(1200,))
inputs.shape
x = layers.Dense(128, activation="sigmoid")(inputs)
x1 = layers.Dense(128, activation="sigmoid")(x)
x2 = layers.Dense(128, activation="sigmoid")(x1)
outputs = layers.Dense(1, activation="sigmoid")(x2)
model = keras.Model(inputs=inputs, outputs=outputs, name="fully_connected")

# Restore the weights
model.load_weights('../tensorflow/checkpoints/my_checkpoint-20220607-111315')
model.summary()
json_model = []


# print(json_model)
for i in range(2, 3):

    print("\n###\n\n DataSet: ", i, "\n\n")
    dataset_number = i
    dataset_path = "../resources/dataSet" + (str)(dataset_number) + "/"

    # reading in from file
    dataSetX = np.load(dataset_path + "dataSetArray.npy")
    dataSetY = np.load(dataset_path + "dataSetArraysOutput.npy")

    print("Size: ", dataSetX.size, ", max value= ", np.max(dataSetX), "\n\n")
    nr_of_imgs = (int)(dataSetX.size/1200)
    dataSetX = (dataSetX.reshape(nr_of_imgs, 1200) /
                (np.max(dataSetX))).astype("float32")
    dataSetY = dataSetY.astype("float32")

    results = model.predict(dataSetX, batch_size=None, verbose=0, steps=None, callbacks=None
                            )

print(results)
np.savetxt("results.txt", results)