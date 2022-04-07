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
x = layers.Dense(300, activation="sigmoid")(inputs)
x2 = layers.Dense(300, activation="sigmoid")(x)
outputs = layers.Dense(1, activation="sigmoid")(x2)
model = keras.Model(inputs=inputs, outputs=outputs, name="fully_connected")
model.summary()

# Instantiate a loss function.
#loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn = tf.losses.mae
# Restore the weights
model.load_weights('./checkpoints/my_checkpoint-20220404-185619')
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

# DATA SET

dataset_number = 2
dataset_path = "../resources/dataSet" + (str)(dataset_number) + "/"

# reading in from file
dataSetX = np.load(dataset_path + "dataSetArray.npy")
dataSetY = np.load(dataset_path + "dataSetArraysOutput.npy")

print("Size: ", dataSetX.size, ", max value= ", np.max(dataSetX), "\n\n")
nr_of_imgs = (int)(dataSetX.size/300)
dataSetX = (dataSetX.reshape(nr_of_imgs, 300) /
            (np.max(dataSetX))).astype("float32")
dataSetY = dataSetY.astype("float32")
(dataSetX, dataSetY) = randomize(dataSetX, dataSetY)

nr_of_test=10
prediction = model.predict(dataSetX[:10])
print("Prediction       Labels")

for i in range(nr_of_test):
    print(prediction[i], "      ", dataSetY[i])






# Save the weights
model.save_weights('./checkpoints/my_checkpoint-'+ name_specific)
model.save('saved_model/my_model-'+ name_specific)