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
model.summary()

# Instantiate a loss function.
#loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn = tf.losses.mae
# Restore the weights
#model.load_weights('./checkpoints/my_checkpoint-20220405-100502')
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"],
)
# saving the model during runtime and after finishing
checkpoint_path = "model_checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
#logging data
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = profile_logs, histogram_freq = 1, profile_batch = '500,520')

# DATA SET

for i in range(2, 6):

    print("\n###\n\n DataSet: ", i, "\n\n")
    dataset_number = i
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



    print("################\nNr of images: ", nr_of_imgs)

    # TRAINING
    history = model.fit(dataSetX,
                        dataSetY,
                        batch_size=5,
                        epochs=100,
                        validation_split=0.2,
                        verbose=2,
                        callbacks=[cp_callback, tboard_callback],
                        shuffle=True)
    test_scores = model.evaluate(dataSetX, dataSetY, verbose=1)

    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint-'+ name_specific)
    model.save('saved_model/my_model-'+ name_specific)

    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
