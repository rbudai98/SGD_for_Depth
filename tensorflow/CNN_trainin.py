import sys
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


img_height = 20
img_width = 15

inputs = keras.Input(shape=(300,))
inputs.shape

x = layers.Dense(300, activation="relu")(inputs)
outputs = layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

model.summary()

dataset_number = 2
dataset_path = "../resources/dataSet" + (str)(dataset_number) + "/"

# reading in from file
dataSetX = np.load(dataset_path + "dataSetArray.npy")
dataSetY = np.load(dataset_path + "dataSetArraysOutput.npy")

nr_of_imgs = (int)(dataSetX.size/300)
print("Nr of lengths: ", nr_of_imgs)


dataSetX = dataSetX.reshape(nr_of_imgs, 300).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

#saving the model during runtime and after finishing
checkpoint_path = "model_checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


history = model.fit(dataSetX, 
                    dataSetY,
                    batch_size=10,
                    epochs=10,
                    validation_split=0.2, 
                    verbose=1,
                    callbacks=[cp_callback])

test_scores = model.evaluate(dataSetX, dataSetY, verbose=1)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])


