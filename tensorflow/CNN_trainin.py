import sys
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn import utils


img_height = 20
img_width = 15


def shuffle(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

### MODEL

inputs = keras.Input(shape=(300,))
inputs.shape
x = layers.Dense(300, activation="sigmoid")(inputs)
x2 = layers.Dense(300, activation="sigmoid")(x)
outputs = layers.Dense(1)(x2)
model = keras.Model(inputs=inputs, outputs=outputs, name="fully_connected")
model.summary()
# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=2e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Restore the weights
#model.load_weights('./checkpoints/my_checkpoint')
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

### DATA SET

for i in range(2,5):

    print("\n###\n\n DataSet: ", i, "\n\n")
    dataset_number = i
    dataset_path = "../resources/dataSet" + (str)(dataset_number) + "/"

    # reading in from file
    dataSetX = np.load(dataset_path + "dataSetArray.npy")
    dataSetY = np.load(dataset_path + "dataSetArraysOutput.npy")

    print("Size: ",dataSetX.size)
    nr_of_imgs = (int)(dataSetX.size/300)
    dataSetX = dataSetX.reshape(nr_of_imgs, 300)


    print("################\nNr of images: ", nr_of_imgs)


    ### TRAINING
    history = model.fit(dataSetX, 
                        dataSetY,
                        batch_size=5,
                        epochs=100,
                        validation_split=0.2,   
                        verbose=2,
                        callbacks=[cp_callback],
                        shuffle=True)
    test_scores = model.evaluate(dataSetX, dataSetY, verbose=1)

    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint')
    model.save('saved_model/my_model')

    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])


