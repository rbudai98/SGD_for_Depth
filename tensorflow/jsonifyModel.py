import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import datetime
import json
#from sklearn import utils


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
model.load_weights('./checkpoints/my_checkpoint-20220601-201948')
model.summary()
json_model = []


# for layer in model.layers[1:]:
#     json_model.append({
#         "name": layer.name,
#         "weights": model.get_weights()[poz].tolist(),
#         "bias": model.get_weights()[poz+1].tolist()
#     })
#     poz = poz+2

# with open("JSON_Model.json", 'w') as json_file:
#     json.dump(json_model, json_file, 
#                         indent=0,  
#                         separators=(',',':'))

# print(json_model)
poz = 0
for layer in model.layers[1:]:
        
    f1 = open("json_model/layer_"+(str)(poz)+"_weights.txt", "w")
    f2 = open("json_model/layer_"+(str)(poz)+"_bias.txt", "w")
    array = model.get_weights()[poz*2]
    array2 = model.get_weights()[poz*2+1]
    
    array=np.reshape(array,(1,-1)).tolist()
    array2=np.reshape(array2,(1,-1)).tolist()

    # for d in array:
    #     f1.write(f"{d}\n")
    # f1.close()   
    # for d in array2:
    #     f2.write(f"{d}\n")
    # f2.close()

    np.savetxt("json_model/layer_"+(str)(poz)+"_weights.txt", array, delimiter=" ")
    np.savetxt("json_model/layer_"+(str)(poz)+"_bias.txt", array2, delimiter=" ")

    poz = poz+1