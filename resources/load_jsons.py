# Python program to read
# json file
'''
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''
import os
from cgi import FieldStorage
import json
import cv2
import numpy as np
from tempfile import TemporaryFile
from numpy import byte, savetxt
import array
import numpy as np
import tensorflow as tf
import re

dataSetArraysOk = np.array([])

sizees = [640, 320, 160, 80, 40, 20]

nrOfCroppedFileOk = 0
nrOfCroppedFileNotOk = 0
outputArray = np.array([])


nr_of_dataset = 4
dataset_path = "dataSet" + (str)(nr_of_dataset) + "/"
ir_path = "ir/ir_"
depth_path = "depth/depth_"
json_path = "jsons/ir_"

for filename in os.listdir(os.getcwd() + "/" + dataset_path + "jsons"):
    with open(os.path.join(os.getcwd()+"/" + dataset_path + "jsons", filename), 'r') as f:  # open in readonly mode

        file_number = os.path.splitext(filename)[0]

        emp_str = ""
        for m in file_number:
            if m.isdigit():
                emp_str = emp_str + m
        k = (int)(emp_str)

        f = open(dataset_path + json_path + (str)(k) + ".json")

        # returns JSON object as
        # a dictionary
        data = json.load(f)

        # Iterating through the json
        # list
        # for i in data['emp_details']:
        # print(data["shapes"])
        x1 = round(data["shapes"][0]['points'][0][0])
        y1 = round(data["shapes"][0]['points'][0][1])
        x2 = round(data["shapes"][0]['points'][1][0])
        y2 = round(data["shapes"][0]['points'][1][1])

        if (y2 < y1):
            y3 = y2
            y2 = y1
            y1 = y3
        if (x2 < x1):
            x3 = x2
            x2 = x1
            x1 = x3

        if (abs(x2-x1) > 320):
            size = (640, 480)
        elif abs(x2-x1) > 160:
            size = (320, 240)
        elif abs(x2-x1) > 80:
            size = (160, 120)
        elif abs(x2-x1) > 40:
            size = (80, 60)
        elif abs(x2-x1) > 20:
            size = (40, 30)
        else:
            size = (20, 15)
        print(k, ". ", x2-x1, y2-y1, "size=", size)

        # IR frame
        irFileName = dataset_path + ir_path + (str)(k) + ".png"
        irFrame = cv2.imread(irFileName)

        # depth frame
        depthFileName = dataset_path + depth_path + (str)(k) + ".bin"
        f = open(depthFileName, "r")

        data = f.buffer.read(640*480*2)
        # reading in into cv2:Mat object the frame
        depthFrame = np.frombuffer(
            data, dtype=np.uint16).reshape((480, 640, 1))
        # converting the frame into 8 bit array (255/4055) = alpha, hence deft = src*alpha

        #depthFrame = cv2.convertScaleAbs(depthFrame, alpha=0.063)
        depthFrame = depthFrame * 255/1023
        depthFrame = depthFrame.astype(np.uint8)

        for i in range((int)((640-size[0])/(size[0]/2)+1)):
            for j in range((int)((480-size[1])/(size[1]/2)+1)):
                x_cord = (int)((i)*(size[0]/2))
                y_cord = (int)((j)*(size[1]/2))

                # cropping the regions of the image
                irCrop = irFrame[y_cord:(
                    (y_cord+size[1])), x_cord:(x_cord+size[0])]
                depthCrop = depthFrame[y_cord:(
                    (y_cord+size[1])), x_cord:(x_cord+size[0])]
                # resizing to have the same size
                irCrop = cv2.resize(
                    irCrop, (20, 15), interpolation=cv2.INTER_LINEAR)
                depthCrop = cv2.resize(
                    depthCrop, (20, 15), interpolation=cv2.INTER_LINEAR)

                # calculate the next region to check if it includes the right boundbox
                x_cord_next = (int)((i+1)*(size[0]/2))
                y_cord_next = (int)((j+1)*(size[1]/2))

                # if contains we have as output 1
                if x1 >= x_cord and x1 < x_cord_next and y1 >= y_cord and y1 < y_cord_next:
                    # found. right bounding box
                    # compensate found boxes, copy object multipple times
                    for l in range((int)((640-size[0])/(size[0]/2)+1)*(int)((480-size[1])/(size[1]/2)+1) - 1):
                        #irCroppedFileName = "../ir_cropped/test_ir_"+(str)(nrOfCroppedFileOk)+".jpg"
                        # with open("../depth_cropped/1/1_image_"+(str)(nrOfCroppedFileOk)+".jpg", "wb") as file:
                        #    file.write(bytearray(depthCrop))
                        # file.close()
                        #cv2.imwrite(irCroppedFileName, irCrop)
                        outputArray = np.append(outputArray, 1)

                        depthCrop = np.reshape(np.asarray(
                            depthCrop), (1, 300)).flatten()
                        dataSetArraysOk = np.append(dataSetArraysOk, depthCrop)

                        nrOfCroppedFileOk = nrOfCroppedFileOk + 1

                # in other case 0
                else:
                    # not found, not rigth bounding box
                    #irCroppedFileName = "../ir_cropped/test_ir_"+(str)(nrOfCroppedFileNotOk)+".jpg"
                    # with open("../depth_cropped/0/0_image_"+(str)(nrOfCroppedFileNotOk)+".jpg", "wb") as file:
                    #    file.write(bytearray(depthCrop))
                    # file.close()
                    #cv2.imwrite(irCroppedFileName, irCrop)

                    outputArray = np.append(outputArray, 0)
                    depthCrop = np.reshape(np.asarray(
                        depthCrop), (1, 300)).flatten()
                    dataSetArraysOk = np.append(dataSetArraysOk, depthCrop)

                    nrOfCroppedFileNotOk = nrOfCroppedFileNotOk + 1


np.save(dataset_path + "dataSetArray", dataSetArraysOk)
np.save(dataset_path + "dataSetArraysOutput", outputArray)


myDataSet = tf.data.Dataset.from_tensor_slices(
    np.reshape(dataSetArraysOk, (-1, 300)))
myDataSetLabel = tf.data.Dataset.from_tensor_slices(outputArray)

print("Dataset cardinality: ", myDataSet.cardinality().numpy())
print("Dataset label cardinality: ", myDataSetLabel.cardinality().numpy())


# print(outputArray)
savetxt(dataset_path + "outputArray.csv", outputArray.astype(np.uint8), delimiter=" ")
