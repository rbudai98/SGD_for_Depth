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
import matplotlib.pyplot as plt

dataSetArraysOk = np.array([])

sizees = [640, 320, 160, 80, 40]

nrOfCroppedFileOk = 0
nrOfCroppedFileNotOk = 0
outputArray = np.array([])

for i in range (2,3):
    nr_of_dataset = i
    print("DataSet: ", nr_of_dataset)
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

            if (abs(x2-x1) > 320 or abs(y2-y1)>240):
                size = (640, 480)
            elif (abs(x2-x1) > 160 or abs(y2-y1)>120):
                size = (320, 240)
            elif (abs(x2-x1) > 80 or abs(y2-y1)>60):
                size = (160, 120)
            elif (abs(x2-x1) > 40 or abs(y2-y1)>30):
                size = (80, 60)
            else:
                size = (40, 30)
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
                #data, dtype=np.uint16).reshape((480, 640, 1))
                data, dtype=np.uint16).reshape((480, 640))
            # converting the frame into 8 bit array (255/4055) = alpha, hence deft = src*alpha
            
            #depthFrame = cv2.convertScaleAbs(depthFrame, alpha=0.063)
            
            # print((depthFrame.dtype))
            # depthFrame = (depthFrame/4).astype('uint8')
            #depthFrame = depthFrame.astype(np.uint8)
            
            
            '''heatmapshow = None
            print("Max of matrices: ", np.max(depthFrame), ", min of matrices: ", np.min(depthFrame))
            heatmapshow = cv2.normalize(depthFrame, heatmapshow, alpha=np.min(depthFrame), beta=np.max(depthFrame), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            cv2.imshow("Depth map", heatmapshow)
            cv2.imshow("IR map", irFrame)
            cv2.waitKey(0)'''

    

            for i in range((int)((640-size[0])/(size[0]/2)+1)):
                for j in range((int)((480-size[1])/(size[1]/2)+1)):
                    x_cord = (int)((i)*(size[0]/2))
                    y_cord = (int)((j)*(size[1]/2))

                    # cropping the regions of the image
                    irCropOriginal = irFrame[y_cord:(
                        (y_cord+size[1])), x_cord:(x_cord+size[0])]
                    depthCropOriginal = depthFrame[y_cord:(
                        (y_cord+size[1])), x_cord:(x_cord+size[0])]
                    # resizing to have the same size
                    irCrop = cv2.resize(
                        irCropOriginal, (40, 30), interpolation=cv2.INTER_LINEAR)
                    depthCrop = cv2.resize(
                        depthCropOriginal, (40, 30), interpolation=cv2.INTER_LINEAR)

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
                            
                            #cv2.imshow("ir", irCropOriginal)
                            #cv2.imshow("depth", depthCropOriginal) 
                            #cv2.waitKey(0)

                            outputArray = np.append(outputArray, 1)

                            depthCrop = np.reshape(np.asarray(
                                depthCrop), (1, 1200)).flatten()
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
                            depthCrop), (1, 1200)).flatten()
                        dataSetArraysOk = np.append(dataSetArraysOk, depthCrop)

                        nrOfCroppedFileNotOk = nrOfCroppedFileNotOk + 1


    np.save(dataset_path + "dataSetArray", dataSetArraysOk)
    np.save(dataset_path + "dataSetArraysOutput", outputArray)

    np.savetxt("inputData.txt", dataSetArraysOk)


    myDataSet = tf.data.Dataset.from_tensor_slices(
        np.reshape(dataSetArraysOk, (-1, 1200)))
    myDataSetLabel = tf.data.Dataset.from_tensor_slices(outputArray)

    print("Dataset cardinality: ", myDataSet.cardinality().numpy())
    print("Dataset label cardinality: ", myDataSetLabel.cardinality().numpy())


    # print(outputArray)
    savetxt(dataset_path + "outputArray.csv", outputArray.astype(np.uint8), delimiter=" ")
