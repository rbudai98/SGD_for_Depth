# Python program to read
# json file


from cgi import FieldStorage
import json
import cv2
import numpy as np
from tempfile import TemporaryFile
from numpy import byte, savetxt
import array
import numpy as np
sizees = [640, 320, 160, 80, 40, 20]

nrOfCroppedFile = 0
outputArray = np.array([])

# Opening JSON file
for k in range(5, 6):

    fileName = "ir_" + (str)(k) + ".json"
    f = open(fileName)

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
        size = (80, 60)
    elif abs(x2-x1) > 40:
        size = (40, 30)
    else:
        size = (20, 15)
    print(k, ". ", x2-x1, y2-y1, "size=",size)

    #IR frame
    irFileName = "../ir/ir_" + (str)(k) + ".jpg"
    irFrame = cv2.imread(irFileName)

    #depth frame
    depthFileName = "../depth/depth_" + (str)(k) + ".raw"
    f = open(depthFileName, "r")
    data = f.buffer.read(640*480*2)
    #reading in into cv2:Mat object the frame
    depthFrame = np.frombuffer(data, dtype=np.uint16).reshape((480,640, 1))
    #converting the frame into 8 bit array (255/4055) = alpha, hence deft = src*alpha
    depthFrame = cv2.convertScaleAbs(depthFrame, alpha=0.063)
    cv2.waitKey(0)
    for i in range ((int)((640-size[0])/(size[0]/2)+1)):
        for j in range ((int)((480-size[1])/(size[1]/2)+1)):
            x_cord = (int)((i)*(size[0]/2))
            y_cord = (int)((j)*(size[1]/2))

            #cropping the regions of the image
            irCrop = irFrame[y_cord:((y_cord+size[1])), x_cord:(x_cord+size[0])]
            depthCrop = depthFrame[y_cord:((y_cord+size[1])), x_cord:(x_cord+size[0])]
            #resizing to have the same size
            irCrop = cv2.resize(irCrop, (20,15),interpolation=cv2.INTER_LINEAR)
            depthCrop = cv2.resize(depthCrop, (20,15),interpolation=cv2.INTER_LINEAR)

            #calculate the next region to check if it includes the right boundbox
            x_cord_next = (int)((i+1)*(size[0]/2))
            y_cord_next = (int)((j+1)*(size[1]/2))

            #if contains we have as output 1
            if x1 >= x_cord and x1 < x_cord_next and y1 >= y_cord and y1 < y_cord_next:
                #found. right bounding box
                irCroppedFileName = "../ir_cropped/test_ir_"+(str)(nrOfCroppedFile)+".jpg"
                with open("../depth_cropped/depth_data_"+(str)(nrOfCroppedFile)+".raw", "wb") as file:
                    file.write(bytearray(depthCrop))
                file.close()
                cv2.imwrite(irCroppedFileName, irCrop)
                outputArray = np.append(outputArray,1)
            #in other case 0
            else:
                #not found, not rigth bounding box
                irCroppedFileName = "../ir_cropped/test_ir_"+(str)(nrOfCroppedFile)+".jpg"
                with open("../depth_cropped/depth_data_"+(str)(nrOfCroppedFile)+".raw", "wb") as file:
                    file.write(bytearray(depthCrop))
                file.close()
                cv2.imwrite(irCroppedFileName, irCrop)
                outputArray = np.append(outputArray,0)
    
            nrOfCroppedFile = nrOfCroppedFile + 1
print(outputArray)
    

# cv2.waitKey(0)
# Closing file
# f.close()