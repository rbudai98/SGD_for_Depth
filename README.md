# Tensorflow for depth frames
Depth frame processing with reduced network dimensions, and dataset preparation for tensorlfow training


### ```resources``` directory:
As structure contains multiple ```dataSet<nr>``` directories that include:
  * ```dataSet<nr>/depth```            saved depth frames for training
  * ```dataSet<nr>/ir```               saved ir frames for labeling
  * ```dataSet<nr>/jsons```            generated label jsons 
  * ```load_jsons.py```           dataset generator from the previopus directories 
  * ```dataSet<nr>/*.npy```                   generated arrays from the dataset that is going to be used for trtaining
  
  ### ```tensorflow``` directory:
* ```checkpoints```        contains the last trained network's data and loads them at further training
* ```CNN_train.py```       tensorflow model definition and training
  
