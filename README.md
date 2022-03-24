# Tensorflow for depth frames
Depth frame processing with reduced network dimensions, and dataset preparation for dataSet


### ```resources``` directory:
As structure contains multiple ```dataSet<nr>``` directories that include:
  * ```depth```            saved depth frames for training
  * ```ir```               saved ir frames for labeling
  * ```jsons```            generated label jsons 
  *```load_jsons.py```           dataset generator from the previopus directories 
  * ```.npy```                   generated arrays from the dataset that is going to be used for trtaining
  
  ### ```tensorflow``` directory:
* ```checkpoints```        contains the last trained network's data and loads them at further training
* ```CNN_train.py```       tensorflow model definition and training
  
