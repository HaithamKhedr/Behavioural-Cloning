# Behavioural-Cloning
Making a Car drive in a simulator with my driving behaviour

# Model Architecture

A deep convolutional neural network is used that consistes of 5 convolutional layers and 4 fully connected layers, this architecture was inspired by the paper(End to End Learning for Self-Driving Cars) published by Nvidia for behavioural cloning on the comma.ai dataset.The details of each of the layers is described below.

1- input shape 160X320X3

2- conv1 filter:5X5 channels:24 stride:2 activation:relu padding:same

3- conv2 filter:5X5 channels:36 stride:2 activation:relu padding:same

4- conv3 filter:5X5 channels:48 stride:2 activation:relu padding:same

5- conv4 filter:3X3 channels:63 stride:1 activation:relu padding:valid

6- Dropout with drop probability 0.2

7- conv5 filter:3X3 channels:63 stride:1 activation:relu padding:valid

8- Dropout with drop probability 0.2

9- FC1 (1164)

10- Dropout (0.5)

11- FC2 (100)

11- Dropout(0.5)

12- FC3(50)

13- FC4(10)

# Training Process

The Dataset was split into training and validation sets, the validation examples are 20 % of the training set that are chosen randomly.The learning rate is chosen after many manual trials to be 1e-4 which gave best results, the batch size is 100 which gave satisfying results and didn't need to change it, the number of epochs is set to 5 after inspecting the training progress and I found that 5 epochs achieve great results and no more epochs were needed to train.Also, adding maxpooling layers or weight regularization performed worse on the data so I just used dropout to decrease overfitting.Multiple Architectures were tested but this architecture was the best,even trying a pretrained VGG16 network didn't perform well on the dataset.

# Data Collection

The data was collected using Udacity Simulator which samples the video @10 Hz and record each frame from 3 different cameras together with the steering angle and the throttle (only the center camera was used here).I recorded 14,256 frame to use as my dataset for training and validation.Data collection was the most important part of this project.I basically drove the car for 2 laps with the car at the center of the road, and 1 lap recovering from wrong directions.After that I recorded more data at road curves and more recovery samples which the model didn't predict correctly at first.Examples of the recorded images from the center camera are shown below

![alt text](/images/center_2016_12_15_10_43_27_207.jpg)
![alt text](/images/center_2016_12_15_10_43_27_629.jpg)
