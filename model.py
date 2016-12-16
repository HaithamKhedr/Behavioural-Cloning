from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import glob

#### generator for training data

def generate_data(img_list,Y_train,batch_size):
    while 1:
        batch_mask = np.random.choice(len(img_list),batch_size,replace = False)
        fileList = img_list[batch_mask]
        X_train  = np.array([np.array(plt.imread(file)) for file in fileList])
        X_train = X_train /255 - 0.5
        y_train = Y_train[batch_mask]
        yield(X_train,y_train)

#### Training parameters ####
batch_size = 100
epochs = 2


csv_path = './driving_log.csv'
csv = pd.read_csv(csv_path)
labels = np.array(csv['steering']) #steering angles

fileList = np.array(glob.glob('./IMG/center*.jpg'))
val_mask  = np.random.choice(len(fileList) , int(0.2 * len(fileList)) )
val_List = fileList[val_mask] #List of images paths for validation set
val_labels = labels[val_mask] # steering angles of validation set
train_List = np.delete(fileList,val_mask)
train_labels = np.delete(labels,val_mask)
validation =np.array([np.array(plt.imread(file)) for file in val_List]) #Load validation images
gen = generate_data(train_List,train_labels,batch_size = 100)
sPerEpoch =len(train_List)

reg = 0
model = Sequential()
model.add(Convolution2D(24, 5,5,border_mode='same',activation='relu',subsample=(2, 2), input_shape=(160,320,3)))
model.add(Convolution2D(36, 5,5,border_mode='same',activation='relu',subsample=(2, 2)))
model.add(Convolution2D(48, 5,5,border_mode='same',activation='relu',subsample=(2, 2)))
model.add(Convolution2D(64, 3,3,border_mode='valid',activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3,3,border_mode='valid',activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1164,activation = 'relu',W_regularizer=l2(reg)))
model.add(Dropout(0.5))
model.add(Dense(100,activation = 'relu',W_regularizer=l2(reg)))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer=Adam(1e-4),loss="mse")
model.fit_generator(gen,samples_per_epoch = sPerEpoch,validation_data=(validation,val_labels),nb_epoch=epochs,callbacks = [ModelCheckpoint('./best_model',save_best_only=True)])

model.save_weights('model.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
        json.dump(model_json,json_file)
print('Model saved')


