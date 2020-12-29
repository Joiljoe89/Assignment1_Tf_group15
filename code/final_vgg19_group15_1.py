# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 22:05:53 2018
Assignment2
Group15
@author: Joe , Murali & Soma
"""
# keras imports
from keras.applications.vgg19 import preprocess_input
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras import models, layers
from keras.optimizers import SGD, RMSprop,adam
from keras.models import Sequential, model_from_json

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from numpy import *
from PIL import Image, ImageFile
import numpy
#import glob
import h5py
import os,os.path
#import datetime
#import time

##############################################################################
#import 'pickled' file
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#unpacking training and test data
#test = unpickle("E:\joe\Assignment_02_Data\Assignment_Data\Test_Data\Test_Data\images1.pickle")
test = unpickle("/home/dlgroup2/dl_assignment/assignment2/Assignment_Data/Test_Data/images1.pickle")

X_test = test.reshape(4033, img_rows, img_cols, 1)

# Data preprocessing
X_test = X_test.astype('float32')
mean1 = numpy.mean(X_test) # for finding the mean for centering  to zero

X_test -= mean1

##############################################################################
# input image dimensions
img_rows, img_cols, img_channels = 312, 372, 1
epochs=1
batch_size1=40
target_names = ['class 1', 'class 2']

##############################################################################
def larger_model():
	# create model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(img_rows,img_cols,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    learning_rate=0.01
    decay_rate=learning_rate/epochs
    momentum=0.6
    sgd=SGD(lr=learning_rate,momentum=momentum,decay=decay_rate,nesterov=True)
	
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', 
                  metrics=['accuracy'])
    return model
##############################################################################

# build the model
model = larger_model()
model.load_weights("/home/dlgroup2/dl_assignment/assignment2/assignmnt2_3.h5", by_name="False") 

##############################################################################
# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(test_data)
print(Y_pred)
'''
y_pred = numpy.argmax(Y_pred, axis=1)
#print(y_pred)
p=model.predict_proba(test_data) # to predict probability
print(classification_report(numpy.argmax(test_data,axis=1), y_pred,
                            target_names=target_names))
print(confusion_matrix(numpy.argmax(test_data,axis=1), y_pred))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5 and saving the weights
#model.save_weights("model.h5")
#print("Saved model to disk")
    
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/dlgroup2/dl_assignment/assignment2/assignmnt2_3.h5", by_name="False")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                     metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=2)
print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''
