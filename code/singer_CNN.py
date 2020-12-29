#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:33:13 2018

@author: muralikrishna # Traini data is from desktop system. Check the train data file.

"""
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Activation
from keras.utils import to_categorical

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def data(f):
    print("Loading the train data...")    
    df = np.load(f)
    data = df.astype(np.float32)

    np.random.shuffle(data)
    X = data[:, :-1] #first column contains the label, rest columns contain the pixel values.
    X = np.array(X)
    Y = data[:, -1] #labels
    return X, Y


# by default Keras wants one-hot encoded labels
def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 7))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

#train data
K=5 # Number of classes
cl=3

X, Y = data('/home/groupdl2/singer_identification/MFECtrain5.dat')
Y = to_categorical(Y,K)

mu = X.mean(axis=0)
std = X.std(axis=0)
np.place(std, std == 0, 1) #so that the standard deviation never becomes zero.
X = (X - mu) / std # normalize the data
#X = X.reshape(X.shape[0], 21, 40, 1)
X = X.reshape(X.shape[0], 2*cl+1, 40, 1)
input_shape=X.shape[1:]

# Now running with cl=10
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
#model.add(Dropout(0.25))
model.add(Dense(2048, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(2048, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(256, activation='sigmoid'))  #2048
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(256, activation='sigmoid'))  #2048
model.add(BatchNormalization())
model.add(Dropout(0.25))



model.add(Dense(K,activation='softmax'))

sgd = SGD(lr=0.0005, decay=1e-7, momentum=0.9)

model.compile(loss=keras.losses.categorical_crossentropy, # Changed to adam
              optimizer='adam',
              metrics=['accuracy'])

print('Now traning the CNN....')
r = model.fit(X, Y, batch_size=100, epochs=5, validation_split = 0, verbose=1,shuffle=True)


model.save('/home/groupdl2/singer_identification/singer_MFECCNN5class.h5')

########################

'''
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
#model.add(Dropout(0.25))
model.add(Dense(2048, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(2048, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(256, activation='sigmoid'))  #2048
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(256, activation='sigmoid'))  #2048
model.add(BatchNormalization())
model.add(Dropout(0.25))



model.add(Dense(K,activation='softmax'))

sgd = SGD(lr=0.0005, decay=1e-7, momentum=0.9)

model.compile(loss=keras.losses.categorical_crossentropy, # Changed to adam
              optimizer='adam',
              metrics=['accuracy'])

print('Now traning the CNN....')
r = model.fit(X, Y, batch_size=100, epochs=5, validation_split = 0, verbose=1,shuffle=True)
'''
