import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical


import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def data(f):
    print("Reading in and transforming data...")

    
    #df = np.load(f)
    #data = df.astype(np.float32)

    df = pd.read_csv(f)
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, :-1] #first column contains the label, rest columns contain the pixel values.
    X = np.array(X)
    Y = data[:, -1] #labels
    return X, Y

def data2(f):
    print("Reading in and transforming data...")

    
    df = np.load(f)
    data = df.astype(np.float32)

    #df = pd.read_csv(f)
    #data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, :-1] #first column contains the label, rest columns contain the pixel values.
    X = np.array(X)
    Y = data[:, -1] #labels
    return X, Y

# by default Keras wants one-hot encoded labels
def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 4))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

#train data
X, Y = data('train1.csv')
Y = to_categorical(Y,4)

mu = X.mean(axis=0)
std = X.std(axis=0)
np.place(std, std == 0, 1) #so that the standard deviation never becomes zero.
X = (X - mu) / std # normalize the data
X = X.reshape(X.shape[0], 21, 40, 1)

#K = len(set(Y))
#Y = y2indicator(Y)


model = Sequential()
model.add(Conv2D(64, kernel_size=(4, 3), activation='relu', input_shape=(21, 40, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(2, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(1, 3), activation='relu'))

model.add(Flatten())
#model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])
'''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''

r = model.fit(X, Y, batch_size=100, epochs=10, validation_split = 0, verbose=1)


# plot some data

# accuracies
plt.figure(1)
plt.plot(r.history['acc'], label='acc')
#plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()
plt.savefig('figures/CNN/acc/keras_CNN_A2_512.png' , bbox_inches = 'tight')

plt.figure(2)
plt.plot(r.history['loss'], label='loss')
#plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig('figures/CNN/loss/keras_CNN_A2_512.png' , bbox_inches = 'tight')

model.save('models/CNN/keras_CNN_A2_512.h5')

miss = 0
hit = 0
indir = '/home/iit/Desktop/kishalaya/FAU_AEC/fold1_test/'
os.chdir(indir)
for rootDirPath, subDir, files in os.walk(indir):
    for fn in (files):
        x,y = data2(fn)
        x = (x - mu) / std
        x = x.reshape(x.shape[0], 21, 40, 1)
        labels = model.predict(x)
        #print(fn)
        #print(labels.mean(axis=0))
        k = labels.mean(axis=0)
        P1 = np.argmax(k, axis=0)
        #print(P1)
        #print(y[0])

        if(P1 == y[0]):
            hit+=1            
        else:
            miss+=1
        #print(fn)
        
score = (hit*100)/(hit+miss)
print(hit)
print(miss)
print(score)
        
        
