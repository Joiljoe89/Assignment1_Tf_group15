# keras imports
from keras.applications.vgg19 import VGG19, preprocess_input
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
# input image dimensions
img_rows, img_cols, img_channels = 312, 372, 1
epochs=1
batch_size1=40
target_names = ['class 1', 'class 2']

# Path to the folder that is containing all the folders correspond to different class
path = '/home/dlgroup2/dl_assignment/assignment2/converted1/'

list_files = os.listdir(path) # List_Files contains names of different folders inside 
print (list_files)
number_class = len(list_files) # Number of classes
print (number_class) 
num_samples = 0

#Processing images for each class
global combine_matrix # declaration of variable
n_da_per_class = [] # create list for containing size(total number of images) of each class
check_first = 1

for class_n in list_files:
    class_list = os.listdir(path + '/'+ class_n)
    #print(class_list)
    class_im = array(Image.open(path + '/'+ class_n + '/' + class_list[0]))
    print(class_im)
    # open one image to get size
    image_m,image_n = class_im.shape[0:2] # get the size of the one image in 
    #Class
    class_imnbr = len(class_list) # get the number of images corresponding 
    #to each class
    n_da_per_class.append(class_imnbr) # append size of folder in
    #n_da_per_class list
    num_samples = num_samples + class_imnbr
    class_immatrix_n = array([array(Image.open(path + '/'+ class_n +'/' + class_im2)).flatten() for class_im2 in class_list],'f')
    
    print (size(class_immatrix_n))
    if check_first == 1:
        check_first = 0
        combine_matrix = class_immatrix_n
    else:
        combine_matrix  = numpy.concatenate((combine_matrix, class_immatrix_n), axis=0)  
        # Combining each matrix
################################################    
label=numpy.ones((num_samples,),dtype = int) # number of elements in label = number of samples
class_ind = 0; # initializing class label
number = 0
sum_number = 0
for data_p_c in n_da_per_class:  # for each image, label is assigned
    number = number + int(data_p_c)
    for i in range(sum_number, number):
        label[i] = class_ind
    class_ind = class_ind + 1
    sum_number = sum_number + int(data_p_c)

data,Label = shuffle(combine_matrix,label, random_state=2)
train_data = [data,Label]
print ((train_data)) 
(X, y) = (train_data[0],train_data[1])
# number of output classes
nb_classes = number_class

# STEP 1: split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)


# Data preprocessing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
mean1 = numpy.mean(X_train) # for finding the mean for centering  to zero
X_train -= mean1
X_test -= mean1
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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
    model.add(Dense(nb_classes, activation='softmax'))
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
y_train = y_train.reshape((-1, 1)) # -1 refers to unknown; here for each image,
# one coloumn vector is generated
model.load_weights("/home/dlgroup2/dl_assignment/assignment2/assignmnt2_4.h5", by_name="False") 
#print y_train
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=batch_size1, nb_epoch=epochs, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save_weights('assignmnt2_5.h5')

############################################################################

# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
#print(Y_pred)
y_pred = numpy.argmax(Y_pred, axis=1)
#print(y_pred)
p=model.predict_proba(X_test) # to predict probability
print(classification_report(numpy.argmax(Y_test,axis=1), y_pred,
                            target_names=target_names))
print(confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred))

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

###########################################################################
