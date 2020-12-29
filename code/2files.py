#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:38:36 2018

@author: muralikrishna
"""
def data(f):
    print("Loading the train data frm file...",f)    
    df = np.load(f)
    data = df.astype(np.float32)

    np.random.shuffle(data)
    X = data[:, :-1] #first column contains the label, rest columns contain the pixel values.
    X = np.array(X)
    Y = data[:, -1] #labels
    return X, Y

traindata='/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/trainingdata/'
nb_epoch=5
for e in range(1,nb_epoch):
    print("epoch %d" % e)
    for f in range(1,3):
        path='/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/trainingdata/train'+str(f)+'.dat'
        print(path)
        #X, Y = data('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/trainingdata/train'+str(f)+'.dat')
   
        #model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)