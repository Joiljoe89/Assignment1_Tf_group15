#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:13:25 2018

@author: muralikrishna
"""

# Testing at frame level

import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
import pandas as pd # data frame
import numpy as np # matrix math
from scipy.io import wavfile # reading the wavfile
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
from scipy.fftpack import dct



indir='/home/muralikrishna/Desktop/DL_project/Songs_data/male_female/lata/'

for root, dirs, filenames in os.walk(indir):
    for f in sorted(filenames[:1]):
        print('Selected file:',f)
        fs, signal = wavfile.read(indir+path_file)
        print("File being read: ",path_file)
        f = os.path.splitext(path_file)[0]
        length=len(signal)
        window_hop_length=0.05 #10ms change here
        window_size=0.2 #25 ms,change here
        overlap=int(fs*window_hop_length)
        framesize=int(window_size*fs)
        number_of_frames=int(np.floor(length/overlap))
        frame_start=0
        frame_end=framesize
        print('The frame size is: ',framesize)
        frames=np.array([])
        for i in range(0,number_of_frames): 
            frame_start=frame_start+overlap
            frame_end=frame_start+framesize
            if frame_end>length:
                break

            frame=signal[frame_start:frame_end]
            frame=np.array(frame)
            frame_class=test_frame(frame)




        
       