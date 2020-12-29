#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates MFCC for 2 class songs/ saves as dat file
"""
import pandas as pd # data frame
import numpy as np # matrix math
from scipy.io import wavfile # reading the wavfile
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
#from tqdm import tqdm # progress bar
import matplotlib.pyplot as plt # to view graphs

# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
import librosa
import librosa.display
import numpy as np

indir = '/home/muralikrishna/Desktop/DL_project/songs_class1'   # Directory of class1 wav file
os.chdir(indir)
for root, dirs, filenames in os.walk(indir):
    
    print('Total songs in class1: ',len(filenames))
    for f in sorted(filenames):
        
        path_file = f
        x, fs = librosa.load(path_file)
        
        f = os.path.splitext(path_file)[0]  # gives first name of file without extension
    
#x, fs = librosa.load('/home/muralikrishna/Desktop/DL_project/songs/01 - Thattathin Marayathu - Anuraagathin Velayil [Maango.info].wav')
        D = librosa.stft(x,882*2,882,882*2,scipy.signal.hamming)
        D = np.abs(D)**2
        S = librosa.feature.melspectrogram(S=D,n_mels=40)
        S=librosa.power_to_db(S,ref=np.max)
        fb=np.transpose(fb)
       
'''
        
        k = fb.shape[0]
        buf=[]
        cl = 10 #+/- frames to consider (Change to 3 or 5) for context segmenting
        for i in range(k):
            if (i>cl-1) & (i<k-cl):  # 9<i<90 / starts from 0 to 99
                buf.append(fb[i-cl:i+cl+1])                       
        buf = np.array(buf)
        buf = np.reshape(buf,(buf.shape[0],buf.shape[1]*buf.shape[2]))
        N,D = buf.shape

        #adding the label in the last column; 0 for class A, 1 for class E, 2 for class M, 3 for class N
        label_index = 0
        
        buf2 = label_index*np.ones((N,D+1))
        buf2[:,:-1] = buf 

        buf2 = np.matrix(buf2)
        #print(buf2.shape)
        #train.append(buf2)
        print('For song: {',f,'} The feature size: ',buf2.shape)
        buf2.dump('/home/muralikrishna/Desktop/DL_project/features_class1/'+f+'.dat')
        print('Size after appending lables:',buf2.shape)

#########################################For class2

indir = '/home/muralikrishna/Desktop/DL_project/songs_class2'   # Directory of class1 wav file
os.chdir(indir)
frame_size = 16e-3
frame_stride = 10e-3


for root, dirs, filenames in os.walk(indir):
    print('Total songs in class2: ',len(filenames))
    for f in sorted(filenames):
        
        path_file = f
        x, fs = librosa.load(path_file)
        
        f = os.path.splitext(path_file)[0]  # gives first name of file without extension
    
#x, fs = librosa.load('/home/muralikrishna/Desktop/DL_project/songs/01 - Thattathin Marayathu - Anuraagathin Velayil [Maango.info].wav')
        fb = librosa.feature.mfcc(x, sr=fs,n_mfcc=40, hop_length=int(0.010*fs), n_fft=int(0.025*fs))  # Get MFEC and MFCC using only MFEC
        fb=np.transpose(fb)
        print('For song: {',f,'} The feature size: ',fb.shape)

        
        k = fb.shape[0]
        buf=[]
        cl = 10 #+/- frames to consider (Change to 3 or 5) for context segmenting
        for i in range(k):
            if (i>cl-1) & (i<k-cl):  # 9<i<90 / starts from 0 to 99
                buf.append(fb[i-cl:i+cl+1])                       
        buf = np.array(buf)
        buf = np.reshape(buf,(buf.shape[0],buf.shape[1]*buf.shape[2]))
        N,D = buf.shape

        #adding the label in the last column; 0 for class A, 1 for class E, 2 for class M, 3 for class N
        label_index = 1
        
        buf2 = label_index*np.ones((N,D+1))
        buf2[:,:-1] = buf 

        buf2 = np.matrix(buf2)
        #print(buf2.shape)
        #train.append(buf2)
        # Save all features in same file
        print('For song: {',f,'} The feature size: ',buf2.shape)
        buf2.dump('/home/muralikrishna/Desktop/DL_project/features_class1/'+f+'.dat')
        print('Size after appending lables:',buf2.shape)

#################################

'''
