#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:41:17 2018

@author: muralikrishna
"""


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


test_model=load_model('/home/groupdl2/singer_identification/singer_MFECCNN1.h5')
indir = '/home/groupdl2/singer_identification/3Rafi_cut/'  # Directory of class1 wav file

def mfcc_features(path_file, frame_size, frame_stride):
    sample_rate, signal = wavfile.read(path_file)
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_size = 16e-3 # Frame size in seconds / can be mentioned from main function, (frame_size=frame_size)
    frame_stride = 10e-3 # stride

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) +\
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # hamming window
    frames *= np.hamming(frame_length) # Hamming window

    NFFT = 512 #FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

	#Converting to Melscale
    nfilt = 40  # Number of filters
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # This is MFEC
    
    # MFCC calculation
    num_ceps = 20
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    
    
    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*
    
    return filter_banks, mfcc

def data(f):
    print("Reading in and transforming data...")
    mat1 = np.load(f)
    #print(mat1.shape)
    np.random.shuffle(mat1)
    X = mat1[:, :-1] #Last column contains the label, rest columns contain the pixel values.
    #print(X.shape)
    Y = mat1[:, -1] #labels
    return X, Y
#indir='/home/muralikrishna/Desktop/DL_project/Songs_data/male_female/lata_test/'

os.chdir(indir)
frame_size = 16e-3
frame_stride = 10e-3

c1=0
c2=0
c3=0
c4=0
for root, dirs, filenames in os.walk(indir):
	for f in sorted(filenames):
		path_file = f
		fb,MFCC = mfcc_features(path_file, frame_size, frame_stride)  # Get MFEC and MFCC using only MFEC
		     
		f = os.path.splitext(f)[0]  # gives first name of file without extension
		print(f)

		#saving the Mel filter bank features
		df = pd.DataFrame(fb)   # Saving using pandas
		       
		#Saving the features as context segments/ Gets the context by considering 10 frames before and 10 frames after the current frame

		k = fb.shape[0]
		buf=[]
		cl = 3 #+/- frames to consider (Change to 3 or 5)
		for i in range(k):
			if (i>cl-1) & (i<k-cl):  # 9<i<90 / starts from 0 to 99
				buf.append(fb[i-cl:i+cl+1])                       
		buf = np.array(buf)
		buf = np.reshape(buf,(buf.shape[0],buf.shape[1]*buf.shape[2]))
		x=buf
		mu = x.mean(axis=0)
		std = x.std(axis=0)
		np.place(std, std == 0, 1) #so that the standard deviation never becomes zero.   
		x = (x - mu) / std
		x = x.reshape(x.shape[0], 7, 40, 1)  # We have 40 MFEC   or20 MFCC/frame with cl=2
		labels = test_model.predict(x)

		k = labels.mean(axis=0)
		P1 = np.argmax(k, axis=0)
		print('K=',k)
		if P1==0:
			print('This is Predicted as Alka') 
			c1=c1+1
		elif P1==1:            
			print('Predicted as Lata') 
			c2=c2+1
		elif P1==2:
			print('Predicted as Rafi')
			c3=c3+1 
		else:
			print('This is Music')
			c4=c4+1
print('Total files predicted as Alka:',c1,' As Lata:',c2,' As Rafi: ',c3,' As Music: ',c4)
