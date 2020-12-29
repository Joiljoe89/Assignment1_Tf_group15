#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:14:23 2018

@author: muralikrishna
"""

# importing dependencies
import pandas as pd # data frame
import numpy as np # matrix math
from scipy.io import wavfile # reading the wavfile
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
#from tqdm import tqdm # progress bar
#import matplotlib.pyplot as plt # to view graphs

# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
#import librosa # library for audio processing
import pandas as pd

cl = 1
'''
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

def normalized_fb(fb):  # Removes noise by subtracting mean
    fb1 = fb - (np.mean(fb, axis=0) + 1e-8)
    return fb1

def normalized_mfcc(mfcc):
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

#give the directory path where the .wav files are kept
indir = 'E:/singer_identification/1Alka_cut/'  # Directory of class1 wav file
os.chdir(indir)
frame_size = 16e-3
frame_stride = 10e-3


for root, dirs, filenames in os.walk(indir):
    for f in sorted(filenames):
        path_file = f
        fb,MFCC = mfcc_features(path_file, frame_size, frame_stride)  # Get MFEC and MFCC using only MFEC
        #print(fb.shape)
        
        f = os.path.splitext(f)[0]  # gives first name of file without extension
        print(f)

        #saving the Mel filter bank features
        df = pd.DataFrame(fb)   # Saving using pandas
        
        #Saving the features as context segments/ Gets the context by considering 10 frames before and 10 frames after the current frame
        
        k = fb.shape[0]
        buf=[]
        #cl = 1 #+/- frames to consider (Change to 3 or 5)
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
        print('The train data shape: ',buf2.shape)
        #train.append(buf2)

        buf2.dump('E:/singer_identification/features1/ak'+f+'.dat')
        
'''

indir = 'E:/singer_identification/1Alka_cut/'  # For class 2
os.chdir(indir)
frame_size = 16e-3
frame_stride = 10e-3
print('Reading files for class 2')

for root, dirs, filenames in os.walk(indir):
    for f in sorted(filenames):
        path_file = f
        fb,MFCC = mfcc_features(path_file, frame_size, frame_stride)  # Get MFEC and MFCC using only MFEC
        #print(fb.shape)
        
        f = os.path.splitext(f)[0]  # gives first name of file without extension
        print(f)

        #saving the Mel filter bank features
        df = pd.DataFrame(fb)   # Saving using pandas
        #df.to_csv('/home/shikha/Desktop/FAU_AEC/features_melfb2164/set3/A/'+f+'.csv',header=None, index=None) # Avoid index numbers merging with actual MFEC values
        
        #Saving the features as context segments/ Gets the context by considering 10 frames before and 10 frames after the current frame
        
        k = fb.shape[0]
        buf=[]
        #cl = 1 #+/- frames to consider (Change to 3 or 5)
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
        print('The train data shape: ',buf2.shape)
        #print(buf2.shape)
        #train.append(buf2)

        buf2.dump('E:/singer_identification/features1/ak'+f+'.dat')
'''
##############################
        
indir = '/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/3Rafi_cut/'  # For class 2
os.chdir(indir)
frame_size = 16e-3
frame_stride = 10e-3
print('Reading files for class 2')

for root, dirs, filenames in os.walk(indir):
    for f in sorted(filenames):
        path_file = f
        fb,MFCC = mfcc_features(path_file, frame_size, frame_stride)  # Get MFEC and MFCC using only MFEC
        #print(fb.shape)
       
        f = os.path.splitext(f)[0]  # gives first name of file without extension
        print(f)

        #saving the Mel filter bank features
        df = pd.DataFrame(fb)   # Saving using pandas
        #df.to_csv('/home/shikha/Desktop/FAU_AEC/features_melfb2164/set3/A/'+f+'.csv',header=None, index=None) # Avoid index numbers merging with actual MFEC values
        
        #Saving the features as context segments/ Gets the context by considering 10 frames before and 10 frames after the current frame
        
        k = fb.shape[0]
        buf=[]
        #cl = 1 #+/- frames to consider (Change to 3 or 5)
        for i in range(k):
            if (i>cl-1) & (i<k-cl):  # 9<i<90 / starts from 0 to 99
                buf.append(fb[i-cl:i+cl+1])                       
        buf = np.array(buf)
        buf = np.reshape(buf,(buf.shape[0],buf.shape[1]*buf.shape[2]))
        N,D = buf.shape

        #adding the label in the last column; 0 for class A, 1 for class E, 2 for class M, 3 for class N
        label_index = 2
        
        buf2 = label_index*np.ones((N,D+1))
        buf2[:,:-1] = buf 

        buf2 = np.matrix(buf2)
        print('The train data shape: ',buf2.shape)
        #print(buf2.shape)
        #train.append(buf2)

        buf2.dump('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/features/c3'+f+'.dat')

##############################
indir = '/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/4Shreya_cut/'  # For class 2
os.chdir(indir)
frame_size = 16e-3
frame_stride = 10e-3
print('Reading files for class 2')

for root, dirs, filenames in os.walk(indir):
    for f in sorted(filenames):
        path_file = f
        fb,MFCC = mfcc_features(path_file, frame_size, frame_stride)  # Get MFEC and MFCC using only MFEC
        #print(fb.shape)
        
        f = os.path.splitext(f)[0]  # gives first name of file without extension
        print(f)

        #saving the Mel filter bank features
        df = pd.DataFrame(fb)   # Saving using pandas
        #df.to_csv('/home/shikha/Desktop/FAU_AEC/features_melfb2164/set3/A/'+f+'.csv',header=None, index=None) # Avoid index numbers merging with actual MFEC values
        
        #Saving the features as context segments/ Gets the context by considering 10 frames before and 10 frames after the current frame
        
        k = fb.shape[0]
        buf=[]
        #cl = 1 #+/- frames to consider (Change to 3 or 5)
        for i in range(k):
            if (i>cl-1) & (i<k-cl):  # 9<i<90 / starts from 0 to 99
                buf.append(fb[i-cl:i+cl+1])                       
        buf = np.array(buf)
        buf = np.reshape(buf,(buf.shape[0],buf.shape[1]*buf.shape[2]))
        N,D = buf.shape

        #adding the label in the last column; 0 for class A, 1 for class E, 2 for class M, 3 for class N
        label_index = 3
        
        buf2 = label_index*np.ones((N,D+1))
        buf2[:,:-1] = buf 

        buf2 = np.matrix(buf2)
        print('The train data shape: ',buf2.shape)
        #print(buf2.shape)
        #train.append(buf2)

        buf2.dump('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/features/c4'+f+'.dat')

##################################
        
indir = '/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/5Yesu_cut/'  # For class 2
os.chdir(indir)
frame_size = 16e-3
frame_stride = 10e-3
print('Reading files for class 2')

for root, dirs, filenames in os.walk(indir):
    for f in sorted(filenames):
        path_file = f
        fb,MFCC = mfcc_features(path_file, frame_size, frame_stride)  # Get MFEC and MFCC using only MFEC
        #print(fb.shape)
        
        f = os.path.splitext(f)[0]  # gives first name of file without extension
        print(f)

        #saving the Mel filter bank features
        df = pd.DataFrame(fb)   # Saving using pandas
        #df.to_csv('/home/shikha/Desktop/FAU_AEC/features_melfb2164/set3/A/'+f+'.csv',header=None, index=None) # Avoid index numbers merging with actual MFEC values
        
        #Saving the features as context segments/ Gets the context by considering 10 frames before and 10 frames after the current frame
        
        k = fb.shape[0]
        buf=[]
        #cl = 1 #+/- frames to consider (Change to 3 or 5)
        for i in range(k):
            if (i>cl-1) & (i<k-cl):  # 9<i<90 / starts from 0 to 99
                buf.append(fb[i-cl:i+cl+1])                       
        buf = np.array(buf)
        buf = np.reshape(buf,(buf.shape[0],buf.shape[1]*buf.shape[2]))
        N,D = buf.shape

        #adding the label in the last column; 0 for class A, 1 for class E, 2 for class M, 3 for class N
        label_index = 4
        
        buf2 = label_index*np.ones((N,D+1))
        buf2[:,:-1] = buf 

        buf2 = np.matrix(buf2)
        print('The train data shape: ',buf2.shape)
        #print(buf2.shape)
        #train.append(buf2)

        buf2.dump('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/features/c5'+f+'.dat')
'''
