#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 10:06:18 2018

@author: muralikrishna
"""

import os
import numpy as np

indir = '/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/features/'
os.chdir(indir)
cl=3
D =((2*cl+1)*40)+1#841 #context length*original dimension + 1(labels). For eg. 40 filter banks with context segment length of 21 = 21*40+1=841
data1 = np.empty((0,D), dtype=float)
data2 = np.empty((0,D), dtype=float)
data3 = np.empty((0,D), dtype=float)
data4 = np.empty((0,D), dtype=float)

for root, dirs, filenames in os.walk(indir):
    print('Total songs in all class: ',len(filenames))
    c=0
    for f in sorted(filenames):
        #f = os.path.splitext(f)[0]
        if c%4==0:
            print('Reading B1 file: ',f)
            mat1 = np.load(f)
            dataa1 = mat1.astype(np.float32)
            data1 = np.append(data1,dataa1,axis=0)
        elif c%4==1:
            print('Reading B2 file: ',f)
            mat2 = np.load(f)
            dataa2 = mat2.astype(np.float32)
            data2 = np.append(data2,dataa2,axis=0)
        elif c%4==2:
            print('Reading B3 file: ',f)
            mat3 = np.load(f)
            dataa3 = mat3.astype(np.float32)
            data3 = np.append(data3,dataa3,axis=0)
        else:
            print('Reading B4 file: ',f)
            mat4 = np.load(f)
            dataa4 = mat4.astype(np.float32)
            data4 = np.append(data4,dataa4,axis=0)
        c+=1
            

print('Shape of combined data1: ',data1.shape)
data1.dump('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/trainingdata/MFECtrain1.dat')
print('Shape of combined data2: ',data2.shape)
data2.dump('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/trainingdata/MFECtrain2.dat')
print('Shape of combined data3: ',data3.shape)
data3.dump('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/trainingdata/MFECtrain3.dat')
print('Shape of combined data4: ',data4.shape)
data4.dump('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/trainingdata/MFECtrain4.dat')


