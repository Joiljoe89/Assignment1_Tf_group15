#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:27:59 2018

@author: muralikrishna
"""


import os
import numpy as np

indir = '/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/features/'
os.chdir(indir)
cl=3
D =((2*cl+1)*40)+1#841 #context length*original dimension + 1(labels). For eg. 40 filter banks with context segment length of 21 = 21*40+1=841
data = np.empty((0,D), dtype=float)

for root, dirs, filenames in os.walk(indir):
    print('Total songs in all class: ',len(filenames))
    for f in sorted(filenames):
        #f = os.path.splitext(f)[0]
        print('Reading file: ',f)
        mat1 = np.load(f)
        dataa = mat1.astype(np.float32)
        data = np.append(data,dataa,axis=0)

print('Shape of combined data: ',data.shape)
data.dump('/home/muralikrishna/Desktop/DL_project/Songs_data/singer_identification/MFECtrain.dat')
