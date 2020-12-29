#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:53:12 2018
Program to cut the wave into 10ms intervals
"""

import os
import sox


for each_file in os.listdir("/home/muralikrishna/Desktop/DL_project/songstocut/"):
        if each_file.endswith('.wav'):
            print(each_file)
            new_name=each_file.replace('.wav', '')
            path=(os.path.join("/home/muralikrishna/Desktop/DL_project/songstocut/",each_file))
            
            if not os.path.exists('/home/muralikrishna/Desktop/DL_project/songstocut/cut/'+new_name):
                os.makedirs('/home/muralikrishna/Desktop/DL_project/songstocut/cut/'+new_name)
            #new_name=each_file.replace('.wav', '.mel')
            #mel_path='/media/harshita/BIRDS/assgn_data/glass_mel/'+new_name
            #np.savetxt(mel_path,mel,fmt='%10.5f')
            os.system("sox " +path+ " /home/muralikrishna/Desktop/DL_project/songstocut/cut/"+new_name+"/ouput.wav trim 0 0.5 : newfile : restart" )



