# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 03:25:54 2018

@author: harshita
"""
import os


def get_mfcc(path):
    y, sr = librosa.load(path, sr=44100)
    mfccs=librosa.feature.melspectrogram(y=y,sr=sr)
    #mfccs = librosa.feature.mfcc(y=y, sr=sr,n_fft=1024, hop_length = 441,n_mfcc=13,n_mels=26)
    #np.set_printoptions(suppress=True)
    #print(mfcc1.shape)
    mfccs = mel.transpose()
    #feats1 = extract_delta_features(mfcc1)
    #Tfeats1 = feats1.transpose()
    #feats2 = extract_delta_features(feats1)
    #initial = np.concatenate((mfcc1,feats1), axis = 1)
    #final_39_dim = np.concatenate((initial,feats2),axis = 1)
#final_39_dim = final_39_dim.transpose()
    return mel











for each_file in os.listdir("/media/harshita/BIRDS/assgn_data/to_cut/gun/"):
        if each_file.endswith('.wav'):
            print(each_file)
            new_name=each_file.replace('.wav', '')
            path=(os.path.join("/media/harshita/BIRDS/assgn_data/to_cut/gun/",each_file))
            
            if not os.path.exists('/media/harshita/BIRDS/assgn_data/to_cut/gun/cut/'+new_name):
                os.makedirs('/media/harshita/BIRDS/assgn_data/to_cut/gun/cut/'+new_name)
            #new_name=each_file.replace('.wav', '.mel')
            #mel_path='/media/harshita/BIRDS/assgn_data/glass_mel/'+new_name
            #np.savetxt(mel_path,mel,fmt='%10.5f')
            os.system("sox " +path+ " /media/harshita/BIRDS/assgn_data/to_cut/gun/cut/"+new_name+"/ouput.wav trim 0 0.5 : newfile : restart" )
            
            


