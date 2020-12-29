import os
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import librosa
import scipy.signal
from sklearn.preprocessing import minmax_scale


data_path_wav = '/home/arjun/Desktop/new_work/data_wav'
class_file = np.loadtxt('/home/arjun/Desktop/Untitled Folder/class_data.txt',dtype='str')
mat_train = '/home/arjun/Desktop/new_work/exp_2/mat_train'

#class_labels = class_file[:,1]
class_names = class_file[:,0]

cls_name = []

for i in range(len(class_names)): 
   cls_name.append(class_names[i]+".wav")


feature_file_train = []


for i in range(len(cls_name)):
   [fs, x] = wavfile.read(os.path.join(data_path_wav,cls_name[i]))
   D = librosa.stft(x,882*2,882,882*2,scipy.signal.hamming)
   D = np.abs(D)**2
   S = librosa.feature.melspectrogram(S=D,n_mels=40)
   S=librosa.power_to_db(S,ref=np.max)
   sio.savemat(os.path.join(mat_train,class_names[i]+".mat"), {'vect':S})
 

   
   '''

feature_file_train = np.array(feature_file_train)
feature_file_train = np.reshape(feature_file_train,(15690,40,500,1))
np.save('feature_file_new_net_test',feature_file_test)
#np.save('label_file_new_net',class_labels)
'''

