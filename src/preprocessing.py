# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:55:34 2020

@author: AshwinTR
"""

# -*- coding: utf-8 -*-
import numpy as np
import glob, os
import librosa
import h5py 
from sklearn import preprocessing

# define sequence length
seq_len = 200

#### extract mfcc features and write into a hdf5 file
def create_hdf5(file_list_1, file_list_2, file_list_3):
    train_data_english = []
    train_label_english = []
    train_data_hindi = []
    train_label_hindi = []
    train_data_mandarin = []
    train_label_mandarin = []
    
#### english files
    for file1 in file_list_1:
        data1, sr = librosa.load(file1, sr = 16000)
        data1 = librosa.feature.mfcc(y = data1, sr = sr, n_mfcc = 64,
                                n_fft = int(sr * 0.025), 
                                hop_length = int(sr*0.010)).T
        length = len(data1)
        data1 = data1[:length - (length % seq_len)]
        #data1 = np.array(preprocessing.normalize(data1))
        data1 = data1.reshape(int(len(data1)/seq_len), seq_len, 64)
        train_data_english.extend(data1)

    train_label_english = [[[1,0,0] for i in range(seq_len)] for j in range(len(train_data_english))]
    #train_label_english = np.zeros((len(train_data_english),seq_len,1))

#### hindi files
    for file2 in file_list_2:    
        data2, sr = librosa.load(file2, sr = 16000)
        data2 = librosa.feature.mfcc(y = data2, sr = sr, n_mfcc = 64,
                                n_fft = int(sr * 0.0025), 
                                hop_length = int(sr*0.010)).T
        length = len(data2)
        data2 = data2[:length - (length % seq_len)]
        #data2 = np.array(preprocessing.normalize(data2))
        data2 = data2.reshape(int(len(data2)/seq_len), seq_len, 64)
        train_data_hindi.extend(data2)

    train_label_hindi = [[[0,1,0] for i in range(seq_len)] for j in range(len(train_data_hindi))]
    #train_label_hindi = np.ones((len(train_data_hindi),seq_len,1))

#### mandarin files
    for file3 in file_list_3:
        data3, sr = librosa.load(file3, sr = 16000)
        data3 = librosa.feature.mfcc(y = data3, sr = sr, n_mfcc = 64,
                                n_fft = int(sr * 0.0025), 
                                hop_length = int(sr*0.010)).T
        length = len(data3)
        data3 = data3[:length - (length % seq_len)]
        #data3 = np.array(preprocessing.normalize(data3))
        data3 = data3.reshape(int(len(data3)/seq_len), seq_len, 64)
        train_data_mandarin.extend(data3)

    train_label_mandarin = [[[0,0,1] for i in range(seq_len)] for j in range(len(train_data_mandarin))]
    #train_label_mandarin = np.ones((len(train_data_mandarin),seq_len,1))*2 

    print('data1',np.shape(train_data_english))
    print('label1',np.shape(train_label_english))
    print('data2',np.shape(train_data_hindi))
    print('label2',np.shape(train_label_hindi))
    print('data3',np.shape(train_data_mandarin))
    print('label3',np.shape(train_label_mandarin))
    
    
    with h5py.File('train_data.hdf5', "w") as hf:
        hf.create_dataset('english_data', data=train_data_english)
        hf.create_dataset('english_label', data=train_label_english)   
    hf.close()
    
    with h5py.File('train_data.hdf5', "a") as hf:
        hf.create_dataset('hindi_data', data=train_data_hindi)
        hf.create_dataset('hindi_label', data=train_label_hindi) 
    hf.close()
        
    with h5py.File('train_data.hdf5', "a") as hf:
        hf.create_dataset('mandarin_data', data=train_data_mandarin)
        hf.create_dataset('mandarin_label', data=train_label_mandarin)   
    hf.close()

    
'''
main function
'''
    
if __name__ == '__main__':
    path1 = '/home/ubuntu/hw5/train/train_english/*'
    path2 = '/home/ubuntu/hw5/train/train_hindi/*'
    path3 = '/home/ubuntu/hw5/train/train_mandarin/*'
    file_list_1 = []
    file_list_2 = []
    file_list_3 = []
    for (file1,file2,file3) in zip(glob.glob(path1),glob.glob(path2),glob.glob(path3)):
        file_list_1.append(file1)
        file_list_2.append(file2)
        file_list_3.append(file3)
    print('english_files: ',len(file_list_1))
    print('hindi_files: ',len(file_list_2))
    print('mandarin_files: ',len(file_list_3))
    print(file_list_1)
    create_hdf5(file_list_1, file_list_2, file_list_3)
    
