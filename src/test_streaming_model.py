# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:39:52 2020

@author: AshwinTR
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import librosa
import glob
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, LSTM
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers


#set file path here
path = str(sys.argv[1])

#set sequence length
seq_len = 200

    
'''
main function
'''
    
if __name__ == '__main__':
    
    streaming_in = Input(batch_shape=(1,None,64))
    block1 = GRU(16, return_sequences=False, stateful=True )(streaming_in)
    streaming_pred = Dense(3,activation='softmax')(block1)
    streaming_model = Model(inputs=streaming_in, outputs=streaming_pred)

    streaming_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(streaming_model, to_file='streaming_model.png', show_shapes=True, show_layer_names=True)
    
    streaming_model.load_weights('weights.hd5')

    #training_model = tf.keras.models.load_model('training_model.hdf5')

    X_test = []
    i = 0
    print('Reading files....')
    for file in glob.glob(path):
        data1, sr = librosa.load(file, sr = 16000)
        data1 = librosa.feature.mfcc(y = data1, sr = sr, n_mfcc = 64,
                                        n_fft = int(sr * 0.010),
                                        hop_length = int(sr*0.010)).T
        length = len(data1)
        data1 = data1[:length - (length % seq_len)]
        data1 = data1.reshape(int(len(data1)/seq_len), seq_len, 64)
        X_test.extend(data1)
        i += 1
    print('Number of files read: ',i)
    
    for s in range(len(X_test)):
        #in_seq = X_test[s].reshape( (1, seq_len, 64) )
        #seq_pred = training_model.predict(in_seq)
        #seq_pred = seq_pred.reshape(seq_len,3)
        for n in range(seq_len):
            in_feature_vector = X_test[s][n].reshape(1,1,64)
            single_pred = streaming_model.predict(in_feature_vector)[0]
            print(f'Streaming-Model Prediction: {single_pred}')
        streaming_model.reset_states()



   
