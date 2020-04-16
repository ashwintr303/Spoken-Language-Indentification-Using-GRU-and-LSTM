# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:15:07 2020

@author: AshwinTR
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, LSTM
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from sklearn.utils import class_weight
from sklearn import preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# Set true for debug mode
debug = True

# define sequence length
seq_len = 200

# import data
with h5py.File('train_data.hdf5','r') as hf:
    english_data = hf['english_data'][:]
    english_label = hf['english_label'][:]
    hindi_data = hf['hindi_data'][:]
    hindi_label = hf['hindi_label'][:]
    mandarin_data = hf['mandarin_data'][:]
    mandarin_label = hf['mandarin_label'][:]

# split train and validation data
xtrain_english, xval_english, ytrain_english, yval_english = train_test_split(english_data, english_label, test_size=0.3, shuffle=False) 
xtrain_hindi, xval_hindi, ytrain_hindi, yval_hindi = train_test_split(hindi_data, hindi_label, test_size=0.3, shuffle=False)
xtrain_mandarin, xval_mandarin, ytrain_mandarin, yval_mandarin = train_test_split(mandarin_data, mandarin_label, test_size=0.3, shuffle=False)

# concatenate data 
X_train =  np.concatenate((xtrain_english, xtrain_hindi, xtrain_mandarin), axis=0)
X_val = np.concatenate((xval_english, xval_hindi, xval_mandarin), axis=0)
y_train = np.concatenate((ytrain_english, ytrain_hindi, ytrain_mandarin), axis=0)
y_val = np.concatenate((yval_english, yval_hindi, yval_mandarin), axis=0)

#shuffle data
X_train, y_train = shuffle(X_train, y_train)
X_val, y_val = shuffle(X_val, y_val)

print('X_train: ', np.shape(X_train))
print('X_val: ', np.shape(X_val))
print('y_train: ', np.shape(y_train))
print('y_val: ',np.shape(y_val))

if debug:
    X_train = X_train[:int(0.1*len(X_train))]
    X_val = X_val[:int(0.1*len(X_val))]
    y_train = y_train[:int(0.1*len(y_train))]
    y_val = y_val[:int(0.1*len(y_val))]


# =============================================================================
# Sequence model
# =============================================================================
training_in = Input(batch_shape=(None,seq_len,64))
d1 = Dense(16, activation='relu')(training_in)
#d2 = Dense(128, activation='relu')(d1)
#b1 = BatchNormalization()(training_in)
block1 = GRU(16,  return_sequences=True, stateful=False)(training_in)
#bn = BatchNormalization()(block1)
#block2 = GRU(32, return_sequences=True, stateful=False)(block1)
#d2 = Dense(8, activation='relu')(block1)
training_pred = Dense(3, activation='softmax')(block1)
#output = Dropout(0.5)(training_pred)
training_model = Model(inputs=training_in, outputs=training_pred)

training_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
training_model.summary()
plot_model(training_model, to_file='training_model.png', show_shapes=True, show_layer_names=True)

#class_weight = class_weight.compute_class_weight('balanced', np.unique(np.ravel(y_train,order='C')),np.ravel(y_train,order='C')) 
results = training_model.fit(X_train, y_train, validation_data=(X_val, y_val),  batch_size=16, epochs=1, shuffle=True)
training_model.save('/home/ubuntu/hw5/training_model.hdf5')
training_model.save_weights('weights.hd5', overwrite=True)

loss = results.history['loss']
val_loss = results.history['val_loss']
accuracy = results.history['accuracy']
val_accuracy = results.history['val_accuracy']

epochs = np.arange(len(loss))
plt.figure()
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss for sequence model')
plt.legend()
plt.savefig('/home/ubuntu/hw5/sequence_model_loss.png', dpi=256)
plt.close()

plt.plot(epochs, accuracy, label='acc')
plt.plot(epochs, val_accuracy, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy for sequence model')
plt.legend()
plt.savefig('/home/ubuntu/hw5/sequence_model_acc.png', dpi=256)
plt.close()


# =============================================================================
# Streaming model
# =============================================================================

print('..................streaming model.................')
streaming_in = Input(batch_shape=(1,None,64)) 
block1 = GRU(16, return_sequences=False, stateful=True )(streaming_in)
streaming_pred = Dense(3,activation='softmax')(block1)
streaming_model = Model(inputs=streaming_in, outputs=streaming_pred)

streaming_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
streaming_model.summary()
plot_model(streaming_model, to_file='streaming_model.png', show_shapes=True, show_layer_names=True)

training_model.save_weights('weights.hd5', overwrite=True)
streaming_model.load_weights('weights.hd5')

for s in range(len(X_train)):
    in_seq = X_train[s].reshape( (1, seq_len, 64) )
    seq_pred = training_model.predict(in_seq)
    seq_pred = seq_pred.reshape(seq_len,3)
    for n in range(seq_len):
        in_feature_vector = X_train[s][n].reshape(1,1,64)
        single_pred = streaming_model.predict(in_feature_vector)[0]
        print(f'Seq-model Prediction, Streaming-Model Prediction: {seq_pred[n]}, {single_pred}')
    streaming_model.reset_states()



