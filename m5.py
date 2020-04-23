# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:33:59 2020

@author: Keisha
"""

# adjusted epochs

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import time

start = time.time()

batch_size = 128 # each iteration trains 128 images; 
# increasing batch size improves model up to a certain point
num_classes = 10 # 0 to 9 = 10 classes
epochs = 12 # iterations over the entire dataset
# e=12; l=0.02734212373; a=0.9915999770
# 0.05624; 0.98790; 3073.38; Nadam, dropout(0), dense(512) 

# without conv
# 12; 0.1676106702; 0.94929999113; 56.366
# 14; 0.1626789758; 0.95219999551; 61.045
# 20; 0.1516738290; 0.95389997959; 85.824
# 30; 0.1448284486; 0.95779997110; 127.24

# 0.16761; 0.94929; 56.366; Adadelta
# 0.15733; 0.95370; 53.3296; Adam
# 0.14550; 0.95770; 54.0562; Nadam***
# 0.18246; 0.94880; 50.7746; Adamax
# 0.23330; 0.93480; 53.5086; Adagrad
# 0.17050; 0.95050; 52.1452; RMSprop
# 0.38164; 0.90160; 50.5206; SGD

# 0.25152; 0.93860; 52.8348; 0.5
# 0.14550; 0.95770; 54.0562; Nadam*** 0.25
# 0.13299; 0.95980; 55.2888; 0.2
# 0.12116; 0.96480; 53.3669; 0.25/0.25
# 0.11144; 0.96650; 52.8965; 0.25/0.2
# 0.10975; 0.96600; 52.2433; 0.2/0.2
# 0.09020; 0.97259; 46.4787; 0/0

# 0.09020; 0.97259; 46.4787; Nadam, dropout(0) 
# 0.09532; 0.97119; 50.8982; 0/batch
# 0.10972; 0.97200; 62.3384; batch/0
# 0.10207; 0.96840; 67.1126; batch/batch

# 0.09020; 0.97259; 46.4787; Nadam, dropout(0), dense(128) 
# 0.08688; 0.97560; 63.5518; d(512)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# before reshape
# x_train.shape = (60000, 28, 28)
# y_train.shape = (60000,)
# x_test.shape = (10000, 28, 28)
# y_test.shape = (10000,)

# .shape(rows,columns) returns the dimensions of the array
# x_train.shape returns 3x3 array: (60000, 28, 28)
# reshape the array from a 3x3 array to a 4x4 array so that it can work with the Keras API
if K.image_data_format() == 'channels_first': # used in Theano
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols) # 1x28x28; 
    # color channels = 1 (grayscale)
else: # 'channels last' used in Tensorflow
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1) # 28x28x1 = 784 (length of input vector)

# contains greyscale RGB code (0-255)
# convert to float to get decimal points after division
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
# Normalizes the RGB codes by dividing to the max RGB value
x_train /= 255 
x_test /= 255
# makes each pixel within [0,1] instead of [0, 255]

# after reshape
# x_train, reshape = (60000, 28, 28, 1)
# y_train, reshape = (60000,)
# x_test, reshape = (10000, 28, 28, 1)
# y_test, reshape = (10000,)
print('x_train reshape:', x_train.shape) # (60000, 28, 28, 1)
print('x_test reshape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# contains labels from 0 to 9
# one hot encodes target values (outputs one 1, rest 0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y_train.shape after one-hot encoding: ", y_train.shape) # (60000, 10)
print("y_test.shape after one-hot encoding: ", y_test.shape)
# y_train[0] = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.] for #5

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), # 32 filters, kernel = size of filter,
                 activation='relu', # rectified linear unit
                 # f(x) = max(0,x), sets neg vals to 0, constant input x 
                 # try sigmoid
                 input_shape=input_shape)) # input shape for first layer
                 # infers the shape for later layers
# model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu')) # 64 filters in this layer
model.add(MaxPooling2D(pool_size=(2, 2))) # downsamples the input (reduces parameters)
# chooses max value within kernel window
# model.add(BatchNormalization()) 
# prevents the data out range of the activation function from vanishing
model.add(Dropout(0)) # randomly disables 25% of neurons (reduces overfitting)
model.add(Flatten()) # flattens the 2D arrays to connect the conv and dense layers
model.add(Dense(128, activation='relu')) # 128 = hidden neuron units
# model.add(BatchNormalization())
model.add(Dropout(0))
model.add(Dense(num_classes, activation='softmax')) # num_classes = 10 nodes in output layer
# assigns probabilities to outputs that sum up to 1 then picks the max

model.compile(loss=keras.losses.categorical_crossentropy, # for multi-classification (>2, binary = 2)
              optimizer=keras.optimizers.Nadam(), # learning rate = 1, rho = 0.95
              # try optimizer = Adam, SGD, Nadam, Adamax, Adagrad, RMSprop 
              metrics=['accuracy'])

model.fit(x_train, y_train, # trains the model
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, # prints detailed info in console (eta, loss, acc)
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Execution time:', time.time()-start, 'seconds.')