# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:33:59 2020

@author: Keisha
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import time

start = time.time()

batch_size = 128 # 128 images trained at a time
num_classes = 10 # 0 to 9 = 10 classes
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    input_shape = (img_rows, img_cols, 1) # 28x28x1 = 784

# contains greyscale RGB code (0-255)
# convert to gloat to get decimal points after division
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
# Normalizes the RGB codes by dividing to the max RGB value
x_train /= 255 
x_test /= 255
# makes each pixel within [0,1] instead of [0, 255]

# x_test.shape = (10000, 28, 28)
print('x_train shape:', x_train.shape) # (60000, 28, 28, 1)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# contains labels from 0 to 9
# one hot encodes target values (outputs one 1, rest 0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# y_train[0] = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.] for #5

model = Sequential()
ZeroPadding2D((2,2))
model.add(Conv2D(64, kernel_size=(5, 5), # 32 filters, kernel = size of filter,
                 activation='relu', # rectified linear unit
                 # f(x) = max(0,x), sets neg vals to 0, constant input x 
                 input_shape=input_shape)) # input shape for first layer
ZeroPadding2D((2,2))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
ZeroPadding2D((2,2))
model.add(Conv2D(256, (3, 3), activation='relu'))
ZeroPadding2D((1,1))
model.add(Conv2D(256, (5, 5), activation='relu',padding='same'))
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.2))
ZeroPadding2D((1,1))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(0.2))
ZeroPadding2D((1,1))
model.add(Conv2D(512, (5, 5), activation='relu',padding='same'))
model.add(MaxPooling2D(strides=(2,2))) # downsamples the input
#model.add(Dropout(0.4)) # randomly disables 25% of neurons (reduces overfitting)
model.add(Flatten()) # flattens the 2D arrays for fully connected layers
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu')) # batch size = 128 = hidden neuron units
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, # for multi-classification (>2, binary = 2)
              optimizer=keras.optimizers.Nadam(), # learning rate = 1, rho = 0.95
              metrics=['accuracy'])

model.fit(x_train, y_train, # trains the model
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Execution time:', time.time()-start, 'seconds.')