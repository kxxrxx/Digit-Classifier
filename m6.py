# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:33:59 2020

@author: Keisha
"""

# adding layers

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import time

start = time.time()

batch_size = 128 # each iteration trains 128 images; 
# increasing batch size improves model up to a certain point
num_classes = 10 # 0 to 9 = 10 classes
epochs = 3 # iterations over the entire dataset
# e=12; l=0.02734212373; a=0.9915999770; t=2337.30949
# 0.05624; 0.98790; 3073.38; Nadam, dropout(0), dense(512) 

# 0.0246; 0.9921; 447.597; 32,32,mp,64,64,mp,128** relu e=3
# 0.0455; 0.9880; 449.631; sigmoid
# 0.0650; 0.9832; 477.759; elu
# 0.0427; 0.9876; 657.042; selu
# ; tanh
# ; exponential
# ; hard_sigmoid
# ; linear

# 0.020928; 0.9927999973297; 395.823; 32,bn,32,a,mp,bn,64,a,bn,64,mp,f,d256,dr(0.2),d10** 
# 0.028527; 0.9907000064849; 428.282; 32,bn,32,a,mp,bn,64,a,bn,64,mp,f,d128,dr(0.2),d10**

# 0.020928; 0.9927999973297; 395.823; 32,bn,32,a,mp,bn,64,a,bn,64,mp,f,d256,dr(0.2),d10** 
# dr(0.25)
# 0.036771; 0.9897000193595; 424.448; dr(0)
# dr(0.1)
# dr(0.5)

# 32,bn,32,bn,32,bn,dr0.4, 64,bn,64,bn,64,dr0.4,128,bn,f,dr(0.4),d10,adam


# 0.0246; 0.9921; 447.597; 32,32,mp,64,64,mp,d128 k3 e=3
# 0.023290; 0.99269998; 469.530; bn, dr=0.2, d256
# 0.021265; 0.99279999; 439.535; bn sep
# 0.025545; 0.99250000; 417.433; bn after 2nd mp
# 0.034510; 0.98919999; 431.593; all ax=-1
# 0.026845; 0.99129998; 399.600; bn after 2nd mp, d(0), d128
# 0.024478; 0.99239999; 403.919; bn after 2nd mp, d(0), d256
# 0.028636; 0.99140000; 398.084; bn sep, d(0.2), d128
# 0.024287; 0.99220001; 406.113; 32,bn,32,bn,mp,64,bn,64,mp,d256,d(0.2),d10
# 0.056009; 0.98199999; 440.265; 32,bn,32,mp,bn,64,bn,64,mp,f,bn,d128,bn,dr(0.2),d10
# 0.026549; 0.99110001; 407.686; 32,bn,32,mp,bn,64,bn,64,mp,f,bn,d256,bn,dr(0.2),d10
#  ; 32,bn,32,mp,bn,64,bn,64,mp,f,bn,d512,bn,dr(0.2),d10
# 0.028539; 0.99150002; 402.974; 32,bn,32,mp,bn,64,bn,64,mp,f,d512,dr(0.2),d10
# 0.028563; 0.99040001; 418.228; d256, dr(0.2)
# 0.023058; 0.99279999; 400.474; d128, dr(0.2)
# 0.029581; 0.9911000; 399.324; 32,bn,32,mp,bn,64,64,mp,f,d128,dr(0.2)
# 0.024746; 0.9916999; 394.619; 32,bn,32,mp,bn,64,64,mp,f,d256,dr(0.2)
# 0.020928; 0.9927999973297; 395.823; 32,bn,32,a,mp,bn,64,a,bn,64,mp,f,d256,dr(0.2),d10**

# 0.0246; 0.9921; 447.597; 32,32,mp,64,64,mp,d128 k3, e=3
# 0.031384; 0.9901999; 444.260; dr(0.2)

# 0.0246; 0.9921; 447.597; 32,32,mp,64,64,mp,128 k3,3
# 0.0321; 0.9890; 321.350; k2
# 0.0269; 0.9919; 520.080; k4
# 0.0337; 0.9897; 583.318; k5

# 0.0246; 0.9921; 447.597; 32,32,mp,64,64,mp,d128 k3,3
# 0.0241; 0.9915; 444.718; d256
# 0.0301; 0.9901; 499.545; d512
#  d1024

# 0.0451; 0.9851; 451.119; Nadam, dropout(0) 32,64,mp,128; e=3
# 0.0308; 0.9901; 581.826; 32,32,mp,64,128
# 0.0296; 0.9912; 651.572; 32,32,64,mp,64,128
# 0.0390; 0.9896; 976.124; 32,32,64,64,mp,128
# 0.0246; 0.9921; 447.597; 32,32,mp,64,64,mp,128**
# 0.0301; 0.9901; 499.545; 32,32,mp,64,64,mp,512
# 0.0311; 0.9907; 609.308; 32,64,mp,64,64,128
# 64,64,mp,64,64,mp,128
# 0.0375; 0.9871; 368.908; 32,32,mp,32,32,mp,128
# 0.0278; 0.9922; 1159.915; 32,64,mp,128,256,mp,128
# 0.0290; 0.9897; 163.488; 32,mp,64,mp,128
# 0.0353; 0.9883; 251.427; 32,mp,64,mp,1024

# without conv
# 12; 0.1676106702; 0.94929999113; 56.366
# 14; 0.1626789758; 0.95219999551; 61.045
# 20; 0.1516738290; 0.95389997959; 85.824
# 30; 0.1448284486; 0.95779997110; 127.24

# 0.16761; 0.94929; 56.366; Adadelta; e=12
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
# print("y_train.shape after one-hot encoding: ", y_train.shape) # (60000, 10)
# print("y_test.shape after one-hot encoding: ", y_test.shape)
# y_train[0] = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.] for #5
# y_train.shape = (60000,10)
# y_test.shape = (10000,10)

# 32,bn,32,bn,32,bn,dr0.4, 64,bn,64,bn,64,dr0.4,128,bn,f,dr(0.4),d10,adam

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), # 32 filters, kernel = size of filter,
                 activation='relu', # rectified linear unit
                 # f(x) = max(0,x), sets neg vals to 0, constant input x 
                 # try sigmoid
                 input_shape=input_shape)) # input shape for first layer
                 # infers the shape for later layers
BatchNormalization(axis=-1)
#model.add(Conv2D(32, (3, 3), activation='relu')) #new 
model.add(Conv2D(32,(3, 3))) #new
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # new
BatchNormalization(axis=-1) #-1 for tf
#model.add(Conv2D(64, (3, 3), activation='relu')) #new 
model.add(Conv2D(64,(3, 3))) #new
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3), activation='relu')) # 64 filters in this layer
model.add(MaxPooling2D(pool_size=(2, 2))) # downsamples the input (reduces parameters)
# chooses max value within kernel window
#model.add(BatchNormalization(axis=-1)) 
# prevents the data out range of the activation function from vanishing
model.add(Dropout(0)) # randomly disables 25% of neurons (reduces overfitting)
model.add(Flatten()) # flattens the 2D arrays to connect the conv and dense layers
#model.add(BatchNormalization(axis=-1))
model.add(Dense(256, activation='relu')) # 128 = hidden neuron units
#model.add(BatchNormalization(axis=-1))
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