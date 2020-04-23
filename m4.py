# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:33:59 2020

@author: Keisha
"""

# added batch normalization
# graphs learning curve

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import time

start = time.time()

batch_size = 128 # each iteration trains 128 images; 
# increasing batch size improves model up to a certain point
num_classes = 10 # 0 to 9 = 10 classes
epochs = 12 # iterations over the entire dataset

# input image dimensions for 28x28 input layer
img_rows, img_cols = 28, 28 

# the data array of 784 pixels, split between train and test sets
# x_ contains handwritten digits
# y_ contains the labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# before reshape
# x_train.shape = (60000, 28, 28)
# x_test.shape = (10000, 28, 28)
# y_train.shape = (60000,)
# y_test.shape = (10000,)

# .shape(rows,columns) returns the dimensions of the array
# x_train.shape returns 3x3 array: (60000, 28, 28)
# (n, width, height) to (n, depth, width, height)
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

# after reshape
# x_train.shape = (60000, 28, 28, 1)
# x_test.shape) = (10000, 28, 28, 1)

# contains greyscale RGB code (0-255)
# convert to float to get decimal points after division
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
# Normalizes the RGB codes by dividing to the max RGB value
x_train /= 255 
x_test /= 255
# pixel range within [0,1] instead of [0, 255]

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# contains labels from 0 to 9
# one hot encodes target values (outputs one 1, rest 0)
y_train = keras.utils.to_categorical(y_train, num_classes) # (60000, 10)
y_test = keras.utils.to_categorical(y_test, num_classes) # (10000, 10)
# y_train[0] = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.] for #5

model = Sequential() # creates linear stack of neural net layers 
model.add(Conv2D(32, kernel_size=(3, 3), # 32 filters, kernel = size of filter,
                 activation='relu', # rectified linear unit
                 # f(x) = max(0,x), sets neg vals to 0, constant input x 
                 # try sigmoid
                 input_shape=input_shape)) # input shape for first layer
                 # infers the shape for later layers
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu')) # 64 feature maps in this layer
model.add(MaxPooling2D(pool_size=(2, 2))) # downsamples the input to a 14x14 matrix
# chooses max value within kernel window (reduces parameters)
model.add(BatchNormalization()) 
# prevents the data out range of the activation function from vanishing
# end of feature extraction, begins classification
model.add(Dropout(0.25)) # randomly disables 25% of neurons (reduces overfitting)
model.add(Flatten()) # flattens the 2D arrays to connect the conv and dense layers
model.add(Dense(128, activation='relu')) # 128 neurons in hidden layer
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) # num_classes = 10 nodes in output layer
# assigns probabilities to outputs that sum up to 1 then picks the max

model.compile(loss=keras.losses.categorical_crossentropy, # for multi-classification (>2, binary = 2)
              optimizer=keras.optimizers.Adadelta(), # learning rate = 1, rho = 0.95
              # try optimizer = Adam, SGM, Nadam, Adamax, Adagrad, RMSprop 
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

# training the model and saving metrics in history
history = model.fit(x_train, y_train,
          batch_size, epochs,
          verbose=2,
          validation_data=(x_test, y_test))

# saving the model
save_dir = "/results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig