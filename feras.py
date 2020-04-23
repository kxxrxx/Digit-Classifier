import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt




#load that data from a keras mnist data set module
mnist = tf.keras.datasets.mnist
(train_img, train_ans), (test_img, test_ans) = mnist.load_data()

#class identifiers to identify labels
class_digits = ['Zero', 'One', 'Two', 'Three', 'Four',
               'Five', 'Six', 'Seven', 'Eight', 'Nine']



print('')
print('There are a total of' ,len(train_img), 'images in this training set')

print('Along with the images each image contains an answer label denoting what number it is')

print('each image is:', len(train_img[0]), 'x', len(train_img[0]), 'in pixel size')



print('For example the test image in array number 0 is a 5, with the image being shown below')
print('')  
print("This picture shown represents a:", train_ans[0])
plt.figure()
plt.imshow(train_img[0], cmap = 'gray')
plt.show()  
#turn the tensor into a gray image each number in this tensor
#represents the grayscale from 0 to 255 thus the higher the number in the tensor the higher the brightness



#a print statement to ensure that the data manipulation is correct
#print(dataformating)
#print()
#print(dataformating.reshape(28,28))


print('')



print('below are the first 25 images of the mnist data set and their labels, that we can get a vizualization for')
print('')

train_img = train_img / 255.0    #type cast into a float and turn the range of data from 0-255 to 0-1

test_img = test_img / 255.0    #type cast into a float and turn the range of data from 0-255 to 0-1


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i], cmap=plt.cm.binary)
    plt.xlabel(class_digits[train_ans[i]])
plt.show()


My_Model = tf.keras.Sequential()


My_Model.add(keras.layers.Flatten(input_shape = (28,28)))
My_Model.add(keras.layers.Dense(128, activation = 'selu'))
My_Model.add(keras.layers.Dense(len(class_digits), activation = 'softmax'))




#flatten turns any tensor into a 1 row array from processing, so the input shape is 28x28 pixels which 
#coresponds to 784 into one flattened dimension
#first layer is 128 nuerons using an activation function that is a scaled exponential linear unit
#second layer is output with 10 nuerons out representing numbers and a softmax activation function to extrapolate
#the results further


My_Model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


My_Model.fit(train_img, train_ans, epochs=12)


test_loss, test_acc = My_Model.evaluate(test_img,  test_ans, verbose=2)



print()

print('\nTest accuracy:', test_acc)



Testing = tf.keras.Sequential([My_Model])

test = Testing.predict(test_img)

print()

print('Here are the ten values from the output neurons for a specific test case, this test image is 100 in the test array')
print(test[100])

print()

print('we can see that array index', np.argmax(test[100]), 'has the highest activation')
print('')
print('this value represents digit value:', class_digits[np.argmax(test[100])])
print('')



def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_digits[predicted_label],
                                100*np.max(predictions_array),
                                class_digits[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
  
i = 100
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, test[i], test_ans, test_img)
plt.subplot(1,2,2)
plot_value_array(i, test[i],  test_ans)
plt.show()


print('')
print('here are the activation functions again')
print(test[i])




num_rows = 25
num_cols = 1
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, test[i], test_ans, test_img)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, test[i], test_ans)
plt.tight_layout()
plt.show()
