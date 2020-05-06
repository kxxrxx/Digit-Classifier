This project utilizes the Tensorflow framework and Keras library to develop a convolutional neural network (CNN) able to recognize handwritten digits recorded in the MNIST dataset. The MNIST dataset comprises 60,000 training samples and 10,000 test samples of handwritten digits, 0-9. The goal of this project is to maximize accuracy while minimizing loss and runtime. 

A convolutional neural network (CNN) is designed and implemented to classify digits by applying filters to formatted 28x28 pixel monochrome images to extract and learn higher-level features. It is composed of a series of convolutional layers, pooling layers, and dense layers. The CNN is first trained on the 60000-image training dataset and then tested using the 10000-image testing set for validation. 

The best result is a testing accuracy of 99.59%, testing loss of 0.019198, and a final runtime of 252 seconds in the 12th epoch for a total execution time of 3059.848 seconds.

Update:

- Made a GUI that utilizes a model of the CNN to predict a digit drawn, uploaded, or captured on a camera by the user.

![drawing example](https://github.com/kxxrxx/Digit-Classifier/blob/master/drawing_ex.PNG)
![image example](https://github.com/kxxrxx/Digit-Classifier/blob/master/image_ex.PNG)

Packages used:

- Tkinter: standard Python interface for creating GUI applications
- OpenCV: image and video processing library used to reformat images and recognize objects through a camera
- PIL: Python Imaging Library for image processing

Resources:

https://docs.python.org/3.0/library/tkinter.html#packer-options
https://effbot.org/tkinterbook/tkinter-index.htm
https://pythonprogramming.net/loading-images-python-opencv-tutorial/
https://pillow.readthedocs.io/en/3.0.x/handbook/tutorial.html
