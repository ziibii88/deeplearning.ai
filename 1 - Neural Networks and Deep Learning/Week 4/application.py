__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
# import time
# import h5py
# import scipy
import numpy as np
import matplotlib as plt
# from PIL import Image
# from scipy import ndimage
# from dnn_app_utils_v2 import *
# from helpers import load_data
from models import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

## LOAD DATASET ##
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

## Example of a picture
# index = 11
# plt.imshow(train_x_orig[index])
# plt.show()
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

## Explore your dataset
# m_train = train_x_orig.shape[0]
# num_px = train_x_orig.shape[1]
# m_test = test_x_orig.shape[0]
# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))

## STANDARDIZE INPUT/IMAGES ##
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
# print ("train_x's shape: " + str(train_x.shape))
# print ("test_x's shape: " + str(test_x.shape))

### uncomment to run 2 layer model ###
## DEFINE CONSTANTS - 2 LAYER MODEL ##
# n_x = 12288     # num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)
#
## RUN THE MODEL - 2 LAYER MODEL ##
# params = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iter = 2500, print_cost=True)
#
## PREDICTION - 2 LAYER MODEL ##
# predictions_train = predict(train_x, train_y, params)
# predictions_test = predict(test_x, test_y, params)

## DEFINE CONSTANTS - L-LAYER MODEL ##
layer_dims = [12288, 20, 7, 5, 1] #  4-layer model

## RUN THE MODEL - L-LAYER MODEL ##
params = L_layer_model(train_x, train_y, layer_dims, num_iter = 2500, print_cost = True)

## PREDICTION - L-LAYER MODEL ##
predictions_train = predict(train_x, train_y, params)
predictions_test = predict(test_x, test_y, params)

## MISLABELED IMAGES ##
print_mislabeled_images(classes, test_x, test_y, predictions_test)

## CUSTOM IMAGE ##
# my_image = "my_image.jpg" # change this to the name of your image file
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
# my_image = my_image/255.
# my_predicted_image = predict(my_image, my_label_y, params)
# plt.imshow(image)
# plt.show()
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")