__author__ = 'Zubair Beg'
## RESOURSES ##
# https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset.ipynb
# https://github.com/andersy005/deep-learning-specialization-coursera/tree/master/01-Neural-Networks-and-Deep-Learning/week2/Programming-Assignments

## IMPORT LIBRARIES ##
# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import scipy
# import skimage  # for image testing only
# from skimage import transform # for image testing only
# from helpers import predict # for image testing only
from lr_utils import load_dataset
from model import model

## LOAD THE DATA ##
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

## EXAMPLE OF DATA ##
# index = 10                              # change value to for different pictures
# plt.imshow(train_set_x_orig[index])     # draws the picture in index
# plt.show()                              # shows the picture in index
# print("y = " + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " picture.")

## SHAPE DATASET ##
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

## RESHAPE DATASET ##
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

## STANDARDIZE DATASET ##
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# print(train_set_x)
# print(test_set_x)

## TRAIN DATASET ON MODEL ##
d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iter=2000,learn_rate=0.005,print_cost=True)

## ISSUES - wrongly classified picture ##
# index = 5
# plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
# print ("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[d["Y_predict_test"][0, index]].decode("utf-8") +  "\" picture.")

## PLOT LEARN CURVE ##
# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learn_rate"]))
# plt.show()

## FURTHER ANALYSIS ##
# learn_rates = [0.01, 0.001, 0.0001]
# models = {}
# for i in learn_rates:
#     print ("learning rate is: " + str(i))
#     models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iter = 1500, learn_rate = i, print_cost = False)
#     print ('\n' + "-------------------------------------------------------" + '\n')
# for i in learn_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learn_rate"]))
# plt.ylabel('cost')
# plt.xlabel('iterations')
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()

## TESTING IMAGE ##
# my_image = "my_image.jpg"
# fname = "images/" + my_image
# image = np.array(plt.imread(fname))
# my_image = skimage.transform.resize(image, output_shape=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
# my_predicted_image = predict(d["w"], d["b"], my_image)
# plt.imshow(image)
# plt.show()
# print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")