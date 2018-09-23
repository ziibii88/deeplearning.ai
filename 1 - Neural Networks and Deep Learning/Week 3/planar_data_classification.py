__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
# import numpy as np
# import matplotlib.pyplot as plt
from nn_model import nn_model,predict
from testCases_v2 import *
# import sklearn
# import sklearn.datasets
# import sklearn.linear_model
from planar_utils import plot_decision_boundary, load_planar_dataset

np.random.seed(1) # set a seed so that the results are consistent

X, Y = load_planar_dataset() # load dataset

# Visualize the data:
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size

# print ('The shape of X is: ' + str(shape_X))
# print ('The shape of Y is: ' + str(shape_Y))
# print ('I have m = %d training examples!' % (m))

## SIMPLE LOGISTIC REGRESSION ##
# Train the logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV();
# clf.fit(X.T, Y.T);

# Plot the decision boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")

# Print accuracy
# LR_predictions = clf.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#       '% ' + "(percentage of correctly labelled datapoints)")

## RUN NN_MODEL ##
params = nn_model(X,Y,n_h=4,num_iter=10000,print_cost=True)

# Plot the decision boundary
# plot_decision_boundary(lambda x: predict(params, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))

# Print accuracy
predictions = predict(params, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

## TUNING HIDDEN LAYER SIZE ##
# plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    # plt.subplot(5, 2, i+1)
    # plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iter = 10000)
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))