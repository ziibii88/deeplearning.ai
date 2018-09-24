__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
import numpy as np
import matplotlib as plt
from dnn_app_utils_v2 import *
# from helpers import *

## 2 LAYER MODEL ##
def two_layer_model(X, Y, layer_dims, learn_rate = 0.0075, num_iter = 3000, print_cost = False):
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims

    # Initialize parameters
    params = initialize_parameters(n_x, n_h, n_y)

    # Retrieve W1, b1, W2, b2
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # Gradient descent loop
    for i in range(0,num_iter):
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')

        # Compute cost
        cost = compute_cost(A2, Y)

        # Initialize backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        # Set grads
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        params = update_parameters(params, grads, learn_rate)

        # Retrieve W1, b1, W2, b2 after update
        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        b2 = params["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    ## plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learn_rate))
    # plt.show()

    return params

## L-LAYER MODEL ##
def L_layer_model(X, Y, layer_dims, learn_rate = 0.0075, num_iter = 3000, print_cost = False):
    np.random.seed(1)
    costs = []              # keep track of cost

    # Initialize parameters
    params = initialize_parameters_deep(layer_dims)

    # Gradient descent loop
    for i in range(0, num_iter):
        # Forward propagation
        AL, caches = L_model_forward(X, params)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        params = update_parameters(params, grads, learn_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    ## plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return params