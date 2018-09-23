__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
# import numpy as np
from helpers import *

## NEURAL NETWORK MODEL ##
def nn_model(X,Y,n_h,num_iter=10000,print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    # Initialize parameters
    params = init_params(n_x,n_h,n_y)
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # Gradient descent (loop)
    for i in range(0,num_iter):
        # Forward propagation
        A2,cache = forward_prop(X,params)

        # Cost function
        cost = compute_cost(A2,Y,params)

        # Backward propagation
        grads = backward_prop(params,cache,X,Y)

        # Gradient descent parameter update
        params = update_params(params,grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return params

# X_assess,Y_assess = nn_model_test_case()
# params = nn_model(X_assess,Y_assess,4,num_iter=10000,print_cost=True)
# print("W1 = " + str(params["W1"]))
# print("b1 = " + str(params["b1"]))
# print("W2 = " + str(params["W2"]))
# print("b2 = " + str(params["b2"]))

## PREDICTION ##
def predict(params,X):
    A2,cache = forward_prop(X,params)
    predict = np.round(A2)

    return predict

# params,X_assess = predict_test_case()
# predict = predict(params,X_assess)
# print("predictions mean = " + str(np.mean(predict)))