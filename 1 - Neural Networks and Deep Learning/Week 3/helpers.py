__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
# import numpy as np
from testCases_v2 import *
from planar_utils import sigmoid

## LAYER SIZES ##
def layer_sizes(X,Y):
    n_x = X.shape[0]    # size of input layer
    n_h = 4             # size of hidden layer
    n_y = Y.shape[0]    # size of output layer

    return (n_x,n_h,n_y)

# X_assess,Y_assess = layer_sizes_test_case()
# (n_x,n_h,n_y) = layer_sizes(X_assess,Y_assess)
# print("The size of the input layer is: n_x = " + str(n_x))
# print("The size of the hidden layer is: n_h = " + str(n_h))
# print("The size of the output layer is: n_y = " + str(n_y))

## INITIALIZE PARAMETERS ##
def init_params(n_x,n_h,n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros(shape=(n_y,1))

    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return params

# n_x,n_h,n_y = initialize_parameters_test_case()
# params = init_params(n_x,n_h,n_y)
# print("W1 = " + str(params["W1"]))
# print("b1 = " + str(params["b1"]))
# print("W2 = " + str(params["W2"]))
# print("b2 = " + str(params["b2"]))

## FORWARD PROPAGATION ##
def forward_prop(X,params):
    # Retrieve parameters
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # Implement forward propagation
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1,X.shape[1]))

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2,cache

# X_assess,params = forward_propagation_test_case()
# A2,cache = forward_prop(X_assess,params)
## Note: we use the mean here just to make sure that your output matches ours.
# print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

## COMPUTE COST ##
def compute_cost(A2,Y,params):
    m = Y.shape[1]              # number of examples

    # Compute cross-entropy cost
    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1-Y),np.log(1-A2))
    cost = np.sum(logprobs)/m
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
                                # E.g., turns [[17]] into 17
    assert(isinstance(cost,float))

    return cost

# A2,Y_assess,params = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2,Y_assess,params)))

## BACKWARD PROPAGATION ##
def backward_prop(params,cache,X,Y):
    m = X.shape[1]              # number of examples

    # Retrieve parameters
    W1 = params['W1']
    W2 = params['W2']

    # Retrieve A1 and A2
    A1 = cache['A1']
    A2 = cache['A2']

    # Backward propagation
    dZ2 = A2-Y
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1 = (1/m) * np.dot(dZ1,X.T)
    db1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)

    grads = {"dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}

    return grads

# params,cache,X_assess,Y_assess = backward_propagation_test_case()
# grads = backward_prop(params,cache,X_assess,Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

## UPDATE PARAMETERS ##
def update_params(params,grads,learn_rate=1.2):
    # Retrieve parameters
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # Retrieve gradients
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule
    W1 = W1 - learn_rate * dW1
    b1 = b1 - learn_rate * db1
    W2 = W2 - learn_rate * dW2
    b2 = b2 - learn_rate * db2

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return params

# params,grads = update_parameters_test_case()
# params = update_params(params,grads)
# print("W1 = " + str(params["W1"]))
# print("b1 = " + str(params["b1"]))
# print("W2 = " + str(params["W2"]))
# print("b2 = " + str(params["b2"]))