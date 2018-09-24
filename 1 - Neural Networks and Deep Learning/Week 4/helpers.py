__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
import h5py
import numpy as np
import matplotlib as plt
from testCases_v3 import *
from dnn_utils_v2 import *

## LOAD DATASET ##
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

## INITIALIZE PARAMETERS - 2 Layer NN ##
def init_params(n_x, n_h, n_y):
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return params

# params = init_params(3,2,1)
# print("W1 = " + str(params["W1"]))
# print("b1 = " + str(params["b1"]))
# print("W2 = " + str(params["W2"]))
# print("b2 = " + str(params["b2"]))

## INITIALIZE PARAMETERS - Multiple Layer Deep NN ##
def init_params_deep(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)         # number of layers in the network

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros(shape=(layer_dims[l], 1))

        assert(params['W' + str(l)].shape == (layer_dims[l], layer_dims[l -1]))
        assert(params['b' + str(l)].shape == (layer_dims[l], 1))

    return params

# params = init_params_deep([5,4,3])
# print("W1 = " + str(params["W1"]))
# print("b1 = " + str(params["b1"]))
# print("W2 = " + str(params["W2"]))
# print("b2 = " + str(params["b2"]))

## FORWARD PROPAGATION - LINEAR ##
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

# A, W, b = linear_forward_test_case()
# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))

## FORWARD PROPAGATION - ACTIVATION ##
def activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

# A_prev, W, b = linear_activation_forward_test_case()
# A, cache = activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))
# A, cache = activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))

## FORWARD PROPAGATION - L-MODEL ##
def L_model_forward(X, params):
    caches = []
    A = X
    L = len(params) // 2            # number of layers in the neural network

    # Implement Linear -> ReLU
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, params['W' + str(l)], params['b' + str(l)], activation = 'relu')
        caches.append(cache)

    # Implement Linear -> Sigmoid
    AL, cache = activation_forward(A, params['W' + str(L)], params['b' + str(L)], activation = 'sigmoid')
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

# X, params = L_model_forward_test_case_2hidden()
# AL, caches = L_model_forward(X, params)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))

## COST FUNCTION ##
def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect
                            # (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost

# Y, AL = compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))

## BACKWARD PROPAGATION - LINEAR ##
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

# dZ, linear_cache = linear_backward_test_case()
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

## BACKWARD PROPAGATION - ACTIVATION ##
def activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# dAL, activation_cache = linear_activation_backward_test_case()
# dA_prev, dW, db = activation_backward(dAL, activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")
# dA_prev, dW, db = activation_backward(dAL, activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

## BACKWARD PROPAGATION - L-MODEL ##
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)             # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)     # Y reshaped to the same shape as AL

    # Initialize backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Sigmoid -> Linear gradients
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, activation='sigmoid')

    # Loop from L-2 to L
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 1)], current_cache, activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print_grads(grads)

## UPDATE PARAMETERS ##
def update_params(params, grads, learn_rate):
    L = len(params) // 2                # number of layers in the network

    # Update rule
    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - learn_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learn_rate * grads["db" + str(l + 1)]

    return params

# params, grads = update_parameters_test_case()
# params = update_params(params, grads, 0.1)
# print ("W1 = "+ str(params["W1"]))
# print ("b1 = "+ str(params["b1"]))
# print ("W2 = "+ str(params["W2"]))
# print ("b2 = "+ str(params["b2"]))

## PREDICTIONS ##
def predict(X, y, params):
    m = X.shape[1]
    n = len(params) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, params)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p

## MISLABELED IMAGES ##
def print_mislabeled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))