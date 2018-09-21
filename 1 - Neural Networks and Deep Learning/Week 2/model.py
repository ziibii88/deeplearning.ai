__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
import numpy as np
from helpers import init_with_zeros,optimize,predict

def model(X_train,Y_train,X_test,Y_test,num_iter=2000,learn_rate=0.5,print_cost=False):
    ## PARAMETERS INIT ##
    w,b = init_with_zeros(X_train.shape[0])

    ## GRADIENT DESCENT ##
    params,grads,costs = optimize(w,b,X_train,Y_train,num_iter,learn_rate,print_cost)

    ## RETRIEVE PARAMS ##
    w = params["w"]
    b = params["b"]

    ## PREDICTION ##
    Y_predict_test = predict(w,b,X_test)
    Y_predict_train = predict(w,b,X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_predict_test": Y_predict_test,
         "Y_predict_train": Y_predict_train,
         "w": w,
         "b": b,
         "learn_rate": learn_rate,
         "num_iter": num_iter}

    return d