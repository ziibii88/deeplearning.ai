__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
import numpy as np

## SIGMOID FUNC ##
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(9.2) = " + str(sigmoid(9.2)))

## INITIALIZATION WITH ZEROS ##
def init_with_zeros(dim):
    w = np.zeros(shape=(dim,1))
    b = 0

    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w,b

# dim = 2
# w, b = init_with_zeros(dim)
# print ("w = " + str(w))
# print ("b = " + str(b))

## PROPAGATION ##
def propagate(w,b,X,Y):
    m = X.shape[1]

    ## FORWARD PROP - X TO COST ##
    A = sigmoid(np.dot(w.T,X) + b)                          # compute activation
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))   # compute cost

    ## BACKWARD PROP - TO FIND GRAD ##
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw, "db": db}
    return grads,cost

# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))

## OPTIMIZE ##
def optimize(w,b,X,Y,num_iter,learn_rate,print_cost = False):
    costs = []
    for i in range(num_iter):
        ## COST & GRAD CALC ##
        grads, cost = propagate(w,b,X,Y)

        ## GRADS DERIVATIVES ##
        dw = grads["dw"]
        db = grads["db"]

        ## UPDATE RULE ##
        w = w-learn_rate*dw
        b = b-learn_rate*db

        ## RECORD COSTS ##
        if i % 100 == 0:
            costs.append(cost)

        ## PRINT COST EVERY 100 TRAIN ##
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i,cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params,grads,costs

# params,grads,costs = optimize(w,b,X,Y,num_iter=100,learn_rate=0.009,print_cost=False)
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))

## PREDICTION ##
def predict(w,b,X):
    m = X.shape[1]
    Y_predict = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    ## COMPUTE VECTOR ##
    A = sigmoid(np.dot(w.T,X)+b)

    ## PROBABILITIES ##
    for i in range(A.shape[1]):
        Y_predict[0,i]=1 if A[0,i]>0.5 else 0

    assert(Y_predict.shape==(1,m))

    return Y_predict

# print("predictions = " + str(predict(w, b, X)))