__author__ = 'Zubair Beg'

## IMPORT LIBRARIES ##
import numpy as np
import h5py

## DEFINITION ##
filename_train = 'train_catvnoncat.h5'
filename_test = 'test_catvnoncat.h5'
def load_dataset():
    ## TRAIN_DATASET ##
    train_dataset = h5py.File('datasets/'+filename_train,"r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])    # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])    # train set labels

    ## TEST_DATASET ##
    test_dataset = h5py.File('datasets/'+filename_test,"r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])       # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])       # test set labels
    classes = np.array(test_dataset["list_classes"][:])             # list of classes

    ## SHAPE ##
    train_set_y_orig = train_set_y_orig.reshape(1,train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(1,test_set_y_orig.shape[0])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes