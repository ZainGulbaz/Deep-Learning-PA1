import numpy as np
EPSILON= 1e-8;

def relu(X):
    return np.maximum(0, X)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True)); 
    return exp_X / np.sum(exp_X, axis=1, keepdims=True);


def binary_cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    
    return -np.sum(y_true * np.log(y_pred +EPSILON) + (1 - y_true) * np.log(1 - y_pred +EPSILON)) / m


def categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred +EPSILON)) / m
