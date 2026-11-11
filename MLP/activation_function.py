import numpy as np

def softmax(x : np.array):
    c = np.max(x, axis=1, keepdims=True)        
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True) 
    return exp_a / sum_exp_a

def Affine(x : np.array, w : np.array, b : np.array):
    y = x @ w + b
    return y

def ReLU(x : np.array):
    return np.maximum(0, x)

def cross_entropy_error(y : np.array, t : np.array):
    delta = 1e-7
    batch_size = y.shape[0] # legacy code, which should be deleted
    labels = np.argmax(t, axis=1)
    out = -np.mean(np.log(y[np.arange(batch_size), labels] + delta))
    return out

def mean_squared_error(y : np.array, t : np.array):
    E = np.mean((y - t)**2)
    return E