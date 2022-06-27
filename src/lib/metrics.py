import numpy as np

def mae(y, pred): 
    return np.mean(np.abs(y - pred))

def mse(y, pred):
    return np.mean(np.square(y - pred))

def rmse(y, pred):
    return np.sqrt(mse(y, pred))