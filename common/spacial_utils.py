import numpy as np
def noramlization(data):
    X = np.array(data)
    #归一化
    X_nor = (X - X.min()) / (X.max() - X.min())
    return X_nor

def obs_noramlization(data):
    X = np.array(data)
    D = np.array([[80,120,19,19],
                  [100,100,19,19],
                  [120,80,19,19]])
    #归一化
    X_nor = np.divide(X, D)
    return X_nor

def state_noramlization(data):
    X = np.array(data)
    D = np.array([80,120,100,100,120,80,19,19])
    #归一化
    X_nor = np.divide(X, D)
    return X_nor

