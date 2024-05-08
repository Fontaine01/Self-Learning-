import numpy as np

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self, X, Y):
        XX = np.sum(X**2, axis=1)[:, np.newaxis]
        YY = np.sum(Y**2, axis=1)[np.newaxis, :]
        distances = XX + YY - 2 * X.dot(Y.T)
        return np.exp(-distances / (2 * self.sigma**2))
    
class Linear:
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return  X@Y.T
    
class Polynomial:
    def __init__(self, d = 100, cst = 0):
        self.d = d  
        self.cst = cst 
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return np.power(X @ Y.T + self.cst,self.d)