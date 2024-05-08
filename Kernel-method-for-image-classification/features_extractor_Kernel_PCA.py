import numpy as np

class KernelPCA:

    def __init__(self,kernel, r=2):
        self.kernel = kernel          # <---
        self.alpha = None # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.lmbda = None # Vector of size d representing the top d eingenvalues
        self.support = None # Data points where the features are evaluated
        self.r =r ## Number of principal components
    def compute_PCA(self, X):
        # assigns the vectors
        self.support = X
        self.N = X.shape[0]
        K = self.kernel(self.support, self.support)
        I = np.eye(self.N)
        U =  (1/self.N) * np.ones((self.N, self.N))
        G = ((I - U) @ K @ (I - U))/self.N

        eigenvalue, eigenvectors = np.linalg.eigh(G)

        self.lmbda = eigenvalue[::-1][:self.r]
        self.alpha = eigenvectors[:, ::-1][:, :self.r]


    def transform(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size r
        K_ = self.kernel(x, self.support)
        N_trans = x.shape[0]
        ones_mtx_trans = np.ones((N_trans,self.N))/self.N
        K_trans = K_ - np.dot(ones_mtx_trans,K_)-np.dot(K_,self.ones) + ones_mtx_trans.dot(self.K).dot(self.ones)
        self.alpha = self.alpha/np.sqrt(self.lmbda * self.N)
        return np.dot(K_trans, self.alpha)
        