
import numpy as np
from scipy import optimize


class KernelSVC:

    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel(X, X)
        ones = np.ones((N,1))
        zeros = np.zeros((N,1))
        I = np.eye(N)

        # Lagrange dual problem
        #'''--------------dual loss ------------------ '''
        def loss(alpha):
            return  0.5* alpha.T @ np.diag(y) @ K @ np.diag(y) @ alpha - np.sum(alpha)

        # Partial derivate of Ld on alpha
        # '''----------------partial derivative of the dual loss wrt alpha -----------------'''
        def grad_loss(alpha):
            return np.diag(y) @ K @ np.diag(y) @ alpha - np.ones(N)

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0
         # '''----------------function defining the equality constraint------------------'''
        fun_eq = lambda alpha: (0 - y.T @ alpha).reshape(1,1)
         #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        jac_eq = lambda alpha:  - y
         # '''---------------function defining the ineequality constraint-------------------'''
        fun_ineq = lambda alpha: self.C*np.vstack((ones,zeros)) - (np.vstack((I,-I))@alpha).reshape(2*N,1)
        # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        jac_ineq = lambda alpha:   - np.vstack((I,-I))

        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq',
                        'fun': fun_ineq ,
                        'jac': jac_ineq})
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes
        self.sv_index = (self.alpha > self.epsilon)
         #'''---------------- A matrix with each row corresponding to a point that falls on the margin ---------'''
        self.support = X[self.sv_index]
        alpha_sv = self.alpha[self.sv_index]
        y_diag_sv = np.diag(y[self.sv_index])
         #''' -----------------offset of the classifier------------------ '''
        self.b = (y[self.sv_index] - alpha_sv.T @ y_diag_sv @ self.kernel(self.support, self.support)).mean()
         # '''------------------------RKHS norm of the function f ------------------------------'''
        self.norm_f = alpha_sv.T @ y_diag_sv @ self.kernel(self.support, self.support) @ y_diag_sv @ alpha_sv

        # Will serve to define "separating_function"
        self.part_f = alpha_sv.T @ y_diag_sv

    ### Implementation of the separting function $f$
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.part_f @ self.kernel(self.support, x)


    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1

    
class Multi_Class_SVM_Classifier_OvA(object):

    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.classifiers = []

    def fit(self, X_train, y_train):

        self.nclasses = np.unique(y_train).size
        labels = np.unique(y_train)

        
        for i in range(self.nclasses):

            svm = KernelSVC(C = self.C, kernel = self.kernel)
            y_tr = np.where(y_train == labels[i], 1, -1)
            svm.fit(X_train, y_tr)
            self.classifiers.append(svm)

    def predict(self, X_test):
        predicts = np.zeros((X_test.shape[0], self.nclasses))

        for count, classifier in enumerate(self.classifiers):

            predicts[:,count] = classifier.separating_function(X_test) + classifier.b

        return np.argmax(predicts, axis = 1)
    

class Multi_Class_SVM_Classifier_OvO(object):

    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.classifiers = []

    def fit(self, X_train, y_train):

        self.nclasses = np.unique(y_train).size
        labels = np.unique(y_train)

        
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):

                svm = KernelSVC(C = self.C, kernel = self.kernel)

                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)

                svm.fit(X_train[indexes], y_tr)
                self.classifiers.append([svm,labels[i],labels[j]])
    def predict(self, X_test):
        predicts = np.zeros((X_test.shape[0], self.nclasses))

        for [classifier,label1, label2] in self.classifiers:

            pred = classifier.predict(X_test)
            predicts[np.where(pred == 1),label1] +=1
            predicts[np.where(pred == -1),label2] +=1

        return np.argmax(predicts, axis = 1)