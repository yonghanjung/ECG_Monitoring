# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt

''' Function or Class '''


class RBF_NN:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        # Center by indim matrix
        self.centers = [random.uniform(-1,1,self.indim) for i in xrange(self.numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def RBF_Basis(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d) ** 2)

    def CalcAct(self, X):
        # Activation of RBF
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self.RBF_Basis(c, x)
        return G

    def Train(self, X,Y):
        # X : N by indim
        # Y : N by 1

        # Choose a random center vectors
        rdn_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rdn_idx]

        G = self.CalcAct(X)
        self.W = dot(pinv(G),Y)

    def Test(self, X):
        G = self.CalcAct(X)
        Y = dot(G, self.W)
        return Y








if __name__ == "__main__":
    print None