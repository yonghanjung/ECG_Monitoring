# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNet
''' Function or Class '''


class SparseLDA:
    def __init__(self, Dict_TrainingData, Flt_Lambda, Flt_L1):
        # Only for two class
        # Dict_Trainingdata
            # Key : 0,1
            # Row : data
        self.Data1 = Dict_TrainingData[0] # N by 256 matrix
        self.Data2 = Dict_TrainingData[1] # V by 256 matrix
        self.Dim = len(self.Data1[0]) # 256

        self.X = np.concatenate((self.Data1, self.Data2), axis=0) # N / V augmented matrix
        self.X = self.X - np.mean(self.X,axis=0)

        self.NumClass1 = len(self.Data1) # N
        self.NumClass2 = len(self.Data2) # V
        self.TotalNum = self.NumClass1 + self.NumClass2

        self.Y = self.Construct_Y()
        self.D = np.dot(np.transpose(self.Y), self.Y) / float(self.TotalNum) # P
        self.Q = np.ones((2,1))

        InitialTheta = np.array([2,5])
        I = np.eye(2)
        Theta = np.dot(I - np.dot(np.dot(self.Q, np.transpose(self.Q)), self.D ), InitialTheta)
        Theta /= np.sqrt(np.dot(np.dot(np.transpose(Theta), self.D), Theta))

        MaxIter = 10000
        PrevTheta = InitialTheta
        PrevB = np.ones(self.Dim)
        for idx in range(MaxIter):
            NewResp = np.dot(self.Y, Theta)
            elas = ElasticNet(alpha=Flt_Lambda, l1_ratio=Flt_L1)
            #
            # # Compute Coefficient
            # B = lasso.fit(X=self.X, y= NewResp).coef_
            B = elas.fit(X=self.X, y= NewResp).coef_
            # print B
            #
            # New OptScore
            Part1 = I - np.dot(np.dot(self.Q, np.transpose(self.Q)),self.D)
            Part2 = np.dot(Part1, np.linalg.inv(self.D))
            Part3 = np.dot(Part2, np.transpose(self.Y))
            WaveTheta = np.dot(np.dot(Part3, self.X), B)
            # print WaveTheta
            Theta = WaveTheta / np.sqrt(np.dot(np.dot(np.transpose(WaveTheta),self.D),WaveTheta))

            if np.sum(np.abs(B - PrevB)) < 1e-6:
                break
            else:
                PrevB = B

        # print B
        self.B = B 


    def Construct_Y(self):
        NumClass1 = len(self.Data1) # N
        NumClass2 = len(self.Data2) # V
        TotalNum = NumClass1 + NumClass2

        Y = np.zeros((TotalNum, 2))
        for idx in range(len(Y)):
            if idx < NumClass1:
                Y[idx][0] = 1
            else:
                Y[idx][1] = 1

        return Y


if __name__ == "__main__":
    print None