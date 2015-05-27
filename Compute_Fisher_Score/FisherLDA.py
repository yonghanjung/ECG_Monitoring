# -*- coding: utf-8 -*-
'''
Goal : Implement Highdimensional Fisher LDA
Author : Yonghan Jung, ISyE, KAIST 
Date : 150527
Comment 
- FisherLDA

'''

''' Library '''
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt



''' Function or Class '''


class HansFisherLDA:
    def __init__(self, TrainingData, Num):
        self.TrainData = TrainingData
        self.Num = Num

        self.Class1 = np.array(self.TrainData[0]).T
        self.Mu1 = np.mean(self.Class1, axis=1)
        self.Mu1 = self.Mu1.reshape(len(self.Mu1), 1)

        self.Class2 = np.array(self.TrainData[1]).T
        self.Mu2 = np.mean(self.Class2, axis=1)
        self.Mu2 = self.Mu2.reshape(len(self.Mu2), 1)

    def WithInClass(self):
        Result = np.zeros((len(self.Mu1), len(self.Mu1)))
        for val in self.Class1.T:
            val = val.reshape(len(val),1)
            Result += np.dot((val - self.Mu1), (val - self.Mu1).T)
        for idx, val in enumerate(self.Class2.T):
            val = val.reshape(len(val),1)
            Result += np.dot((val - self.Mu2), (val - self.Mu2).T)
        return Result

    def BetweenClass(self):
        return np.dot((self.Mu1 - self.Mu2), (self.Mu1 - self.Mu2).T)

    def SQRTInverseMatrix(self, MyArray):
        EigVal, EigMat = np.linalg.eigh(MyArray)
        EigDiag = np.eye(len(EigVal))
        # print EigDiag

        EigMat = np.matrix(EigMat)
        for idx in range(len(EigVal)):
            EigDiag[idx][idx] = (np.sqrt(EigVal[idx]))
        EigDiag = np.matrix(EigDiag)
        return EigMat * EigDiag * (EigMat.I)

    def LeadingEig(self):
        WithIn = self.WithInClass()
        Between = np.matrix(self.BetweenClass())
        SQRTInverseWithIn = self.SQRTInverseMatrix(WithIn)

        TargetMat = (SQRTInverseWithIn).I * Between * (SQRTInverseWithIn).I
        # print TargetMat
        EigVal, EigMat = np.linalg.eig(TargetMat)
        IDX = np.argsort(EigVal)[::-1]
        EigMat = EigMat[:,IDX]
        return EigMat[:,:self.Num]


    def LDAOperator(self):
        WithIn = self.WithInClass()

        SQRTInverseWithIn = self.SQRTInverseMatrix(WithIn)
        LeadEig = np.matrix(self.LeadingEig())
        return (SQRTInverseWithIn).I * LeadEig


    ######## W is constructed #########
    def LDATransform(self, TestDataSet):
        W = self.LDAOperator()
        NewTest = []
        for testval in TestDataSet:
            testval = testval.reshape(len(testval) * 1)
            testval = np.array(np.dot(W, testval))
            NewTest.append(testval[0])
        return np.array(NewTest)


def TrainingData(dim, mu1, mu2, Num):
    np.random.seed(12)
    MyTraining = dict()
    Mu1 = np.array([mu1] * dim)
    COV1 = np.eye(dim)
    # It is common to arrange data in column form
    DataC1 = np.random.multivariate_normal(Mu1, COV1, Num).T
    MyTraining[0] = DataC1

    Mu2 = np.array([mu2] * dim)
    COV2 = np.eye(dim)
    DataC2 = np.random.multivariate_normal(Mu2, COV2, Num).T
    MyTraining[1] = DataC2

    return MyTraining


def TestData(dim, mu1, mu2, Num):
    np.random.seed(12)
    Mu1 = np.array([mu1] * dim)
    COV1 = np.eye(dim) * 1
    # It is common to arrange data in column form
    DataC1 = np.random.multivariate_normal(Mu1, COV1, Num).T

    Mu2 = np.array([mu2] * dim)
    COV2 = np.eye(dim)
    DataC2 = np.random.multivariate_normal(Mu2, COV2, Num).T
    Data = np.concatenate([DataC1,DataC2], axis=1)

    return Data.T



if __name__ == "__main__":
    Dim = 4
    Mu1 = -1
    Mu2 = 1
    Num = 100
    MyTest = TestData(Dim,Mu1,Mu2,Num)
    MyTraining = TrainingData(Dim, Mu1, Mu2, Num)
    MyLDA = FisherLDA(MyTraining,Dim)

    NewTest = MyLDA.LDATransform(MyTest)


    plt.figure()
    plt.grid()
    plt.plot(MyTest[:Num,0], MyTest[:Num,1], 'ro')
    plt.plot(MyTest[Num:,0], MyTest[Num:,1], 'bo')

    plt.figure()
    plt.grid()
    plt.plot(NewTest[:Num,0], NewTest[:Num,1], 'ro')
    plt.plot(NewTest[Num:,0], NewTest[Num:,1], 'bo')

    plt.show()