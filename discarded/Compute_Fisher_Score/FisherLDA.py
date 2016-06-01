# -*- coding: utf-8 -*-
'''
Goal : Implement Highdimensional Fisher LDA
Author : Yonghan Jung, ISyE, KAIST 
Date : 150609_FisherLDA
Comment 
- FisherLDA

'''

''' Library '''
import numpy as np
import scipy as sp
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
from HansFisherScore import Fisher_Score_Compute



''' Function or Class '''


class FisherLDA:
    def __init__(self, TrainingData, Num):
        # TrainingData have to be the dictionary
        self.TrainData = TrainingData
        self.Num = Num

        self.Class1 = np.array(self.TrainData[0])
        print self.Class1.shape
        # Row : Data records
        # Column : Data dimension
        # (N by 64)
        self.Mu1 = np.mean(self.Class1, axis=0)
        # 64 by 1
        self.Mu1 = self.Mu1.reshape(len(self.Mu1), 1)

        self.Class2 = np.array(self.TrainData[1])
        self.Mu2 = np.mean(self.Class2, axis=0)
        self.Mu2 = self.Mu2.reshape(len(self.Mu2), 1)
        # 64 by 1

    def WithInClass(self):
        Result = np.zeros((len(self.Mu1), len(self.Mu1)))
        #print "Length of result", len(Result)
        for val in self.Class1:
            val = val.reshape(len(val),1)
            Result += np.dot((val - self.Mu1), (val - self.Mu1).T)
        for idx, val in enumerate(self.Class2):
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
        EigVal, EigMat = np.linalg.eigh(TargetMat)
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
    np.random.seed(30)
    MyTraining = dict()
    Mu1 = np.array([mu1] * dim)
    COV1 = np.eye(dim)
    # It is common to arrange data in column form
    DataC1 = np.random.multivariate_normal(Mu1, COV1, Num)
    MyTraining[0] = DataC1

    Mu2 = np.array([mu2] * dim)
    COV2 = np.eye(dim)
    DataC2 = np.random.multivariate_normal(Mu2, COV2, Num)
    MyTraining[1] = DataC2

    return MyTraining


def TestData(dim, mu1, mu2, Num):
    np.random.seed(30)
    Mu1 = np.array([mu1] * dim)
    COV1 = np.eye(dim)
    # It is common to arrange data in column form
    DataC1 = np.random.multivariate_normal(Mu1, COV1, Num).T

    Mu2 = np.array([mu2] * dim)
    COV2 = np.eye(dim)
    DataC2 = np.random.multivariate_normal(Mu2, COV2, Num).T
    Data = np.concatenate([DataC1,DataC2], axis=1)

    return Data.T



if __name__ == "__main__":
    Dim = 10
    Mu1 = -1
    Mu2 = 1
    Num = 40
    MyTest = TestData(Dim,Mu1,Mu2,Num)
    MyTraining = TrainingData(Dim, Mu1, Mu2, Num)
    MyLDA = FisherLDA(MyTraining,Dim)


    ##### DEFINE LDA ########

    NewTest = MyLDA.LDATransform(MyTest)

    MyTestDict = dict()
    MyTestDict[0] = MyTest[:Num]
    MyTestDict[1] = MyTest[Num:]

    NewTestDict = dict()
    NewTestDict[0] = NewTest[:Num]
    NewTestDict[1] = NewTest[Num:]

    Fisher_Score_Training = Fisher_Score_Compute(Training=MyTraining)
    Fisher_Score_Testing = Fisher_Score_Compute(Training=NewTestDict)

    print Fisher_Score_Training.Fisher_Score()
    print Fisher_Score_Testing.Fisher_Score()

    plt.figure()
    plt.grid()
    plt.title("BEFORE")
    for RowData in MyTest[:Num]:
        plt.plot(RowData, 'bo')
    for RowData in MyTest[Num:]:
        plt.plot(RowData, 'ro')

    plt.figure()
    plt.grid()
    plt.title("AFTER")
    for RowData in NewTest[:Num]:
        plt.plot(RowData, 'bo')
    for RowData in NewTest[Num:]:
        plt.plot(RowData, 'ro')


    plt.show()