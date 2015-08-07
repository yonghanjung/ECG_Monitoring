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
''' Function or Class '''


class TwoClassFisherLDA:
    def __init__(self, TrainingData):
        self.TrainData = TrainingData
        # self.Num = Num

        self.Class1 = np.array(self.TrainData[0])
        self.Mu1 = np.mean(self.Class1, axis=0)
        self.Mu1 = self.Mu1.reshape(len(self.Mu1), 1)

        self.Class2 = np.array(self.TrainData[1])
        self.Mu2 = np.mean(self.Class2, axis=0)
        # print "MU2", self.Mu2
        self.Mu2 = self.Mu2.reshape(len(self.Mu2), 1)

    def WithInClass(self):
        Result = np.zeros((len(self.Mu1), len(self.Mu1)))
        # print "Shape of mu",self.Mu1.shape, self.Mu2.shape
        for val in self.Class1:
            val = val.reshape(len(val),1)
            Result += np.dot((val - self.Mu1), (val - self.Mu1).T)
        for idx, val in enumerate(self.Class2):
            val = val.reshape(len(val),1)
            Result += np.dot((val - self.Mu2), (val - self.Mu2).T)
        return Result

    def BetweenClass(self):
        return np.dot((self.Mu1 - self.Mu2), (self.Mu1 - self.Mu2).T)

    def ConstructW(self):
        Sw = self.WithInClass()
        SwInv = np.linalg.inv(Sw)
        return np.dot(SwInv, (self.Mu2 - self.Mu1))

def TrainingData(dim, mu1, mu2, Num):
    np.random.seed(0)
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
    np.random.seed(17249)
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
    Mu1 = -2
    Mu2 = 2
    Num = 40
    MyTest = TestData(Dim,Mu1,Mu2,Num)
    MyTraining = TrainingData(Dim, Mu1, Mu2, Num)

    ObjLDA = TwoClassFisherLDA(TrainingData=MyTraining)
    print ObjLDA.ConstructW()
