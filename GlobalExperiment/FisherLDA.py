# -*- coding: utf-8 -*-
'''
Goal : Fisher LDA 를 계산한다.
Author : Yonghan Jung, ISyE, KAIST
Date : 150730 Fisher LDA 를 계산한다.
Comment


'''

''' Library '''
import numpy as np

''' Function or Class '''


class TwoClassFisherLDA:
    # Input으로 DataDictionary 가 들어간다.
    def __init__(self, Dict_DataMatrix):
        self.Key1 = Dict_DataMatrix.keys()[0]
        self.Key2 = Dict_DataMatrix.keys()[1]
        # 데이터는 row 가 record, column 이 feature
        self.ArrayMat_Data1 = Dict_DataMatrix[self.Key1]
        self.ArrayMat_Data2 = Dict_DataMatrix[self.Key2]

        self.Sw = self.Compute_CovBW(Data_Class1=self.ArrayMat_Data1, Data_Class2=self.ArrayMat_Data2)
        self.Sb = self.Compute_CovWC(Data_Class1=self.ArrayMat_Data1, Data_Class2=self.ArrayMat_Data2)
        Mu1 = np.mean(self.ArrayMat_Data1 , axis=0)
        self.Mu1 = np.reshape(Mu1, (len(Mu1), 1))
        Mu2 = np.mean(self.ArrayMat_Data2 , axis=0)
        self.Mu2 = np.reshape(Mu2, (len(Mu2), 1))

        print "WithMat shape", self.Sw.shape
        print "Mu shape", self.Mu1.shape
        self.W = np.dot(np.linalg.inv(self.Sw), (self.Mu1 - self.Mu2))




    # (Mean1 - Mean2) * (Mean1 - Mean2).T
    # Mu 는 D by 1 데이터이다.
    def Compute_CovBW(self, Data_Class1, Data_Class2):
        Mu1 = np.mean(Data_Class1, axis=0)
        Mu1 = np.reshape(Mu1, (len(Mu1), 1))
        Mu2 = np.mean(Data_Class2, axis=0)
        Mu2 = np.reshape(Mu2, (len(Mu2), 1))
        return np.dot((Mu1 - Mu2), np.transpose(Mu1 - Mu2) )

    # 각 데이터는 D by 1 데이터이다.
    def Compute_CovWC(self,Data_Class1, Data_Class2):
        Mu1 = np.mean(Data_Class1, axis=0)
        Mu1 = np.reshape(Mu1, (len(Mu1), 1))
        Mu2 = np.mean(Data_Class2, axis=0)
        Mu2 = np.reshape(Mu2, (len(Mu2), 1))

        Cov = np.zeros((len(Mu1), len(Mu1)))

        for data1 in Data_Class1:
            data1 = np.reshape(data1, (len(data1),1))
            Cov += np.dot(data1 - Mu1, np.transpose(data1 - Mu1))
        for data2 in Data_Class2:
            data2 = np.reshape(data2, (len(data2),1))
            Cov += np.dot(data2 - Mu2, np.transpose(data2 - Mu2))

        return Cov

if __name__ == "__main__":
    print None