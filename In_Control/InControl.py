# -*- coding: utf-8 -*-
'''
Goal : Implement InControl Signal
Author : Yonghan Jung, ISyE, KAIST 
Date : 150504
Comment 
- 

'''

''' Library '''
from Training_Set.Construct_Training_Set import Construct_Training
import pandas as pd
import numpy as np

''' Function or Class '''


class InControl():
    def __init__(self, RecordNum, RecordType, Seconds, WaveletBasis, Level):
        self.RecordNum = RecordNum
        self.RecordType = RecordType
        self.Seconds = Seconds
        self.WaveletBasis = WaveletBasis
        self.Level = Level

        ConstTrain = Construct_Training(self.RecordNum, self.RecordType, self.Seconds)
        self.WCTrainECG, self.WCTrainLabel \
            = ConstTrain.TrainDataLoad(self.WaveletBasis, self.Level)


    def TrainData(self):
        return self.WCTrainECG, self.WCTrainLabel

    def TestData(self):
        return self.WCTestECG, self.WCTestLabel

    def NormalExtractor(self):
        Data = self.WCTrainECG.T
        NormalData = dict((key,value) for key, value in Data.iteritems() if self.WCTrainLabel[key] == 'N')
        PVCData = dict((key,value) for key, value in Data.iteritems() if self.WCTrainLabel[key] == 'V')
        return NormalData, PVCData

    def AvgNormal(self):
        NormalData = self.NormalExtractor()[0]
        AvgNum = float(len(NormalData.keys()))
        DataLength = len(NormalData[NormalData.keys()[1]])
        AvgSignal = np.array([float(0)] * DataLength)

        for key in NormalData.keys():
            AvgSignal += NormalData[key]
        return AvgNum, AvgSignal / AvgNum

if __name__ == "__main__":
    Wavelet_Basis = 'db8'
    InCObj = InControl(RecordNum=119, RecordType=0, Seconds=30, WaveletBasis=Wavelet_Basis, Level=4)
    TrainData, DataLabel = InCObj.TrainData()
    NormalData = InCObj.NormalExtractor()[0]
    PVCData = InCObj.NormalExtractor()[1]




