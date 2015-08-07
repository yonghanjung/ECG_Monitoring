# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
from DataConstruction1 import DataConstruction
import pandas as pd
import numpy as np

''' Function or Class '''


class InControlConstruction(DataConstruction):
    def __init__(self, RecordNum, RecordType, Seconds, StrWaveletBasis, IntDecompLevel):
        ## Default Setting ##
        DataConstruction.__init__(self, RecordNum, RecordType, Seconds, StrWaveletBasis, IntDecompLevel)
        self.Sampling_rate = 360
        self.SecondsToSample = Seconds * self.Sampling_rate
        self.Str_WaveletBasis = StrWaveletBasis
        self.Int_DecompLevel = IntDecompLevel

        ## Train Data Load ##
        self.DictArray_TrainWC, self.DictArray_TrainWCNormal, self.DictArray_TrainWCPVC, self.Dict_TrainLabel = self.TrainWCConstruction()
        self.Int_NumTrainNormal = len(self.DictArray_TrainWCNormal)
        self.Int_NumTrainFeature = len(self.DictArray_TrainWCNormal[self.DictArray_TrainWCNormal.keys()[0]])
        ## Test Data Load ##
        self.DictArray_TestWC, self.Dict_TestLabel = self.TestWCConstruction()
        # print self.Dict_TestLabel

    def InControlConstruction(self):
        # InControl constructed only with normal
        Array_SumTrainWCNormal = np.array([float(0)] * self.Int_NumTrainFeature)

        for idx, key in enumerate(self.DictArray_TrainWCNormal):
            Array_SumTrainWCNormal += self.DictArray_TrainWCNormal[key]
        Array_SumTrainWCNormal /= float(self.Int_NumTrainNormal)
        return Array_SumTrainWCNormal


if __name__ == "__main__":
    IntRecordNum = 223
    IntRecordType = 0
    IntSeconds = 300

    StrWaveletBasis = 'db6'
    IntDecompLevel = 4

    ObjInControl = InControlConstruction(RecordType=IntRecordType, RecordNum=IntRecordNum, Seconds=IntSeconds, StrWaveletBasis= StrWaveletBasis, IntDecompLevel = IntDecompLevel)
    print ObjInControl.InControlConstruction()