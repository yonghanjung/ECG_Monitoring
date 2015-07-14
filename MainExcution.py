# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
from Data_Preparation import Data_Preparation
from Wavelet_Analysis.Wavelet_Analysis import Wavelet_Coefficient_Extractor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

''' Function or Class '''


class MainExecution(Data_Preparation):
    def __init__(self,RecordNum, RecordType, Seconds):
        ## Data Load
        Data_Preparation.__init__(self, RecordNum, RecordType)
        self.Sampling_rate = 360
        self.SecondsToSample = Seconds * self.Sampling_rate

    def DataLoad(self):
        ECG_Segment, ECG_Segment_Type = Data_Preparation.Segment(self)
        NormalKeys = Data_Preparation.KeyNum_Type(self,'N')
        PVC_Keys = Data_Preparation.KeyNum_Type(self,'V')
        return ECG_Segment, ECG_Segment_Type, NormalKeys, PVC_Keys





if __name__ == "__main__":
    print None