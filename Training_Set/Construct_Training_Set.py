# -*- coding: utf-8 -*-
'''
Goal : Construct the Training set
Author : Yonghan Jung, IE, KAIST 
Date : 150501
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
class Construct_Training(Data_Preparation):
    def __init__(self, RecordNum, RecordType, Seconds):
        ## Data Load
        Data_Preparation.__init__(self, RecordNum, RecordType)
        self.Sampling_rate = 360
        self.SecondsToSample = Seconds * self.Sampling_rate

    def DataLoad(self):
        ECG_Segment, ECG_Segment_Type = Data_Preparation.Segment(self)
        NormalKeys = Data_Preparation.KeyNum_Type(self,'N')
        PVC_Keys = Data_Preparation.KeyNum_Type(self,'V')
        return ECG_Segment, ECG_Segment_Type, NormalKeys, PVC_Keys

    def TrainDataLoad(self, Level):
        self.Level = Level
        ECG_Segment, ECG_Segment_Type, NormalKeys, PVC_Keys \
            = self.DataLoad()
        Training_ECG_Segment \
            = {R_Idx : ECG_Segment[R_Idx] for R_Idx in ECG_Segment.keys() if R_Idx < self.SecondsToSample}
        Training_ECG_Type \
            = {R_Idx : ECG_Segment_Type[R_Idx] for R_Idx in ECG_Segment_Type.keys() if R_Idx < self.SecondsToSample}

        ## Wavelet analysis needed
        WC_ECG = {}
        for idx, key in sorted(enumerate(Training_ECG_Segment)):
            WCExtractor = Wavelet_Coefficient_Extractor(Training_ECG_Segment[key], Wavelet_Basis_Fun='db8',Level=self.Level)
            WC_ECG_elem = WCExtractor.WaveDec()
            WC_ECG[key] = np.concatenate([WC_ECG_elem[0], WC_ECG_elem[1], WC_ECG_elem[2]])
        WC_ECG = pd.DataFrame.from_dict(data=WC_ECG,orient='index')
        return WC_ECG, Training_ECG_Type


if __name__ == "__main__":
    MyConstruct = Construct_Training(119,0,300)
    WCECG, WCType = MyConstruct.TrainDataLoad(4)
    print WCECG, WCType

    RowIterable = WCECG.T
    for row_idx in RowIterable:
        if WCType[row_idx] == "N":
            plt.plot(RowIterable[row_idx], 'bo')
        elif WCType[row_idx] == "V":
            plt.plot(RowIterable[row_idx], 'ro')
    plt.grid()
    plt.show()










