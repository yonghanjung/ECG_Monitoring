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

    def NoiseTrainDataLoad(self, Wavelet_Basis, Level):
        ECG_Segment, ECG_Segment_Type, NormalKeys, PVC_Keys \
            = self.DataLoad()
        Training_ECG_Segment \
            = {R_Idx : ECG_Segment[R_Idx] for R_Idx in ECG_Segment.keys() if R_Idx < self.SecondsToSample}
        Training_ECG_Type \
            = {R_Idx : ECG_Segment_Type[R_Idx] for R_Idx in ECG_Segment_Type.keys() if R_Idx < self.SecondsToSample}

        ## Wavelet analysis needed
        Noise_WC = {}
        for idx, key in sorted(enumerate(Training_ECG_Segment)):
            WCExtractor = Wavelet_Coefficient_Extractor(Training_ECG_Segment[key], Wavelet_Basis_Fun=Wavelet_Basis,Level=Level)
            WC_ECG_elem = WCExtractor.WaveDec()
            Noise_WC[key] = WC_ECG_elem[Level]

        Noise_WC = pd.DataFrame.from_dict(data = Noise_WC, orient='index')

        return Noise_WC, Training_ECG_Type



    def TrainDataLoad(self, Wavelet_Basis, Level):
        ECG_Segment, ECG_Segment_Type, NormalKeys, PVC_Keys \
            = self.DataLoad()
        Training_ECG_Segment \
            = {R_Idx : ECG_Segment[R_Idx] for R_Idx in ECG_Segment.keys() if R_Idx < self.SecondsToSample}
        Training_ECG_Type \
            = {R_Idx : ECG_Segment_Type[R_Idx] for R_Idx in ECG_Segment_Type.keys() if R_Idx < self.SecondsToSample}

        ## Wavelet analysis needed
        WC_ECG = {}
        for idx, key in sorted(enumerate(Training_ECG_Segment)):
            WCExtractor = Wavelet_Coefficient_Extractor(Training_ECG_Segment[key], Wavelet_Basis_Fun=Wavelet_Basis,Level=Level)
            WC_ECG_elem = WCExtractor.WaveDec()
            # WC_ECG[key] = np.concatenate([WC_ECG_elem[1], WC_ECG_elem[2]])
            WC_ECG[key] = np.concatenate([WC_ECG_elem[0], WC_ECG_elem[1], WC_ECG_elem[2]])
        WC_ECG = pd.DataFrame.from_dict(data=WC_ECG,orient='index')

        return WC_ECG, Training_ECG_Type

    def NoiseTestDataLoad(self, Wavelet_Basis, Level):
        ECG_Segment, ECG_Segment_Type, NormalKeys, PVC_Keys \
            = self.DataLoad()
        Test_ECG_Segment \
            = {R_Idx : ECG_Segment[R_Idx] for R_Idx in ECG_Segment.keys() if R_Idx > self.SecondsToSample}
        Test_ECG_Type \
            = {R_Idx : ECG_Segment_Type[R_Idx] for R_Idx in ECG_Segment_Type.keys() if R_Idx > self.SecondsToSample}
        Noise_WC = {}
        for idx, key in sorted(enumerate(Test_ECG_Segment)):
            WCExtractor = Wavelet_Coefficient_Extractor(Test_ECG_Segment[key], Wavelet_Basis_Fun=Wavelet_Basis,Level=Level)
            WC_ECG_elem = WCExtractor.WaveDec()
            Noise_WC[key] = WC_ECG_elem[Level]
        Noise_WC = pd.DataFrame.from_dict(data=Noise_WC,orient='index')

        return Noise_WC, Test_ECG_Type

    def TestDataLoad(self, Wavelet_Basis, Level):
        ECG_Segment, ECG_Segment_Type, NormalKeys, PVC_Keys \
            = self.DataLoad()
        Test_ECG_Segment \
            = {R_Idx : ECG_Segment[R_Idx] for R_Idx in ECG_Segment.keys() if R_Idx > self.SecondsToSample}
        Test_ECG_Type \
            = {R_Idx : ECG_Segment_Type[R_Idx] for R_Idx in ECG_Segment_Type.keys() if R_Idx > self.SecondsToSample}
        WC_ECG = {}
        for idx, key in sorted(enumerate(Test_ECG_Segment)):
            WCExtractor = Wavelet_Coefficient_Extractor(Test_ECG_Segment[key], Wavelet_Basis_Fun=Wavelet_Basis,Level=Level)
            WC_ECG_elem = WCExtractor.WaveDec()
            WC_ECG[key] = np.concatenate([WC_ECG_elem[0], WC_ECG_elem[1], WC_ECG_elem[2]])
        WC_ECG = pd.DataFrame.from_dict(data=WC_ECG,orient='index')

        return WC_ECG, Test_ECG_Type

    def Plot(self, Wavelet_Basis, Level):
        WCECG, WCType = self.TrainDataLoad(Wavelet_Basis,Level)
        RowIterable = WCECG.T
        for row_idx in RowIterable:
            if WCType[row_idx] == "N":
                plt.plot(RowIterable[row_idx], 'bo')
            elif WCType[row_idx] == "V":
                plt.plot(RowIterable[row_idx], 'ro')
        plt.grid()
        plt.show()


if __name__ == "__main__":
    MyConstruct = Construct_Training(119,0,300)
    WCECG, WCType = MyConstruct.TrainDataLoad('db8',4)
    print WCECG, WCType

    RowIterable = WCECG.T
    for row_idx in RowIterable:
        if WCType[row_idx] == "N":
            plt.plot(RowIterable[row_idx], 'bo')
        elif WCType[row_idx] == "V":
            plt.plot(RowIterable[row_idx], 'ro')
    plt.grid()
    plt.show()










