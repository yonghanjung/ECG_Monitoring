# -*- coding: utf-8 -*-
'''
Goal : Extract the wavelet coefficient for ECG signal
Author : Yonghan Jung, IE, KAIST 
Date : 150501
Comment 
- 
'''

''' Library '''
import pywt
''' Function or Class '''
class Wavelet_Coefficient_Extractor:
    def __init__(self, Data, Wavelet_Basis_Fun, Level):
        self.Data = Data
        self.WBF = pywt.Wavelet(Wavelet_Basis_Fun)
        self.Level = Level
    def WaveDec(self):
        return pywt.wavedec(data = self.Data, mode = 'per' , wavelet = self.WBF, level = self.Level)

