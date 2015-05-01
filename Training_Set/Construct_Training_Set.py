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

''' Function or Class '''
class Construct_Training(Data_Preparation, Wavelet_Coefficient_Extractor):
    def __init__(self, RecordNum, RecordType, Seconds):
        Data_Preparation.__init__(self, RecordNum, RecordType)

if __name__ == "__main__":
    print None
