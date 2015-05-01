# -*- coding: utf-8 -*-
'''
Goal : Plot
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

if __name__ == "__main__":
    Record_Num_List = \
        [105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 217, 219, 221, 223, 228, 233]
    testnum = 106
    # ECG_data = Data_Loading(testnum,record_type=0).Data_Loading()
    # print ECG_data
    ECG_data = Data_Preparation(testnum,0)
    ECG_segment, ECG_type =  ECG_data.Segment()
    Normal_KeyNum = ECG_data.KeyNum_Type('N')
    PVC_KeyNum = ECG_data.KeyNum_Type('V')

    PVC_Example = PVC_KeyNum[0]
    PVC_Example_ECG = ECG_segment[PVC_Example]
    WA = Wavelet_Coefficient_Extractor(PVC_Example_ECG, Wavelet_Basis_Fun='db4', Level=4)
    A = WA.WaveDec()

    plt.plot(A[0])
    plt.show()
    # plt.plot(ECG_segment[PVC_Example])
    # plt.grid()
    # plt.show()

