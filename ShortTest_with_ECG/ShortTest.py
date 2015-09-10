# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 150905
Comment 
- 

'''

''' Library '''
from LocalExperiment.DataConstruction1 import DataConstruction
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

''' Function or Class '''

if __name__ == "__main__":
    RecordNum = 111
    RecordType = 0
    Seconds = 300
    StrWaveletBasis = 'db8'
    IntDecompLevel = 4
    Obj = DataConstruction(RecordNum, RecordType, Seconds, StrWaveletBasis, IntDecompLevel)
    IndexDict = Obj.Index_Loading()

    R_Point =  IndexDict['Sample']
    RRType = IndexDict['Type']
    RR_Dict = dict()

    RR_interval = np.array(R_Point[1:]) - np.array(R_Point[:len(R_Point)-1])
    RR_interval = np.array(RR_interval) / 360.0
    print RR_interval
    RRType = RRType[1:]


    # IR_Normal = list()
    # IR_PVC = list()
    # IR_List = list()
    # for idx_RR in range(len(RR_interval)):
    #     if idx_RR > 0 and idx_RR < len(RR_interval)-1:
    #         IR = (RR_interval[idx_RR] - RR_interval[idx_RR-1]) / (RR_interval[idx_RR+1] - RR_interval[idx_RR])
    #         if np.abs(IR) > 1000:
    #             continue
    #         if RRType[idx_RR] == 'N':
    #             IR_Normal.append(IR)
    #             IR_List.append(IR)
    #         elif RRType[idx_RR] == 'V':
    #             IR_PVC.append(IR)
    #             IR_List.append(IR)

    RR_interval_Normal = list()
    RR_interval_PVC = list()

    for idx in range(len(RR_interval)):
        RR = RR_interval[idx]
        if RRType[idx] == 'N' or RRType[idx] == 'L' or RRType[idx] == 'R':
            RR_interval_Normal.append(RR)
        elif RRType[idx] == 'V':
            RR_interval_PVC.append(RR)



    plt.figure()
    plt.title(str(RecordNum) + " RR interval")
    for idx in range(len(RR_interval)):
        RR = RR_interval[idx]
        if RRType[idx] == 'N' or RRType[idx] == 'L' or RRType[idx] == 'R':
            plt.plot(idx, RR, 'bo')
        elif RRType[idx] == 'V':
            plt.plot(idx, RR, 'ro')

    Density_PVC = gaussian_kde(RR_interval_PVC)
    Density_Normal = gaussian_kde(RR_interval_Normal)
    Domain_PVC = np.linspace(np.min(RR_interval_PVC)-1, np.max(RR_interval_PVC)+1, 1000)
    Domain_Normal = np.linspace(np.min(RR_interval_Normal)-1, np.max(RR_interval_Normal)+1, 1000)

    plt.figure()
    plt.title(str(RecordNum) + " Density")
    plt.plot(Domain_PVC, Density_PVC(Domain_PVC),'r', label="PVC")
    plt.plot(Domain_Normal, Density_Normal(Domain_Normal),'b', label="Normal")
    plt.legend()

    # plt.figure()
    # plt.title("IR Ratio")
    # for idx in range(len(IR_List)):
    #     RR = IR_List[idx]
    #     if RRType[idx] == 'N':
    #         plt.plot(idx, RR, 'bo')
    #     elif RRType[idx] == 'V':
    #         plt.plot(idx, RR, 'ro')

    plt.show()








