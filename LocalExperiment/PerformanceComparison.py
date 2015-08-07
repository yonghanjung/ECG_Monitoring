# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
import numpy as np
import csv
''' Function or Class '''
from Compete_NeuralNet import Competitive_NN
from Compete_SVM import Competitive_SVM
from Statistics4 import ConstructStatistics
from FeatureSelector3 import FeatureSelector


class Performance_Comparison(FeatureSelector):
    def __init__(self, IntRecordNum , IntRecordType, IntSeconds, StrWaveletBasis, IntDecompLevel, LDAorNOT, Threshold, alpha):
        FeatureSelector.__init__(self, IntRecordNum , IntRecordType, IntSeconds, StrWaveletBasis, IntDecompLevel, LDAorNOT, Threshold)
        Obj_NN = Competitive_NN(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold, alpha=alpha)
        Obj_SVM = Competitive_SVM(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold, alpha=alpha)
        Obj_Mine = ConstructStatistics(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold, alpha=alpha)

        self.Dict_NN = Obj_NN.Result()
        self.Dict_SVM = Obj_SVM.Result()
        self.Dict_Mine, _ = Obj_Mine.AccuracyComputation()





if __name__ == "__main__":
    List_RecordNum = [100, 105, 106, 108, 109, 113, 114, 116, 118, 119,
                      200, 201, 202, 203, 207, 208, 209, 210, 213, 214,
                      215, 219, 220, 221, 223, 228, 233]

    List_Test = [100, 103, 105, 111, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 233, 234]
    List_Train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 220]
    VEB = [200, 202, 210, 213, 214, 219, 221, 228, 231, 233, 234]
    SVEB = [200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


    IntRecordType = 0
    IntSeconds = 300

    BoolLDAorNOT = True
    # BoolLDAorNOT = False

    if BoolLDAorNOT == True:
        FltThreshold = 0.99
    else:
        FltThreshold = 0.8
    IntDecompLevel = 4

    StrWaveletBasis = 'db8'

    alpha = 0.975

    print "DataRecordNum | Measure | Mine | SVM | NN"

    # CSV open
    with open('150807_Performance.csv', 'wb') as csvfile:
        fieldnames = ['DataRecordNum', 'Measure', 'Mine', 'SVM', 'NN']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for IntRecordNum in List_RecordNum:
            Obj_Comp = Performance_Comparison(IntRecordNum = IntRecordNum , IntRecordType = IntRecordType, IntSeconds = IntSeconds, StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT = True, Threshold = FltThreshold, alpha = alpha)
            Dict_Mine = Obj_Comp.Dict_Mine
            Dict_SVM = Obj_Comp.Dict_SVM
            Dict_NN = Obj_Comp.Dict_NN


            for (k1,v1), (k2,v2), (k3,v3) in zip(Dict_Mine.items(), Dict_SVM.items(), Dict_NN.items()):
                print IntRecordNum, k1, "|", v1, "|", v2, "|", v3
                writer.writerow({'DataRecordNum': IntRecordNum, 'Measure' : k1, 'Mine': v1, 'SVM' : v2, 'NN' : v3})
