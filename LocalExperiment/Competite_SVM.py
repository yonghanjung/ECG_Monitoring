# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f
from FeatureSelector3 import FeatureSelector
from sklearn import svm

''' Function or Class '''


class Competitive_SVM(FeatureSelector):
    def __init__(self, RecordNum , RecordType, Seconds, StrWaveletBasis, IntDecompLevel, LDAorNOT, Threshold, alpha):
        ### Class Inheritance ###
        FeatureSelector.__init__(self, RecordNum , RecordType, Seconds, StrWaveletBasis, IntDecompLevel, LDAorNOT, Threshold)
        #########################

        self.RecordNum = RecordNum
        self.RecordType = RecordType
        self.Sampling_rate = 360
        self.SecondsToSample = Seconds * self.Sampling_rate
        self.LDAorNOT = LDAorNOT
        self.alpha = alpha
        self.Str_WaveletBasis = StrWaveletBasis
        self.Int_DecompLevel = IntDecompLevel
        self.Float_Threshold = Threshold

        # COL #
        self.Int_NumTrainFeature = self.Int_NumTrainFeature
        # ROW #
        self.Int_NumTrainNormal = self.Int_NumTrainNormal

        # self.LDAOFF_FisherScore, self.LDAOFF_NumSelected , self.LDAOFF_IdxFeatureSelected = self.LDAOFF_FisherScoreCompute()
        # self.LDAON_FisherScore, self.LDAON_NumSelected, self.LDAOFF_IdxFeatureSelected  = self.LDAON_FisherScoreComputation()

        # Training Data
        self.LDAON_TrainWCNormalConstruction()
        self.LDAON_TrainWCPVCConstruction()

        # TestData
        self.LDAON_TestWCConstruction()


        # TestLabel
        self.DictArray_TestWC, self.Dict_TestLabel = self.TestWCConstruction()




if __name__ == "__main__":
    List_RecordNum = [100, 105, 106, 108, 109, 113, 114, 116, 118, 119,
                      200, 201, 202, 203, 207, 208, 209, 210, 213, 214,
                      215, 219, 220, 221, 223, 228, 233]

    List_Test = [100, 103, 105, 111, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 233, 234]
    List_Train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 220]
    VEB = [200, 202, 210, 213, 214, 219, 221, 228, 231, 233, 234]
    SVEB = [200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

    IntRecordNum = 228
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
    alpha = 0.9999

    ObjSVM = Competitive_SVM(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold, alpha=alpha)
    Dict_KeyRecord_ValWX_TrainWCNormal = ObjSVM.LDAON_TrainWCNormalConstruction()
    Dict_KeyRecord_ValWX_TrainWCPVC = ObjSVM.LDAON_TrainWCPVCConstruction()
    Dict_KeyRecord_ValWX_Test = ObjSVM.LDAON_TestWCConstruction()
    Dict_KeyRecord_ValLabel_Test = ObjSVM.Dict_TestLabel

    ArrayMat_Train = list()
    Y = list()
    for idx, key in enumerate(sorted(Dict_KeyRecord_ValWX_TrainWCNormal)):
        ArrayMat_Each = Dict_KeyRecord_ValWX_TrainWCNormal[key]
        ArrayMat_Train.append(ArrayMat_Each)
        Y.append(0)
    for idx, key in enumerate(sorted(Dict_KeyRecord_ValWX_TrainWCPVC)):
        ArrayMat_Each = Dict_KeyRecord_ValWX_TrainWCPVC[key]
        ArrayMat_Train.append(ArrayMat_Each)
        Y.append(1)

    ArrayMat_Test = list()
    for idx, key in enumerate(sorted(Dict_KeyRecord_ValWX_Test)):
        ArrayMat_Each = Dict_KeyRecord_ValWX_Test[key]
        ArrayMat_Test.append(ArrayMat_Each)

    TrueAnswer = list()
    for idx, key in enumerate(sorted(Dict_KeyRecord_ValLabel_Test)):
        if Dict_KeyRecord_ValLabel_Test[key] == "N" or Dict_KeyRecord_ValLabel_Test[key] == "R" or Dict_KeyRecord_ValLabel_Test[key] == "L" or Dict_KeyRecord_ValLabel_Test[key] == "e" or Dict_KeyRecord_ValLabel_Test[key] == "j":
            TrueAnswer.append(0)
        # elif Dict_KeyRecord_ValLabel_Test[key] == "A" or "a" or "S" or "J":
        elif Dict_KeyRecord_ValLabel_Test[key] == "V" or Dict_KeyRecord_ValLabel_Test[key] == "E":
            # print Dict_KeyRecord_ValLabel_Test[key]
            TrueAnswer.append(1)

    ArrayMat_Train= np.array(ArrayMat_Train)
    ArrayMat_Train = np.reshape(ArrayMat_Train, (len(ArrayMat_Train),1))
    ArrayMat_Test = np.array(ArrayMat_Test)
    ArrayMat_Test = np.reshape(ArrayMat_Test, (len(ArrayMat_Test),1))
    Y = np.array(Y)
    print len(Y)

    clf = svm.SVC()
    clf.fit(ArrayMat_Train, Y)
    # 0 : Normal
    # 1 : PVC
    SVMAnswer = clf.predict(ArrayMat_Test)

    NormalAsNormal = 0
    NormalAsVEB = 0
    VEBASVEB = 0
    VEBASNormal = 0

    for a,b in zip(SVMAnswer,TrueAnswer):
        if b ==0 and a == 0:
            NormalAsNormal+= 1
        elif b == 0 and a == 1:
            NormalAsVEB += 1
        elif a == 0 and b == 1:
            VEBASNormal += 1
        elif a == 1 and b == 1 :
            VEBASVEB += 1

    print "Record", IntRecordNum
    print "Normal(G) as Normal" , NormalAsNormal
    print "Normal(G) as VEB" , NormalAsVEB
    print "VEB(G) as Normal", VEBASNormal
    print "VEB(G) as VEB", VEBASVEB

