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
''' Function or Class '''


class ConstructStatistics(FeatureSelector):
    def __init__(self, RecordNum , RecordType, Seconds, StrWaveletBasis, IntDecompLevel, LDAorNOT, Threshold, alpha):

        ### Class Inheritance ###

        FeatureSelector.__init__(self, RecordNum , RecordType, Seconds, StrWaveletBasis, IntDecompLevel, LDAorNOT, Threshold)
        #########################

        ### Initial Value Declaration ###

        # print len(self.DictArray_TrainWC), len(self.DictArray_TrainWCNormal), len(self.DictArray_TrainWCPVC)
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

        self.LDAOFF_FisherScore, self.LDAOFF_NumSelected , self.LDAOFF_IdxFeatureSelected = self.LDAOFF_FisherScoreCompute()
        self.LDAON_FisherScore, self.LDAON_NumSelected, self.LDAOFF_IdxFeatureSelected  = self.LDAON_FisherScoreComputation()


    def StatisticsConstruction(self):
        if self.LDAorNOT == False:
            DictFloat_Stat = dict()
            List_FisherScoreForFeatures, Int_NumFeatureSelected, List_IdxFeatureSelected = self.LDAOFF_FisherScoreCompute()
            Array_ReducedMean = self.LDAOFF_ReducedMeanComputation()
            Matrix_ReducedCov = self.LDAOFF_ReducedCoVarComputation()

            for idx, key in sorted(enumerate(self.DictArray_TestWC)):
                Val = self.DictArray_TestWC[key][List_IdxFeatureSelected[:Int_NumFeatureSelected]]
                Val = np.reshape(Val - Array_ReducedMean, (1,len(Val)))
                NewVal = Val * np.matrix(Matrix_ReducedCov).I * Val.T
                DictFloat_Stat[key] = np.squeeze(np.asarray(NewVal))
            return DictFloat_Stat

        else:
            DictFloat_Stat = dict()
            List_FisherScoreForFeatures, Int_NumFeatureSelected, List_IdxFeatureSelected = self.LDAON_FisherScoreComputation()
            Int_TempNumFeatureSelected = self.LDAOFF_NumSelected
            Array_ReducedMean = self.LDAON_ReducedMeanComputation()
            Matrix_ReducedCov = self.LDAON_ReducedCoVarComputation()

            DictArray_TestWC_LDA = self.LDAON_TestWCConstruction()

            for idx, key in sorted(enumerate(DictArray_TestWC_LDA)):
                Val = DictArray_TestWC_LDA[key][List_IdxFeatureSelected[:Int_NumFeatureSelected]]
                # Val = DictArray_TestWC_LDA[key][List_IdxFeatureSelected[:Int_TempNumFeatureSelected]]
                Val = np.reshape(Val - Array_ReducedMean, (1,len(Val)))
                NewVal = Val * np.matrix(Matrix_ReducedCov).I * Val.T
                DictFloat_Stat[key] = np.squeeze(np.asarray(NewVal))
            return DictFloat_Stat

    def UCL(self):
        if self.LDAorNOT == False:
            P = self.LDAOFF_NumSelected
            N = self.Int_NumTrainNormal
            return (P*((N-1)**2) * f.ppf(self.alpha, P, N-P)) / (N*(N-P))
        else:
            S = self.LDAON_NumSelected
            # S = self.LDAOFF_NumSelected
            N = self.Int_NumTrainNormal
            return (S*((N-1)**2) * f.ppf(self.alpha, S, N-S)) / (N*(N-S))

    def AccuracyComputation(self):
        DictFloat_Stat = self.StatisticsConstruction()
        UCLVAL = self.UCL()

        # TYPE1 Error : Blue (Normal) Out Control
        # TYPE2 Error : Red (PVC) In Control
        # TYPE1 Duzi : Blue In Control
        # Type2 Duzi : Red Out Control

        Int_TotalTestPoint = 0
        Int_Type1_Error = 0
        Int_Type1_Duzi = 0
        Int_Type2_Error = 0
        Int_Type2_Duzi = 0

        DictInt_Accuracy = dict()
        DictFloat_Accuracy = dict()

        for idx, key in enumerate(sorted(DictFloat_Stat)):
            print key
            Int_TotalTestPoint += 1
            if self.Dict_TestLabel[key] == 'N' or self.Dict_TestLabel[key] == 'L' or \
                self.Dict_TestLabel[key] == 'R' or self.Dict_TestLabel[key] == 'e' or \
                self.Dict_TestLabel[key] == 'j' :
                print "HOHO"
                if DictFloat_Stat[key] < UCLVAL: # Normal In Control
                    Int_Type1_Duzi += 1
                elif DictFloat_Stat[key] > UCLVAL: # Normal Out Control
                    Int_Type1_Error += 1
            elif self.Dict_TestLabel[key] == 'A' or self.Dict_TestLabel[key] == 'a' or \
                    self.Dict_TestLabel[key] == 'S' or  self.Dict_TestLabel[key] == 'V' or \
                    self.Dict_TestLabel[key] == 'E' :
                print "HAHA"
                if DictFloat_Stat[key] < UCLVAL: # PVC In Control
                    Int_Type2_Error += 1
                elif DictFloat_Stat[key] > UCLVAL: # PVC Out Control
                    Int_Type2_Duzi += 1
            # except:
            #     pass

        DictInt_Accuracy['Type1_Error'] = Int_Type1_Error
        DictInt_Accuracy['Type1_Duzi'] = Int_Type1_Duzi
        DictInt_Accuracy['Type2_Error'] = Int_Type2_Error
        DictInt_Accuracy['Type2_Duzi'] = Int_Type2_Duzi
        DictInt_Accuracy['TotalBeat'] = Int_TotalTestPoint
        DictInt_Accuracy['TotalError'] = Int_Type1_Error + Int_Type2_Error

        Int_TP = Int_Type1_Duzi # Correctly detected beat (Normal as Normal)
        Int_TN = Int_Type2_Duzi # Correctly rejected beat (PVC as PVC)
        Int_FP = Int_Type1_Error # Falsely detected (Normal as PVC)
        Int_FN = Int_Type2_Error # Misssed beat (PVC as Normal)

        # Acc = NT - NE (=NC) / NT
        DictFloat_Accuracy['Accuracy'] = float(Int_TotalTestPoint - (Int_Type1_Error+ Int_Type2_Error)) / float(Int_TotalTestPoint)
        # Se = TP / TP + FN, ratio
        DictFloat_Accuracy['Sensitivity'] = float(Int_TP) / float(Int_TP + Int_FN)
        # Positive Predictivity = TP / TP + FP, ratio of correctly detected beat to the total num of beats
        DictFloat_Accuracy['Positive_Predictivity'] = float(Int_TP) / float(Int_TP + Int_FP)
        # Sp = TN / TN + FP , Correctly rejected to num of nonevent
        DictFloat_Accuracy['Specificity'] = float(Int_Type2_Duzi) / float(Int_Type2_Duzi + Int_Type1_Error)

        return DictInt_Accuracy, DictFloat_Accuracy



    def StatPlot(self):
        DictFloat_Stat = self.StatisticsConstruction()
        print "HAHA"

        if self.LDAorNOT == False:
            UCLVal = self.UCL()
            plt.figure()
            plt.grid()
            plt.xlabel("Index")
            plt.ylabel("Stat")
            plt.title("LDAOFF |" + str(self.RecordNum)+ "|" + str(self.LDAOFF_NumSelected))
            for idx, key in enumerate(sorted(DictFloat_Stat)):
                print key
                print DictFloat_Stat[key]
                if self.Dict_TestLabel[key] == 'N' or self.Dict_TestLabel[key] == 'L' or \
                self.Dict_TestLabel[key] == 'R' or self.Dict_TestLabel[key] == 'e' or \
                self.Dict_TestLabel[key] == 'j' :
                    plt.plot(idx,UCLVal ,'m.')
                    plt.plot(idx, DictFloat_Stat[key], 'bo')
                elif self.Dict_TestLabel[key] == 'A' or self.Dict_TestLabel[key] == 'a' or \
                    self.Dict_TestLabel[key] == 'S' or  self.Dict_TestLabel[key] == 'V' or \
                    self.Dict_TestLabel[key] == 'E' :
                    plt.plot(idx,UCLVal ,'m.')
                    plt.plot(idx, DictFloat_Stat[key], 'ro')

            plt.show()
        else:
            UCLVal = self.UCL()
            plt.figure()
            plt.grid()
            plt.xlabel("Index")
            plt.ylabel("Stat")
            plt.title("LDAON |" + str(self.RecordNum)+"|" + str(self.LDAON_NumSelected))
            # plt.title("LDA Applied Hoteling T |" + str(self.RecordNum)+"|")
            for idx, key in enumerate(sorted(DictFloat_Stat)):
                # print "HAHA "
                if self.Dict_TestLabel[key] == 'N' or self.Dict_TestLabel[key] == 'L' or \
                    self.Dict_TestLabel[key] == 'R' or self.Dict_TestLabel[key] == 'e' or \
                    self.Dict_TestLabel[key] == 'j' :
                    plt.plot(idx, DictFloat_Stat[key], 'bo')
                    plt.plot(idx,UCLVal ,'m.')
                elif self.Dict_TestLabel[key] == 'A' or self.Dict_TestLabel[key] == 'a' or \
                    self.Dict_TestLabel[key] == 'S' or  self.Dict_TestLabel[key] == 'V' or \
                    self.Dict_TestLabel[key] == 'E' :
                    plt.plot(idx, DictFloat_Stat[key], 'ro')
                    plt.plot(idx,UCLVal ,'m.')


            plt.show()
        return None

# ---- EXCLUDED LIST ---------
    # Excluded = 102, 104, 107, 217 (AAMI)
    # 101 : Not Enough Abnormal beats
    # 103 : Not Enough Abnormal beats
    # 111 : Not Enough Abnormal beats
    # 112 : Not Enough Abnormal beats
    # 115 : Not Enough Abnormal beats
    # 121 : Not Enough Abnormal beats
    # 122 : Not Enough Abnormal beats
    # 123 : Not Enough Abnormal beats
    # 124 : Not Enough Abnormal beats
    # 205 : Not Enough Abnormal beats
    # 212 : Not Enough Abnormal beats
    # 222 : Not Enough Abnormal beats
    # 230 : Not Enough Abnormal beats
    # 231 : Not Enough Abnormal beats
    # 232 : Not Enough Abnormal beats

    # 120 : MA of 119


if __name__ == "__main__":
    List_RecordNum = [100, 105, 106, 108, 109, 113, 114, 116, 118, 119,
                      200, 201, 202, 203, 207, 208, 209, 210, 213, 214,
                      215, 219, 220, 221, 223, 228, 233]

    IntRecordNum = 223
    IntRecordType = 0
    IntSeconds = 120

    BoolLDAorNOT = True
    # BoolLDAorNOT = False

    if BoolLDAorNOT == True:
        FltThreshold = 0.99
    else:
        FltThreshold = 0.8
    IntDecompLevel = 4

    StrWaveletBasis = 'db8'
    alpha = 0.9999

    ObjConstructStatistics \
        = ConstructStatistics(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold, alpha=alpha)
    print len(ObjConstructStatistics.DictArray_TrainWC)

    LDAON_FisherScore = ObjConstructStatistics.LDAON_FisherScore
    LDAOFF_FisherScore = ObjConstructStatistics.LDAOFF_FisherScore
    if BoolLDAorNOT == True:
        for val in LDAON_FisherScore:
            print val
    else:
        for val in LDAOFF_FisherScore:
            print val
    print ""

    DictInt_Accuracy, DictFloat_Accuracy = ObjConstructStatistics.AccuracyComputation()
    for idx, key in enumerate(DictInt_Accuracy):
        print key, DictInt_Accuracy[key]

    print ""
    for idx, key in enumerate(DictFloat_Accuracy):
        print key, DictFloat_Accuracy[key]
    #
    # print len(ObjConstructStatistics.DictArray_TrainWCNormal)

    ObjConstructStatistics.StatPlot()






