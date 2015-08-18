# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- VEB, SVEB 로 모드 바꿀 때는
Plot, Accuracy, 데이터 읽는거
3개를 모두 바꾸어야 한다.
VEBSVEB 를 검색해서 바꾸자.

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
        self.NumTrain = len(self.DictArray_TrainWC)
        self.NumTrain_Normal = len(self.DictArray_TrainWCNormal)
        self.NumTrain_VEB = len(self.DictArray_TrainWCPVC)



        ### Initial Value Declaration ###

        # print len(self.DictArray_TrainWC), len(self.DictArray_TrainWCNormal), len(self.DictArray_TrainWCPVC)
        self.RecordNum = RecordNum
        self.RecordType = RecordType
        self.Sampling_rate = 360
        self.SecondsToSample = Seconds * self.Sampling_rate
        self.LDAorNOT = LDAorNOT
        self.alpha = self.NumTrain_Normal / float(self.NumTrain)
        self.Str_WaveletBasis = StrWaveletBasis
        self.Int_DecompLevel = IntDecompLevel
        self.Float_Threshold = Threshold

        # COL #
        self.Int_NumTrainFeature = self.Int_NumTrainFeature
        # ROW #
        self.Int_NumTrainNormal = self.Int_NumTrainNormal

        self.LDAOFF_FisherScore, self.LDAOFF_NumSelected , self.LDAOFF_IdxFeatureSelected = self.LDAOFF_FisherScoreCompute()
        # self.LDAON_FisherScore, self.LDAON_NumSelected, self.LDAOFF_IdxFeatureSelected  = self.LDAON_FisherScoreComputation()


    def StatisticsConstruction(self):
        # Working on Test set
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

        else: # Fisher LDA
            DictFloat_Stat = dict()
            # List_FisherScoreForFeatures, Int_NumFeatureSelected, List_IdxFeatureSelected = self.LDAON_FisherScoreComputation()
            Array_ReducedMean = self.LDAON_ReducedMeanComputation() # MuW
            Array_ReducedMean = Array_ReducedMean[0]
            Matrix_ReducedCov = self.LDAON_ReducedCoVarComputation() # wTAW
            # Matrix_ReducedCov = Matrix_ReducedCov[0]

            # Key : record, Val : 1dim
            DictArray_TestWC_LDA = self.LDAON_TestWCConstruction()

            for idx, key in sorted(enumerate(DictArray_TestWC_LDA)):
                Val = DictArray_TestWC_LDA[key] # 1 Dim
                # Val = DictArray_TestWC_LDA[key][List_IdxFeatureSelected[:Int_TempNumFeatureSelected]]
                Val = np.array(Val)
                Val = np.array(Val - Array_ReducedMean)

                NewVal = Val * (Matrix_ReducedCov**(-1)) * Val.T
                DictFloat_Stat[key] = np.squeeze(np.asarray(NewVal))
            return DictFloat_Stat

    def UCL(self):
        # if self.LDAorNOT == False:
        #     P = self.LDAOFF_NumSelected
        #     N = self.Int_NumTrainNormal
        #     return (P*((N-1)**2) * f.ppf(self.alpha, P, N-P)) / (N*(N-P))
        # else:
        #     # S = self.LDAON_NumSelected
        #     # S = self.LDAOFF_NumSelected
        S = 1
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
            # VEBSVEB
            Int_TotalTestPoint += 1
            if self.Dict_TestLabel[key] == 'N' or self.Dict_TestLabel[key] == 'L' or self.Dict_TestLabel[key] == 'R' or self.Dict_TestLabel[key] == 'e' or self.Dict_TestLabel[key] == 'j' :
                if DictFloat_Stat[key] < UCLVAL: # Normal In Control
                    Int_Type1_Duzi += 1 # Normal 을 Normal 로
                elif DictFloat_Stat[key] > UCLVAL: # Normal Out Control
                    print self.Dict_TestLabel[key], DictFloat_Stat[key],  UCLVAL
                    Int_Type1_Error += 1 # Normal 을 VEB / SVEB 로
            # SVEB
            # elif self.Dict_TestLabel[key] == 'A' or self.Dict_TestLabel[key] == 'a' or self.Dict_TestLabel[key] == 'S' or self.Dict_TestLabel[key] == 'J':
            # VEB
            elif self.Dict_TestLabel[key] == 'V' or self.Dict_TestLabel[key] == 'E' :
                if DictFloat_Stat[key] < UCLVAL: # PVC In Control
                    Int_Type2_Error += 1 # VEB / SVEB 를 Normal 로
                elif DictFloat_Stat[key] > UCLVAL: # PVC Out Control
                    Int_Type2_Duzi += 1 # VEB / SVEB 를 VEB/SVEB 로
            # except:
            #     pass

        DictInt_Accuracy['Normal(G) as VEB'] = Int_Type1_Error # Normal 을 VEB / SVEB 로
        DictInt_Accuracy['Normal(G) as Normal'] = Int_Type1_Duzi # Normal 을 Normal 로
        DictInt_Accuracy['VEB(G) as Normal'] = Int_Type2_Error # VEB / SVEB 를 Normal 로
        DictInt_Accuracy['VEB(G) as VEB'] = Int_Type2_Duzi # VEB / SVEB 를 VEB/SVEB 로
        # DictInt_Accuracy['TotalBeat'] = Int_TotalTestPoint
        # DictInt_Accuracy['TotalError'] = Int_Type1_Error + Int_Type2_Error

        Int_TP = Int_Type1_Duzi # Correctly detected beat (Normal as Normal)
        Int_TN = Int_Type2_Duzi # Correctly rejected beat (PVC as PVC)
        Int_FP = Int_Type1_Error # Falsely detected (Normal as PVC)
        Int_FN = Int_Type2_Error # Misssed beat (PVC as Normal)

        # # Acc = NT - NE (=NC) / NT
        # DictFloat_Accuracy['Accuracy'] = float(Int_TotalTestPoint - (Int_Type1_Error+ Int_Type2_Error)) / float(Int_TotalTestPoint)
        # # Se = TP / TP + FN, ratio
        # DictFloat_Accuracy['Sensitivity'] = float(Int_TP) / float(Int_TP + Int_FN)
        # # Positive Predictivity = TP / TP + FP, ratio of correctly detected beat to the total num of beats
        # DictFloat_Accuracy['Positive_Predictivity'] = float(Int_TP) / float(Int_TP + Int_FP)
        # # Sp = TN / TN + FP , Correctly rejected to num of nonevent
        # DictFloat_Accuracy['Specificity'] = float(Int_Type2_Duzi) / float(Int_Type2_Duzi + Int_Type1_Error)

        return DictInt_Accuracy, DictFloat_Accuracy



    def StatPlot(self):
        DictFloat_Stat = self.StatisticsConstruction()

        if self.LDAorNOT == False:
            UCLVal = self.UCL()
            plt.figure()
            plt.grid()
            plt.xlabel("Index")
            plt.ylabel("Stat")
            plt.title("LDAOFF |" + str(self.RecordNum)+ "|" + str(self.LDAOFF_NumSelected))
            for idx, key in enumerate(sorted(DictFloat_Stat)):
                if self.Dict_TestLabel[key] == 'N' or self.Dict_TestLabel[key] == 'L' or self.Dict_TestLabel[key] == 'R' or self.Dict_TestLabel[key] == 'e' or self.Dict_TestLabel[key] == 'j' :
                    plt.plot(idx,UCLVal ,'m.')
                    plt.plot(idx, DictFloat_Stat[key], 'bo')
                elif self.Dict_TestLabel[key] == 'A' or self.Dict_TestLabel[key] == 'a' or \
                    self.Dict_TestLabel[key] == 'S' or  self.Dict_TestLabel[key] == 'V' or \
                    self.Dict_TestLabel[key] == 'E' :
                    plt.plot(idx,UCLVal ,'m.')
                    plt.plot(idx, DictFloat_Stat[key], 'ro')

            plt.show()
        else:
            #VEBSVEB
            UCLVal = self.UCL()
            plt.figure()
            plt.grid()
            plt.xlabel("Index")
            plt.ylabel("Stat")
            plt.title("LDA with Hotelling T |" + str(self.RecordNum)+"|" )
            for idx, key in enumerate(sorted(DictFloat_Stat)):
                if self.Dict_TestLabel[key] == 'N' or self.Dict_TestLabel[key] == 'L' or self.Dict_TestLabel[key] == 'R' or self.Dict_TestLabel[key] == 'e' or self.Dict_TestLabel[key] == 'j' :
                    plt.plot(idx, DictFloat_Stat[key], 'bo')
                    plt.plot(idx,UCLVal ,'m.')
                elif self.Dict_TestLabel[key] ==  'V' or self.Dict_TestLabel[key] ==  'E':
                # elif self.Dict_TestLabel[key] ==  'A' or self.Dict_TestLabel[key] ==  'a' or self.Dict_TestLabel[key] ==  'S' or self.Dict_TestLabel[key] ==  'J':
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

    # Checking VEB
    # NO VEB
    # 103 : NO VEB
    # 111 : NO VEB
    # 121 : NO VEB
    # 123 : NO VEB
    # 202 : V : 19개 중 5개만 300초 안에
    # 207 : No VEB in Test set
    # 212 : NO VEB
    # 220 : NO VEB set in Test
    # 222 : NO VEB in Training set
    # 230 : NO VEB in Training set
    # 231 : V 랑 E 없잖아 ㅡㅡ
    # 234 : Training 에 V 가 없고, Totally 4개 밖에 없다.

    # Checking SVEB
    # 103 : NO SVEB
    # 105 : NO SVEB
    # 111 : NO SVEB in Training Time
    # 121 : NO SVEB in Training Time
    # 123 : NO SVEB in Training Time
    # 203 : NO SVEB in Test
    # 208 : NO SVEB in Test
    # 212 : 노말밖에 없잖아 ㅡㅡ
    # 214 : NO SVEB in Test Trainign
    # 215 : No SVEB in Training
    # 219 : No SVEB in Training
    # 221 : No SVEB in Training
    # 222 : No SVEB in Training
    # 229 : No SVEB in Test
    # 230 : No SVEB in Training
    # 231 : No SVEB in Test







if __name__ == "__main__":
    List_RecordNum = [100, 105, 106, 108, 109, 113, 114, 116, 118, 119,
                      200, 201, 202, 203, 207, 208, 209, 210, 213, 214,
                      215, 219, 220, 221, 223, 228, 233]

    List_Test = [100, 103, 105, 111, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 233, 234]
    List_Train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 220]
    VEB = [200, 202, 210, 213, 214, 219, 221, 228, 231, 233, 234]
    SVEB = [200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

    IntRecordNum = 220
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
    ObjConstructStatistics \
        = ConstructStatistics(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold, alpha=alpha)

    NumTrain =  len(ObjConstructStatistics.DictArray_TrainWC)
    NumTrain_Normal =  len(ObjConstructStatistics.DictArray_TrainWCNormal)
    NumTrain_PVC = len(ObjConstructStatistics.DictArray_TrainWCPVC)
    alpha = NumTrain_Normal / float(NumTrain)

    ObjConstructStatistics \
        = ConstructStatistics(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold, alpha=alpha)



    # LDAON_FisherScore = ObjConstructStatistics.LDAON_FisherScore
    # LDAOFF_FisherScore = ObjConstructStatistics.LDAOFF_FisherScore
    # if BoolLDAorNOT == True:
    #     for val in LDAON_FisherScore:
    #         print val
    # else:
    #     for val in LDAOFF_FisherScore:
    #         print val
    print "My Algorithm"
    print IntRecordNum

    DictInt_Accuracy, DictFloat_Accuracy = ObjConstructStatistics.AccuracyComputation()
    for idx, key in enumerate(DictInt_Accuracy):
        if key == "Type1_Duzi":
            print "Normal(G) as Normal ", DictInt_Accuracy[key]
        elif key == "Type1_Error":
            print "Normal(G) as VEB ", DictInt_Accuracy[key]
        elif key == "Type2_Duzi":
            print "VEB(G) as VEB ", DictInt_Accuracy[key]
        elif key == "Type2_Error":
            print "VEB(G) as Normal ", DictInt_Accuracy[key]
        # key, DictInt_Accuracy[key]

    print ""
    # for idx, key in enumerate(DictFloat_Accuracy):
    #     print key, DictFloat_Accuracy[key]
    #
    # print len(ObjConstructStatistics.DictArray_TrainWCNormal)

    ObjConstructStatistics.StatPlot()






