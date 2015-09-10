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
    def __init__(self, RecordNum , RecordType, Seconds, StrWaveletBasis, IntDecompLevel, LDAorNOT, Threshold):

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
        # self.alpha = self.NumTrain_Normal / float(self.NumTrain)
        self.Str_WaveletBasis = StrWaveletBasis
        self.Int_DecompLevel = IntDecompLevel
        self.Float_Threshold = Threshold

        # COL #
        self.Int_NumTrainFeature = self.Int_NumTrainFeature
        # ROW #
        self.Int_NumTrainNormal = self.Int_NumTrainNormal

        self.LDAOFF_FisherScore, self.LDAOFF_NumSelected , self.LDAOFF_IdxFeatureSelected = self.LDAOFF_FisherScoreCompute()
        # self.LDAON_FisherScore, self.LDAON_NumSelected, self.LDAOFF_IdxFeatureSelected  = self.LDAON_FisherScoreComputation()

    # CUSUM Threshold 를 어떻게 계산하느냐는 문제
    def CUSUM_PValue(self, K):
        from CUSUM_HotellingT import CUSUM_HotellingT
        Obj = CUSUM_HotellingT(K)
        Alpha_VEB, Threshold = self.Compute_PValue()
        MyPValue = Obj.Compute_PValue(Threshold)
        # Threshold = Obj.Compute_AlphaThreshold(90)
        # Normal 일 때, statistics, p-value, CDF 를 계산한다.
        # Abnormal 일 때 statistics, p-value, CDF 를 계산한다.

        Array_ReducedMean = self.LDAON_ReducedMeanComputation() # MuW
        Array_ReducedMean = Array_ReducedMean[0]
        Matrix_ReducedCov = self.LDAON_ReducedCoVarComputation() # wTAW
        # Matrix_ReducedCov = Matrix_ReducedCov[0]

        # Key : record, Val : 1dim
        DictArray_TrainWC_LDA = self.LDAON_TrainWCConstruction()
        List_EachStat = list()
        Dict_KeyIdx_ValPval = dict()

        for idx, key in enumerate(sorted(DictArray_TrainWC_LDA)):
            Val = DictArray_TrainWC_LDA[key] # 1 Dim
            # Val = DictArray_TestWC_LDA[key][List_IdxFeatureSelected[:Int_TempNumFeatureSelected]]
            Val = np.array(Val)
            Val = np.array(Val - Array_ReducedMean)

            NewVal = Val * (Matrix_ReducedCov**(-1)) * Val.T # Log likelihood statistics
            Stat = np.squeeze(np.asarray(NewVal))
            List_EachStat.append(Stat)
            CusumStat = np.sum(List_EachStat[-K:])

            Pvalue = Obj.Compute_PValue(CusumStat)
            Pvalue = float(Pvalue)

            # print key, Stat, CusumStat, Pvalue, Threshold, MyPValue, self.Dict_TrainLabel[key]
            Dict_KeyIdx_ValPval[key] = Pvalue

        return Dict_KeyIdx_ValPval

    def Construct_CUSUMStat_Test(self, K):
        DictFloat_Stat = dict()
        Alpha_VEB, Threshold = self.Compute_PValue()
        # List_FisherScoreForFeatures, Int_NumFeatureSelected, List_IdxFeatureSelected = self.LDAON_FisherScoreComputation()
        Array_ReducedMean = self.LDAON_ReducedMeanComputation() # MuW
        Array_ReducedMean = Array_ReducedMean[0]
        Matrix_ReducedCov = self.LDAON_ReducedCoVarComputation() # wTAW
        # Matrix_ReducedCov = Matrix_ReducedCov[0]

        # Key : record, Val : 1dim
        DictArray_TestWC_LDA = self.LDAON_TestWCConstruction()
        List_EachStat = list()

        for idx, key in sorted(enumerate(DictArray_TestWC_LDA)):
            Val = DictArray_TestWC_LDA[key] # 1 Dim
            # Val = DictArray_TestWC_LDA[key][List_IdxFeatureSelected[:Int_TempNumFeatureSelected]]
            Val = np.array(Val)
            Val = np.array(Val - Array_ReducedMean)

            NewVal = Val * (Matrix_ReducedCov**(-1)) * Val.T # Log likelihood statistics
            Stat = np.squeeze(np.asarray(NewVal))
            List_EachStat.append(Stat)
            CusumStat = np.sum(List_EachStat[-K:])
            # print "Stat", self.Dict_TestLabel[key], NewVal, "CUSUM Stat", CusumStat
            DictFloat_Stat[key] = CusumStat
            if CusumStat > Threshold:
                List_EachStat = list()
        return DictFloat_Stat

    def Compute_PValue(self):
        # Only assume LDA Applying
        # 1. Training 에서 Abnormal 만 모은다.
        # 1.5 Training Abnormal sample 에 Fisher LDA 를 쓴다.
        # 2. Stat 을 계산한다.
        # 3. F-statistics 로 바꾼다.
        # 4. 여기서 가장 높은 p-value 를 주는 statistics 를 바탕으로 control limit 을 구성한다.


        # 1 ~ 1.5 Fisher LDA to Abnormal
        DictFloat_VEB = self.LDAON_TrainWCPVCConstruction()
        DictFloat_Normal = self.LDAON_TrainWCNormalConstruction()

        # 필요한 parameter 들
        Array_ReducedMean = self.LDAON_ReducedMeanComputation() # MuW
        Array_ReducedMean = Array_ReducedMean[0]
        Matrix_ReducedCov = self.LDAON_ReducedCoVarComputation() # wTAW

        # Hotelling T to F distribution 을 위한 constant
        # T = CF
        p = 1.0
        N = float(self.Int_NumTrainNormal)
        C = (p*((N-1) ** 2)) / (N * (N-p))

        # P-value List
        # ValueStat

        List_PValue_VEB = list()
        List_Stat_VEB = list()
        List_PValue_Normal = list()
        List_Stat_Normal = list()

        for idx, key in sorted(enumerate(DictFloat_VEB)):
            Val = DictFloat_VEB[key] # 1 Dim
            # Val = DictArray_TestWC_LDA[key][List_IdxFeatureSelected[:Int_TempNumFeatureSelected]]
            Val = np.array(Val)
            Val = np.array(Val - Array_ReducedMean)

            # 2. Statistics 를 계산한다.
            NewVal = Val * (Matrix_ReducedCov**(-1)) * Val.T
            Stat = np.squeeze(np.asarray(NewVal))
            Stat /= C # 이건 F_statistics

            PValue = 1 - f.cdf(Stat, 1, N)
            # print "P value", PValue * 100, "%", "Stat", Stat

            List_Stat_VEB.append(Stat)
            List_PValue_VEB.append(PValue)

        for idx, key in sorted(enumerate(DictFloat_Normal)):
            Val = DictFloat_Normal[key] # 1 Dim
            # Val = DictArray_TestWC_LDA[key][List_IdxFeatureSelected[:Int_TempNumFeatureSelected]]
            Val = np.array(Val)
            Val = np.array(Val - Array_ReducedMean)

            # 2. Statistics 를 계산한다.
            NewVal = Val * (Matrix_ReducedCov**(-1)) * Val.T
            Stat = np.squeeze(np.asarray(NewVal))
            Stat /= C # 이건 F_statistics

            PValue = 1 - f.cdf(Stat, 1, N)
            # print "P value", PValue * 100, "%", "Stat", Stat

            List_Stat_Normal.append(Stat)
            List_PValue_Normal.append(PValue)

        CUTQuantile_VEB = 0
        # print C * np.percentile(List_Stat, CUTQuantile) , np.percentile(List_Stat, CUTQuantile), f.cdf(np.percentile(List_Stat, CUTQuantile),1,N)
        Alpha_VEB = f.cdf(np.percentile(List_Stat_VEB, CUTQuantile_VEB),1,N)
        PValue = 1-Alpha_VEB
        Threshold = C * np.percentile(List_Stat_VEB, CUTQuantile_VEB)

        print "Normal Max Quantile", f.cdf(np.percentile(List_Stat_Normal, 100), 1, N)
        print "Pvalue Normal", 1 - f.cdf(np.percentile(List_Stat_Normal, 100), 1, N)

        print "Alpha", Alpha_VEB
        print "PValue VEB", PValue

        return Alpha_VEB, Threshold




    # Schewartz statistics
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
                Stat = np.squeeze(np.asarray(NewVal))
                # P-value 계산해서 집어넣기. #
                # print "Stat", self.Dict_TestLabel[key], NewVal
                DictFloat_Stat[key] = Stat
            return DictFloat_Stat

    def UCL(self):
        # if self.LDAorNOT == False:
        #     P = self.LDAOFF_NumSelected
        #     N = self.Int_NumTrainNormal
        #     return (P*((N-1)**2) * f.ppf(self.alpha, P, N-P)) / (N*(N-P))
        # else:
        #     # S = self.LDAON_NumSelected
        #     # S = self.LDAOFF_NumSelected
        # S = self.LDAOFF_NumSelected
        Alpha, Threshold = self.Compute_PValue()
        # S = 1
        # N = self.Int_NumTrainNormal
        # print "Total Num", N
        # Alpha = self.Compute_PValue()
        return Threshold
        # return (S*((N-1)**2) * f.ppf(Alpha, S, N-S)) / (N*(N-S))
        # return (S*((N-1)**2) * f.ppf(self.alpha, S, N-S)) / (N*(N-S))

    def Compute_Accuracy_CUSUM(self, K):
        DictFloat_CUSUMStat = self.Construct_CUSUMStat_Test(K)
        Alpha_VEB, Threshold = self.Compute_PValue()
        DictInt_Accuracy = dict()

        Int_TotalTestPoint = 0
        Int_Type1_Error = 0
        Int_Type1_Duzi = 0
        Int_Type2_Error = 0
        Int_Type2_Duzi = 0

        for idx, key in enumerate(sorted(DictFloat_CUSUMStat)):
            # VEBSVEB
            Int_TotalTestPoint += 1
            if self.Dict_TestLabel[key] == 'N' or self.Dict_TestLabel[key] == 'L' or self.Dict_TestLabel[key] == 'R' or self.Dict_TestLabel[key] == 'e' or self.Dict_TestLabel[key] == 'j' :
                if DictFloat_CUSUMStat[key] < Threshold: # Normal In Control
                    Int_Type1_Duzi += 1 # Normal 을 Normal 로
                elif DictFloat_CUSUMStat[key] > Threshold: # Normal Out Control
                    # print self.Dict_TestLabel[key], DictFloat_Stat[key],  UCLVAL
                    Int_Type1_Error += 1 # Normal 을 VEB / SVEB 로
            # SVEB
            # elif self.Dict_TestLabel[key] == 'A' or self.Dict_TestLabel[key] == 'a' or self.Dict_TestLabel[key] == 'S' or self.Dict_TestLabel[key] == 'J':
            # VEB
            elif self.Dict_TestLabel[key] == 'V' or self.Dict_TestLabel[key] == 'E' :
                if DictFloat_CUSUMStat[key] < Threshold: # PVC In Control
                    Int_Type2_Error += 1 # VEB / SVEB 를 Normal 로
                elif DictFloat_CUSUMStat[key] > Threshold: # PVC Out Control
                    Int_Type2_Duzi += 1 # VEB / SVEB 를 VEB/SVEB 로
            # except:
            #     pass

        DictInt_Accuracy['Normal(G) as VEB'] = Int_Type1_Error # Normal 을 VEB / SVEB 로
        DictInt_Accuracy['Normal(G) as Normal'] = Int_Type1_Duzi # Normal 을 Normal 로
        DictInt_Accuracy['VEB(G) as Normal'] = Int_Type2_Error # VEB / SVEB 를 Normal 로
        DictInt_Accuracy['VEB(G) as VEB'] = Int_Type2_Duzi # VEB / SVEB 를 VEB/SVEB 로

        return DictInt_Accuracy




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
                    # print self.Dict_TestLabel[key], DictFloat_Stat[key],  UCLVAL
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

        DictInt_Accuracy['Normal(G) as VEB'] = Int_Type1_Error # Normal 을 VEB / SVEB 로 (FP)
        DictInt_Accuracy['Normal(G) as Normal'] = Int_Type1_Duzi # Normal 을 Normal 로 (TN)
        DictInt_Accuracy['VEB(G) as Normal'] = Int_Type2_Error # VEB / SVEB 를 Normal 로 (FN)
        DictInt_Accuracy['VEB(G) as VEB'] = Int_Type2_Duzi # VEB / SVEB 를 VEB/SVEB 로 (TP)
        # DictInt_Accuracy['TotalBeat'] = Int_TotalTestPoint
        # DictInt_Accuracy['TotalError'] = Int_Type1_Error + Int_Type2_Error

        Int_TP = Int_Type2_Duzi # Correctly detected beat (Normal as Normal)
        Int_TN = Int_Type1_Duzi # Correctly rejected beat (PVC as PVC)
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

    IntRecordNum = 100
    IntRecordType = 0
    IntSeconds = 300

    BoolLDAorNOT = True
    # BoolLDAorNOT = False

    if BoolLDAorNOT == True:
        FltThreshold = 0.99
    else:
        FltThreshold = 0.8
    IntDecompLevel = 4
    Threshold = 11

    StrWaveletBasis = 'db8'

    # alpha = 0.975
    ObjConstructStatistics \
        = ConstructStatistics(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold)

    NumTrain =  len(ObjConstructStatistics.DictArray_TrainWC)
    NumTrain_Normal =  len(ObjConstructStatistics.DictArray_TrainWCNormal)
    NumTrain_PVC = len(ObjConstructStatistics.DictArray_TrainWCPVC)




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
    # DictFloat_CUSUM = ObjConstructStatistics.Construct_CUSUMStat(K=K, Threshold=Threshold)
    ObjConstructStatistics.Compute_PValue()
    print ""
    print ""
    # print ObjConstructStatistics.CUSUM_PValue(5)
    # ObjConstructStatistics.StatisticsConstruction()
    K = 5
    Dict_CUSUM_Accuracy = ObjConstructStatistics.Compute_Accuracy_CUSUM(K=K)
    # print "-" * 50
    # print "CUSUM"
    # for idx, key in enumerate(sorted(Dict_CUSUM_Accuracy)):
    #     if key == "Normal(G) as Normal":
    #         print "Normal(G) as Normal", Dict_CUSUM_Accuracy[key]
    #     elif key == "Normal(G) as VEB":
    #         print "Normal(G) as VEB", Dict_CUSUM_Accuracy[key]
    #     elif key == "VEB(G) as VEB":
    #         print "VEB(G) as VEB", Dict_CUSUM_Accuracy[key]
    #     elif key == "VEB(G) as Normal":
    #         print "VEB(G) as Normal", Dict_CUSUM_Accuracy[key]
    #
    # print ""
    # print "=" * 50
    # print ""

    Dict_Accuracy, _ = ObjConstructStatistics.AccuracyComputation()
    print "-" * 50
    print "Hotelling T"

    for idx, key in enumerate(sorted(Dict_Accuracy)):
        if key == "Normal(G) as Normal":
            print "Normal(G) as Normal", Dict_Accuracy[key] # TN
            TN = Dict_Accuracy[key]
        elif key == "Normal(G) as VEB":
            print "Normal(G) as VEB", Dict_Accuracy[key] # FP
            FP = Dict_Accuracy[key]
        elif key == "VEB(G) as VEB":
            print "VEB(G) as VEB", Dict_Accuracy[key] # TP
            TP = Dict_Accuracy[key]
        elif key == "VEB(G) as Normal":
            print "VEB(G) as Normal", Dict_Accuracy[key] # FN
            FN = Dict_Accuracy[key]

    print "-" * 50
    print "Sensitivity", TP / float(TP + FN)
    print "Specificity", TN / (float(TN + FP))
    print "Positive Predictivity", TP / float(TP + FP)
    print "False Positive Rate", FP / float(TN + FP)
    print "Classification rate", (TP + TN) / float(TN + TP + FN + FP)



    # print "-" * 50
    # print "CUSUM"
    # for idx, key in enumerate(sorted(Dict_CUSUM_Accuracy)):
    #     if key == "Normal(G) as Normal":
    #         print "Normal(G) as Normal", Dict_CUSUM_Accuracy[key]
    #     elif key == "Normal(G) as VEB":
    #         print "Normal(G) as VEB", Dict_CUSUM_Accuracy[key]
    #     elif key == "VEB(G) as VEB":
    #         print "VEB(G) as VEB", Dict_CUSUM_Accuracy[key]
    #     elif key == "VEB(G) as Normal":
    #         print "VEB(G) as Normal", Dict_CUSUM_Accuracy[key]
    #
    # print ""
    # print "=" * 50
    # print ""


    # DictInt_Accuracy, DictFloat_Accuracy = ObjConstructStatistics.AccuracyComputation()
    # for idx, key in enumerate(DictInt_Accuracy):
    #     if key == "Normal(G) as Normal":
    #         print "Normal(G) as Normal", DictInt_Accuracy[key]
    #     elif key == "Normal(G) as VEB":
    #         print "Normal(G) as VEB", DictInt_Accuracy[key]
    #     elif key == "VEB(G) as VEB":
    #         print "VEB(G) as VEB", DictInt_Accuracy[key]
    #     elif key == "VEB(G) as Normal":
    #         print "VEB(G) as Normal", DictInt_Accuracy[key]
    #     key, DictInt_Accuracy[key]
    #
    # print ""
    # for idx, key in enumerate(DictFloat_Accuracy):
    #     print key, DictFloat_Accuracy[key]
    #
    # print len(ObjConstructStatistics.DictArray_TrainWCNormal)

    # ObjConstructStatistics.StatPlot()






