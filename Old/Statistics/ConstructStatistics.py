# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 150512 Construct Statistics
Comment 
- 

'''

''' Library '''
from Training_Set.Construct_Training_Set import Construct_Training
from In_Control.InControl import InControl
from Compute_Fisher_Score.Applying_Fisher_Score import Fisher_Score
from Monitor_Stat. MonitorStat import MonitorStat
import matplotlib.pyplot as plt
from HansVariableSelector import VariableSelector
import numpy as np
import pandas as pd
from scipy.stats import f

''' Function or Class '''

class ConstructStatistics:
    def __init__(self, RecordNum, RecordType, Seconds, WaveletBasis, Level, NumFeature):
        self.RecordNum = RecordNum
        self.RecordType = RecordType
        self.Seconds = Seconds
        self.WaveletBasis = WaveletBasis
        self.Level = Level
        self.NumFeature = NumFeature

        ConstTrain = Construct_Training(self.RecordNum, self.RecordType, self.Seconds)
        self.WCTrainECG, self.WCTrainLabel \
            = ConstTrain.TrainDataLoad(self.WaveletBasis, self.Level)
        self.WCTestECG, self.WCTestLabel \
            = ConstTrain.TestDataLoad(self.WaveletBasis, self.Level)
        self.WCTrainNoise, self.WCTrainLabel \
            = ConstTrain.NoiseTrainDataLoad(self.WaveletBasis, self.Level)
        self.WCTestNoise, self.WCTestLabel \
            = ConstTrain.NoiseTestDataLoad(self.WaveletBasis, self.Level)

        IncObj = InControl(self.RecordNum, self.RecordType, self.Seconds, self.WaveletBasis, self.Level)
        self.AvgNum, self.InControlCoef = IncObj.AvgNormal()


        FisherObj \
            = Fisher_Score(self.RecordNum, self.RecordType, self.Seconds, self.WaveletBasis, self.Level)
        self.CoefIdx, self.CoefSelector \
            = FisherObj.Coef_Selector(NumFeature)
        self.LDATestData = FisherObj.AppltFisherLDA_to_Test()
        self.LDAOperator = FisherObj.FisherLDAOperator()
        self.LDATrain = FisherObj.ApplyFisherLDA_to_Train()

        ## LDA ##
        self.InControlCoef = np.array(np.dot(self.InControlCoef, self.LDAOperator))[0]
        ## LDA ##

        MonitorStatObj = MonitorStat(self.RecordNum, self.RecordType, self.Seconds, self.WaveletBasis, self.Level, self.NumFeature)
        self.IncontrolData = MonitorStatObj.Construct_InControl()
        self.ExtractedTestData, self.TestLabel = MonitorStatObj.Extract_Signal()

        self.Chooser = self.CoefIdx[:self.NumFeature]

    def ComputeMAD(self, List):
        Array = np.array(List, dtype='float32')
        Median = np.median(List)
        return np.median(np.abs(Array - Median))

    def MADNoiseEstimator(self):
        # Estimator for \sigma^2_M
        TrainMAD = dict()
        for idx, key in enumerate(self.WCTrainNoise):
            TrainMAD[key] = self.ComputeMAD(self.WCTrainNoise[key]) / 0.6745
        MeanMADNoise = np.mean(TrainMAD.values())
        return pd.Series(TrainMAD), MeanMADNoise * np.eye(64)

    def SigmaEstimator(self):
        FullInControl = np.mean(self.WCTrainECG, axis=0)
        MyWCTrainECG = self.WCTrainECG.transpose()

        ### LDA ###
        FullInControl = np.array(np.dot(self.LDAOperator, FullInControl))[0]
        MyWCTrainECG = pd.DataFrame.from_dict(data=self.LDATrain,orient='index').T
        ### LDA ###

        print FullInControl.shape
        print MyWCTrainECG.shape
        Result = np.zeros((len(FullInControl), len(FullInControl)))

        for idx, key in enumerate(MyWCTrainECG):
            Difference = (MyWCTrainECG[key] - FullInControl).reshape(len(FullInControl),1 )
            DifferenceTranspose = (MyWCTrainECG[key] - FullInControl).reshape(1, len(FullInControl))

            Result += Difference * DifferenceTranspose
        return pd.DataFrame(Result / (len(self.WCTrainECG))-1)

    def DiagonalEstimator(self):
        VarBox = []
        for idx, key in enumerate(self.WCTrainECG):
            VarBox.append(np.var(self.WCTrainECG[key]))

        # return self.WCTrainECG.transpose(), np.mean(self.WCTrainECG, axis=0)
        DiagonalVariance = np.eye(len(VarBox))
        for idx in range(len(VarBox)):
            DiagonalVariance[idx][idx] = VarBox[idx]
        return pd.DataFrame(DiagonalVariance)

    def VarianceEstimator(self):
        _, NoiseVariance = self.MADNoiseEstimator()
        # Sigma = self.SigmaEstimator()
        Sigma = self.DiagonalEstimator()
        ### LDA ###
        Sigma = np.dot(Sigma, self.LDAOperator)
        ### LDA ###
        Minimum_Value = 1e-8
        return np.matrix(NoiseVariance + Sigma + Minimum_Value * np.eye(len(Sigma)))

    def DimReducedVariance(self):
        Variance = self.VarianceEstimator()
        Var = VariableSelector(Variance, self.Chooser)
        return np.matrix(Var.MatrixSelector())

    def DimReducedInControl(self):
        Var = VariableSelector(self.InControlCoef, self.Chooser)
        return np.array(Var.ArraySelector())

    def StatisticsComputation(self):
        InControlMean = self.DimReducedInControl()
        InControlVar = self.DimReducedVariance().I

        #Target vector extraction
        #TestWCTranspose = self.WCTestECG.T
        TestWCTranspose = self.LDATestData.T
        TestStatBox = dict()
        for idx, key in enumerate(TestWCTranspose):
            Target = TestWCTranspose[key]
            Var = VariableSelector(Target, self.Chooser)
            Target = Var.ArraySelector()
            TestStatBox[key] = np.dot(np.dot((Target - InControlMean).reshape(1,self.NumFeature), InControlVar), (Target - InControlMean).reshape(self.NumFeature,1))

        return TestStatBox, self.WCTestLabel

    def FStat(self):
        FStatBox = dict()
        TestStat, _ = self.StatisticsComputation()
        N = float(len(self.WCTrainLabel))
        P = float(self.NumFeature)
        for idx, key in enumerate(TestStat):
            FStatBox[key] = ((N-P) * TestStat[key]) / (P*(N-1))
        return FStatBox, self.WCTestLabel

    def UCL(self, alpha):
        P = self.NumFeature
        N = len(self.WCTrainLabel)

        return (P*(N-1) * f.ppf(alpha, P, N-P)) / (N-1)

    def Accuracy(self, alpha):
        ResultDict = dict()
        ResultDict['TotalNum'] = 0
        ResultDict['RightInControl'] = 0 # Blue In Control
        ResultDict['RightOutControl'] = 0
        ResultDict['FalseNegative'] = 0 # Red In Control
        ResultDict['FalsePositive'] = 0 # Blue Out Control
        ResultDict['TotalOutControl'] = 0 # Total Number of Out control (NumAlarm)

        TestStatBox, _ = self.StatisticsComputation()
        UCL = self.UCL(alpha)

        for idx, key in enumerate(TestStatBox):
            ResultDict['TotalNum'] += 1
            if TestStatBox[key] > UCL :
                # Out Control
                ResultDict['TotalOutControl'] += 1
                if self.WCTestLabel[key] == 'N':
                    ResultDict['FalsePositive'] += 1
                elif self.WCTestLabel[key] == 'V':
                    ResultDict['RightOutControl'] += 1
            elif TestStatBox[key] <= UCL :
                # In Control
                if self.WCTestLabel[key] == 'V':
                    ResultDict['FalseNegative'] += 1
                elif self.WCTestLabel[key] == 'N':
                    ResultDict['RightInControl'] += 1
        ResultDict['FalsePositiveRate'] = (float(ResultDict['RightOutControl']) / float(ResultDict['FalsePositive'] + ResultDict['RightOutControl']))
        ResultDict['Sensitivity'] = float(ResultDict['RightInControl']) / float(ResultDict['RightInControl'] + ResultDict['FalseNegative'])
        ResultDict['Precision'] = 1- (float(ResultDict['FalsePositive']) / float(ResultDict['FalsePositive'] + ResultDict['RightInControl']))
        return ResultDict



'''
- [105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 219, 221, 223, 228, 233]
105 Not Enough V samples
106 2Min Good
116 2Min Good in record Type 1
119 1Min VERY GOOD
201 No PVC in first 10 minute, Very GOOD in Type 1
203 Type 1 Moderate
208 GOOD in both
210 Type0, 5samples
213 NoPVC in 2min (3min OK), Very Good in Type 1
215 1Min, Type 0 Good
219 NO GOOD in both
221 Good
223 Not GOOD
228 Good
233 Good
'''


if __name__ == "__main__":
    # OUT 201
    RecordList = [106, 116, 119, 203, 208, 210, 213, 215, 219, 221, 223, 228, 233]
    # RecordList = [213, 215, 219, 221, 223, 228, 233]
    RecordTypeList = [0,1]
    FisherRatio = [0.7, 0.8, 0.9]
    TrainingTime = [120, 180, 240, 300]
    alphalist = [0.995, 0.9973] # ARL = 200, 370

    RecordNum = 119
    RecordType = 0
    Seconds = 120
    fisherRatio = 0.9
    alpha = 0.9973
    Min = Seconds / 60
    Time = 30 - Min
    WaveletBasis = 'db8'
    Level = 4
    # alpha = 0.95

    HansFisher = Fisher_Score(RecordNum, RecordType, Seconds, WaveletBasis, Level)
    NumFeature =  HansFisher.NumFeature(fisherRatio)
    # NumFeature = 5

    HansStat = ConstructStatistics(RecordNum, RecordType, Seconds, WaveletBasis, Level, NumFeature)


    TestStat, TestLabel = HansStat.StatisticsComputation()
    UCL = HansStat.UCL(alpha)

    # AccuracyBox = HansStat.Accuracy(alpha)
    # print RecordNum, "|", RecordType,"|",Seconds,"|",WaveletBasis,"|",Level,"|",fisherRatio,"|",NumFeature,"|", alpha,"|",AccuracyBox['FalsePositive'],"|", AccuracyBox['FalseNegative'],"|",AccuracyBox['RightInControl'],"|",AccuracyBox['TotalOutControl'],"|",AccuracyBox["RightOutControl"],"|",AccuracyBox["TotalNum"],"|",AccuracyBox["FalsePositiveRate"],"|",AccuracyBox["Sensitivity"],"|", AccuracyBox['Precision']



    plt.figure()
    plt.grid()
    plt.xlabel("index")
    plt.ylabel("Hotelling T2 Stat")
    plt.title("Diagonal, Record : {Record}, Training : {Train}, NumFeature : {NumFeature}, Alpha : {alpha}".format(Record = RecordNum, Train = Seconds, NumFeature = NumFeature, alpha = alpha))
    for idx, key in enumerate(TestStat):
        # print key
        # print TestStat[key], TestLabel[key]
        if TestLabel[key] == "N":
            plt.plot(idx, TestStat[key].item(0), 'bo')
            plt.plot(idx, UCL, 'm.')
        elif TestLabel[key] == "V":
            plt.plot(idx, TestStat[key].item(0), 'ro')
            plt.plot(idx, UCL, 'm.')
    plt.show()
    # #





