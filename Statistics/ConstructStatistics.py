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
        # print FullInControl.shape
        # print MyWCTrainECG.shape
        Result = np.zeros((len(FullInControl), len(FullInControl)))

        for idx, key in enumerate(MyWCTrainECG):
            Difference = (MyWCTrainECG[key] - FullInControl).reshape(len(FullInControl),1 )
            DifferenceTranspose = (MyWCTrainECG[key] - FullInControl).reshape(1, len(FullInControl))

            Result += Difference * DifferenceTranspose
        return pd.DataFrame(Result / (len(self.WCTrainECG))-1)

    def VarianceEstimator(self):
        _, NoiseVariance = self.MADNoiseEstimator()
        Sigma = self.SigmaEstimator()
        Minimum_Value = 1e-4
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
        TestWCTranspose = self.WCTestECG.T
        TestStatBox = dict()
        for idx, key in enumerate(TestWCTranspose):
            Target = TestWCTranspose[key]
            Var = VariableSelector(Target, self.Chooser)
            Target = Var.ArraySelector()
            TestStatBox[key] = np.dot(np.dot((Target - InControlMean).reshape(1,self.NumFeature), InControlVar), (Target - InControlMean).reshape(self.NumFeature,1))

        return TestStatBox, self.WCTestLabel







if __name__ == "__main__":
    # - [105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 219, 221, 223, 228, 233]

    RecordNum = 203
    RecordType = 0
    Seconds = 120
    Min = Seconds / 60
    Time = 30 - Min
    WaveletBasis = 'db8'
    Level = 4

    HansFisher = Fisher_Score(RecordNum, RecordType, Seconds, WaveletBasis, Level)
    NumFeature =  HansFisher.NumFeature(0.8)

    HansStat = ConstructStatistics(RecordNum, RecordType, Seconds, WaveletBasis, Level, NumFeature)
    TestStat, TestLabel = HansStat.StatisticsComputation()

    # for idx, key in enumerate(TestStat):
    #     print key, TestStat[key].item(0), TestLabel[key]

    plt.figure()
    plt.grid()
    plt.xlabel("index")
    plt.ylabel("Hotelling T2 Stat")
    plt.title("Record : {Record}, Training : {Train}, NumFeature : {NumFeature}".format(Record = RecordNum, Train = Seconds, NumFeature = NumFeature))
    for idx, key in enumerate(TestStat):
        # print key
        # print TestStat[key], TestLabel[key]
        if TestLabel[key] == "N":
            plt.plot(idx, TestStat[key].item(0), 'bo')
        elif TestLabel[key] == "V":
            plt.plot(idx, TestStat[key].item(0), 'ro')
    plt.show()


    # Target = HansStat.WCTestECG.T
    # for idx, key in enumerate(Target):
    #     NewTarget = Target[key]
    #     Label =  HansStat.WCTestLabel[key]
    #     print "TestLabel", Label
    #     # print NewTarget.shape
    #     if Label == "V":
    #         break
    # # print NewTarget
    # Var = VariableSelector(NewTarget, HansStat.Chooser)
    # Target = Var.ArraySelector()
    #
    # InControlMean = HansStat.DimReducedInControl()
    # InControlVar = HansStat.DimReducedVariance().I
    # # print InControlVar
    # print pd.DataFrame(np.dot((Target - InControlMean).reshape(1,NumFeature), InControlVar))
    # print "Chooser!", HansStat.Chooser
    # print Target
    # print InControlMean
    # # print Target - InControlMean
    # print np.dot(np.dot((Target - InControlMean).reshape(1,NumFeature), InControlVar), (Target - InControlMean).reshape(NumFeature,1))




