# -*- coding: utf-8 -*-
'''
Goal :
-
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- [105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 219, 221, 223, 228, 233]
105 Not Enough V samples
106 2Min Good
116 2Min Good in record Type 0
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

''' Library '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' Function or Class '''
from Training_Set.Construct_Training_Set import Construct_Training
from Compute_Fisher_Score.HansFisherScore import Fisher_Score_Compute
from Compute_Fisher_Score.Applying_Fisher_Score import Fisher_Score
from In_Control.InControl import InControl
from scipy.stats import gaussian_kde


class MonitorStat:
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

        IncObj = InControl(self.RecordNum, self.RecordType, self.Seconds, self.WaveletBasis, self.Level)
        self.AvgNum, self.InControlCoef = IncObj.AvgNormal()

        FisherObj \
            = Fisher_Score(self.RecordNum, self.RecordType, self.Seconds, self.WaveletBasis, self.Level)
        self.CoefIdx, self.CoefSelector \
            = FisherObj.Coef_Selector(NumFeature)
        self.LDATestData = FisherObj.AppltFisherLDA_to_Test()

    def Construct_InControl(self):
        return np.asarray(self.InControlCoef[self.CoefIdx[:self.NumFeature]], dtype='float32')

    def Extract_Signal(self):
        # TestData = self.WCTestECG.T
        TestData = self.LDATestData.T
        ExtractCoef = \
            dict((key,value[self.CoefIdx[:self.NumFeature]]) for key, value in TestData.iteritems())
        return pd.DataFrame(ExtractCoef), self.WCTestLabel

    def MonitorStat(self, Sigma):
        Normalizing = float(self.AvgNum) / (float(self.AvgNum + 1) * Sigma)
        ExtractedTestData, TestLabel = self.Extract_Signal()
        InControlData = self.Construct_InControl()
        MonitorStatBox = dict()
        for key in ExtractedTestData.keys():
            result = np.sum((ExtractedTestData[key] - InControlData) ** 2)
            result *= Normalizing
            MonitorStatBox.update({key : result})
        return MonitorStatBox



if __name__ == "__main__":
    # - [105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 219, 221, 223, 228, 233]

    RecordNum = 106
    RecordType = 0
    Seconds = 120
    Min = Seconds / 60
    Time = 30 - Min
    WaveletBasis = 'db8'
    Level = 4

    HansFisher = Fisher_Score(RecordNum, RecordType, Seconds, WaveletBasis, Level)
    # NumFeature =  HansFisher.NumFeature(0.7)
    NumFeature = 1

    MonitorStatObj = \
        MonitorStat(RecordNum=RecordNum, RecordType=RecordType,
                    Seconds=Seconds, WaveletBasis=WaveletBasis,
                    Level=Level, NumFeature = NumFeature)


    TrainECG = MonitorStatObj.WCTrainECG.T
    TestECG = MonitorStatObj.LDATestData.T
    TestLabel = MonitorStatObj.WCTestLabel

    StopCond = 0
    Stop = 20
    NormalIDX = 20

    TargetData = TrainECG[TrainECG.keys()[NormalIDX]]

    for idx, key in enumerate(TestLabel):
        if TestLabel[key] == "V":
            TargetData2 = TestECG[key]
            StopCond += 1

        if StopCond == Stop:
            break

    Density_V = gaussian_kde(TargetData2)
    Domain_V = np.linspace(-max(TargetData2), max(TargetData2), 1000)
    Density = gaussian_kde(TargetData)
    Domain = np.linspace(-max(TargetData), max(TargetData), 1000)

    plt.figure()
    plt.title("V")
    plt.plot(Domain_V, Density_V(Domain_V))
    plt.grid()

    plt.figure()
    plt.title("N")
    plt.plot(Domain, Density(Domain))
    plt.grid()

    plt.show()





    #
    #
    # ExtractedTestData, TestLabel = MonitorStatObj.Extract_Signal()
    # InControlData = MonitorStatObj.Construct_InControl()
    #
    # Result =  MonitorStatObj.MonitorStat(1)
    # TimeDomain = np.linspace(0, Time, num=len(Result))
    # # print Result.keys()
    # #
    # # Result = dict()
    # # for key in ExtractedTestData.keys():
    # #     result = np.sum((ExtractedTestData[key] - InControlData)**2)
    # #     Result.update({key: result})
    # # Label = ['N','V']
    # # ColorMarker = ['bo','ro']
    # #
    # plt.figure()
    # plt.title("Record : {0} // RecordType : {1} // NumFeatures : {2} // Training : {3}".format(RecordNum,RecordType, NumFeature, Seconds ))
    # plt.grid()
    # for idx, key in enumerate(Result):
    #     if TestLabel[key] == 'N':
    #         plt.plot(TimeDomain[idx], Result[key],'bo' )
    #     elif TestLabel[key] == 'V':
    #         plt.plot(TimeDomain[idx],Result[key], 'ro')
    # plt.xlabel("Minute (Test)")
    # plt.ylabel("Square Sum of selected coefficients")
    # plt.show()

