# -*- coding: utf-8 -*-
'''
Goal : Compute Fisher Score
Author : Yonghan Jung, IE, KAIST 
Date : 150501
Comment 
- Compute the fisher score for
Possible records are =
[105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 217, 219, 221, 223, 228, 233]
106 No PVC in 1min
201 : No PVC in first 10 min
'''

''' Library '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Training_Set.Construct_Training_Set import Construct_Training
import HansFisherScore
''' Function or Class '''


class Fisher_Score:
    def __init__(self, RecordNum, RecordType, Seconds, WaveletBasis, Level):
        self.RecordNum = RecordNum
        self.RecordType = RecordType
        self.Seconds = Seconds
        self.WaveletBasis = WaveletBasis
        self.Level = Level

        # ConstTrain = Construct_Training.__init__(self, self.RecordNum, self.RecordType, self.Seconds)
        ConstTrain = Construct_Training(self.RecordNum, self.RecordType, self.Seconds)
        self.WCTrainECG, self.WCTrainLabel \
            = ConstTrain.TrainDataLoad(self.WaveletBasis, self.Level)

    def WCTrain_to_ListMatrix(self):
        Train_Dict = {}
        Train_Dict['Normal'] = []
        Train_Dict['PVC'] = []

        RowIterable_WC = self.WCTrainECG.T
        for row_idx in RowIterable_WC:
            if self.WCTrainLabel[row_idx] == "N" :
                Train_Dict['Normal'].append(RowIterable_WC[row_idx])
            elif self.WCTrainLabel[row_idx] == "V" :
                Train_Dict['PVC'].append(RowIterable_WC[row_idx])
        return Train_Dict

    def Fisher_Score_Vector(self):
        TrainDict = self.WCTrain_to_ListMatrix()
        Fisher_Object = HansFisherScore.Fisher_Score_Compute(TrainDict)
        # return Fisher_Object.Fisher_Score()
        return Fisher_Object.Fisher_Score()

    def Coef_Selector(self, Num):
        TrainDict = self.WCTrain_to_ListMatrix()
        Fisher_Object = HansFisherScore.Fisher_Score_Compute(TrainDict)
        Fisher_Score = self.Fisher_Score_Vector()
        return Fisher_Object.FeatureSelector(Num=Num)

    def NumFeature(self, Threshold):
        TrainDict = self.WCTrain_to_ListMatrix()
        return HansFisherScore.Fisher_Score_Compute(TrainDict).HowMany(Threshold=Threshold)


    def Plot(self):
        RowIterable = self.WCTrainECG.T
        for row_idx in RowIterable:
            if self.WCTrainLabel[row_idx] == "N":
                plt.plot(RowIterable[row_idx], 'bo')
            elif self.WCTrainLabel[row_idx] == "V":
                plt.plot(RowIterable[row_idx], 'ro')
        plt.grid()
        plt.show()


if __name__ == "__main__":
    Wavelet_Basis = 'db8'
    Fisher = \
        Fisher_Score(RecordNum=210, RecordType=0, Seconds=120, WaveletBasis=Wavelet_Basis, Level=4)
    FisherVector = Fisher.Fisher_Score_Vector()
    IdxRank, Fisher_Selector = Fisher.Coef_Selector(10)
    for idx in range(len(FisherVector)):
        print idx, FisherVector[idx]
    print ""
    print pd.DataFrame(Fisher_Selector)
    print IdxRank
    Fisher.Plot()




