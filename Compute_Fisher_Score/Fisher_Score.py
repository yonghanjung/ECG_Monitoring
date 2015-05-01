# -*- coding: utf-8 -*-
'''
Goal : Compute Fisher Score
Author : Yonghan Jung, IE, KAIST 
Date : 150501
Comment 
- Compute the fisher score for
'''

''' Library '''
import pandas as pd
import numpy as np
from Training_Set.Construct_Training_Set import Construct_Training
from Fisher_Score_Computation import Fisher_Score_Compute
''' Function or Class '''


class Fisher_Score:
    def __init__(self, RecordNum, RecordType, Seconds, Level):
        self.RecordNum = RecordNum
        self.RecordType = RecordType
        self.Seconds = Seconds
        self.Level = Level

        # ConstTrain = Construct_Training.__init__(self, self.RecordNum, self.RecordType, self.Seconds)
        ConstTrain = Construct_Training(self.RecordNum, self.RecordType, self.Seconds)
        self.WCTrainECG, self.WCTrainLabel \
            = ConstTrain.TrainDataLoad(self.Level)

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
        Fisher_Object = Fisher_Score_Compute(TrainDict)
        # return Fisher_Object.Fisher_Score()
        return Fisher_Object.Fisher_Score()



if __name__ == "__main__":
    Fisher = Fisher_Score(RecordNum=119, RecordType=0, Seconds=300, Level=4)
    FisherVector = Fisher.Fisher_Score_Vector()
    for idx in range(len(FisherVector)):
        print idx, FisherVector[idx]




