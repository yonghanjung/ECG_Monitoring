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
from FisherLDA import FisherLDA
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
        self.WCTestECG, self.WCTestLabel \
            = ConstTrain.TestDataLoad(self.WaveletBasis, self.Level)

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

    def FisherLDAOperator(self):
        TempDict = self.WCTrain_to_ListMatrix()
        TrainDict = dict()

        TrainDict[0] = TempDict['Normal']
        TrainDict[1] = TempDict['PVC']
        Dim = len(TrainDict[0][0])

        FisherLDAObj = FisherLDA(TrainingData=TrainDict, Num=Dim)
        W = FisherLDAObj.LDAOperator()
        return W

    def ApplyFisherLDA_to_Train(self):
        W = self.FisherLDAOperator()
        TrainDict = self.WCTrain_to_ListMatrix()

        for Idx, key in enumerate(TrainDict):
            EachClass = TrainDict[key] # List
            for IdxClass, ValClass in enumerate(EachClass):
                ValClass = np.array(ValClass)
                ValClass = ValClass.reshape(len(ValClass),1)
                ValClass = np.array(np.dot(W, ValClass))[0]
                TrainDict[key][IdxClass] = ValClass
        return TrainDict

    def AppltFisherLDA_to_Test(self):
        W = self.FisherLDAOperator()
        NewTestDict = dict()
        for idx, key in enumerate(self.WCTestECG.T):
            testval = self.WCTestECG.T[key]
            testval = testval.reshape(len(testval) * 1)
            testval = np.array(np.dot(W, testval))

            NewTestDict[key] = testval[0]

        return pd.DataFrame.from_dict(data=NewTestDict,orient='index')
        # return pd.DataFrame.from_dict(data=NewTestDict, orient='index')


    def Fisher_Score_Vector(self):
        TempDict = self.WCTrain_to_ListMatrix()
        # print type(TempDict['Normal']), len(TempDict['Normal']), len(TempDict['Normal'][0])
        TrainDict = self.ApplyFisherLDA_to_Train()
        # print type(TrainDict['Normal']), len(TrainDict['Normal']), len(TrainDict['Normal'][0])
        Fisher_Object = HansFisherScore.Fisher_Score_Compute(TrainDict)
        # return Fisher_Object.Fisher_Score()
        return Fisher_Object.Fisher_Score()

    def Coef_Selector(self, Num):
        #TrainDict = self.WCTrain_to_ListMatrix()
        TrainDict = self.ApplyFisherLDA_to_Train()
        Fisher_Object = HansFisherScore.Fisher_Score_Compute(TrainDict)
        Fisher_Score = self.Fisher_Score_Vector()
        return Fisher_Object.FeatureSelector(Num=Num)

    def NumFeature(self, Threshold):
        #TrainDict = self.WCTrain_to_ListMatrix()
        TrainDict = self.ApplyFisherLDA_to_Train()
        return HansFisherScore.Fisher_Score_Compute(TrainDict).HowMany(Threshold=Threshold)


    def Plot(self):
        plt.figure()
        plt.grid()
        plt.title("TrainData")
        RowIterable = self.WCTrainECG.T
        for row_idx in RowIterable:
            # print RowIterable[row_idx]
            if self.WCTrainLabel[row_idx] == "N":
                plt.plot(RowIterable[row_idx], 'bo')
            elif self.WCTrainLabel[row_idx] == "V":
                plt.plot(RowIterable[row_idx], 'ro')

        # plt.show()

    def TestLDAPlot(self):
        plt.figure()
        plt.grid()
        plt.title("TestData after LDA")
        RowIterable = self.AppltFisherLDA_to_Test().T
        for row_idx in RowIterable:
            # print RowIterable[row_idx]
            if self.WCTestLabel[row_idx] == "N":
                plt.plot(RowIterable[row_idx], 'bo')
            elif self.WCTestLabel[row_idx] == "V":
                plt.plot(RowIterable[row_idx], 'ro')


        # plt.show()

    def TestPlot(self):
        plt.figure()
        plt.grid()
        plt.title("TestData before LDA")
        RowIterable = self.WCTestECG.T
        for row_idx in RowIterable:
            # print RowIterable[row_idx]
            if self.WCTestLabel[row_idx] == "N":
                plt.plot(RowIterable[row_idx], 'bo')
            elif self.WCTestLabel[row_idx] == "V":
                plt.plot(RowIterable[row_idx], 'ro')

        # plt.show()



if __name__ == "__main__":
    Wavelet_Basis = 'db8'
    Fisher = \
        Fisher_Score(RecordNum=119, RecordType=0, Seconds=120, WaveletBasis=Wavelet_Basis, Level=4)
    FisherVector = Fisher.Fisher_Score_Vector()
    IdxRank, Fisher_Selector = Fisher.Coef_Selector(10)
    # for idx in range(len(FisherVector)):
    #     print idx, FisherVector[idx]
    # print ""
    # print pd.DataFrame(Fisher_Selector)
    # print IdxRank

    ConstructW = dict()
    ConstructW[0] = list()
    ConstructW[1] = list()

    for idx, key in enumerate(Fisher.WCTrainECG.T):
        if Fisher.WCTrainLabel[key] == 'N':
            ConstructW[0].append(np.array(Fisher.WCTrainECG.T[key]))
        elif Fisher.WCTrainLabel[key] == 'V':
            ConstructW[1].append(np.array(Fisher.WCTrainECG.T[key]))

    ConstructW[0] = np.array(ConstructW[0])
    ConstructW[1] = np.array(ConstructW[1])

    Class1 = ConstructW[0].T
    Mu1 = np.mean(Class1, axis=1)
    MyLDA = FisherLDA(TrainingData=ConstructW, Num=64)

    # print Fisher.AppltFisherLDA_to_Test()
    # print Fisher.WCTestECG

    Fisher.Plot()
    Fisher.TestPlot()
    Fisher.TestLDAPlot()

    plt.show()



