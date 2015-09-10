# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
import numpy as np
## Neural Network ##
from pybrain.datasets            import ClassificationDataSet
from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure           import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import GaussianLayer
####
from FeatureSelector3 import FeatureSelector
from NeuralNetwork_RBF import RBF_NN

''' Function or Class '''


class Competitive_NN(FeatureSelector):
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

    def Result(self):
        Dict_KeyRecord_ValWX_TrainWCNormal = self.LDAON_TrainWCNormalConstruction()
        Dict_KeyRecord_ValWX_TrainWCPVC = self.LDAON_TrainWCPVCConstruction()
        Dict_KeyRecord_ValWX_Test = self.LDAON_TestWCConstruction()
        Dict_KeyRecord_ValLabel_Test = self.Dict_TestLabel

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
            # elif Dict_KeyRecord_ValLabel_Test[key] == "V" or Dict_KeyRecord_ValLabel_Test[key] == "E":
            elif Dict_KeyRecord_ValLabel_Test[key] == "S" or Dict_KeyRecord_ValLabel_Test[key] == "A" or Dict_KeyRecord_ValLabel_Test[key] == "J" or Dict_KeyRecord_ValLabel_Test[key] == "A":
                # print Dict_KeyRecord_ValLabel_Test[key]
                TrueAnswer.append(1)

        ArrayMat_Train= np.array(ArrayMat_Train)
        ArrayMat_Train = np.reshape(ArrayMat_Train, (len(ArrayMat_Train),1))
        ArrayMat_Test = np.array(ArrayMat_Test)
        ArrayMat_Test = np.reshape(ArrayMat_Test, (len(ArrayMat_Test),1))
        Y = np.array(Y)




        # 여기서부터 Neural Network 적용
        TrainNum, Dim = ArrayMat_Train.shape
        NNData = ClassificationDataSet(Dim, 1)


        for idx in range(len(Y)):
            NNData.addSample(np.ravel(ArrayMat_Train[idx]), Y[idx])
        NNData._convertToOneOfMany()
        # NNData._convertToOneOfMany()
        # HiddenNum = int(len(ArrayMat_Train)/ float(2 * (NNData.indim + NNData.outdim)))
        HiddenNum = Dim
        print "Indim", NNData.indim
        # print NNData.outdim
        # print HiddenNum
        NNNetwork = buildNetwork(NNData.indim, HiddenNum, NNData.outdim, outclass = GaussianLayer)
        # NNNetwork = buildNetwork(NNData.indim, HiddenNum, NNData.outdim, hiddenclass=TanhLayer)
        Trainer = BackpropTrainer(NNNetwork, dataset=NNData, verbose=False)

        NNTest = ClassificationDataSet(Dim, 1)

        for idx in range(len(TrueAnswer)):
            # print np.ravel(ArrayMat_Test[idx]), TrueAnswer[idx]
            NNTest.addSample(np.ravel(ArrayMat_Test[idx]), TrueAnswer[idx])
        NNTest._convertToOneOfMany()
        trainer = BackpropTrainer( NNNetwork, dataset=NNData)
        trainer.trainUntilConvergence( verbose = True, maxEpochs = 100)
        MyAnswer = trainer.testOnClassData(dataset=NNTest)

        NormalAsNormal = 0
        NormalAsVEB = 0
        VEBASVEB = 0
        VEBASNormal = 0

        for a,b in zip(MyAnswer,TrueAnswer):
            if b ==0 and a == 0:
                NormalAsNormal+= 1
            elif b == 0 and a == 1:
                NormalAsVEB += 1
            elif a == 0 and b == 1:
                VEBASNormal += 1
            elif a == 1 and b == 1 :
                VEBASVEB += 1

        Dict_Result = dict()
        Dict_Result['Normal(G) as Normal'] = NormalAsNormal
        Dict_Result['Normal(G) as VEB'] = NormalAsVEB
        Dict_Result['VEB(G) as VEB'] = VEBASVEB
        Dict_Result['VEB(G) as Normal'] = VEBASNormal

        return Dict_Result







if __name__ == "__main__":
    List_RecordNum = [100, 105, 106, 108, 109, 113, 114, 116, 118, 119,
                      200, 201, 202, 203, 207, 208, 209, 210, 213, 214,
                      215, 219, 220, 221, 223, 228, 233]

    List_Test = [100, 103, 105, 111, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 233, 234]
    List_Train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 220]
    VEB = [200, 202, 210, 213, 214, 219, 221, 228, 231, 233, 234]
    SVEB = [200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

    IntRecordNum = 203
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

    ObjSVM = Competitive_NN(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds,StrWaveletBasis = StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=FltThreshold, alpha=alpha)
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

    indim = 1
    outdim = 2
    numCenters = 256
    OBJ_RBF = RBF_NN(indim=indim, outdim=outdim, numCenters=numCenters)
    OBJ_RBF.Train(X = ArrayMat_Train, Y=Y)
    MyAns = OBJ_RBF.Test(X=ArrayMat_Test)
    MyAnswer = list()
    for ans in MyAns:
        dist0 = np.abs(ans - 0)
        dist1 = np.abs(ans - 1)
        if dist0 < dist1 :
            MyAnswer.append(0)
        else:
            MyAnswer.append(1)

    NormalAsNormal = 0
    NormalAsVEB = 0
    VEBASVEB = 0
    VEBASNormal = 0
    for a,b in zip(MyAnswer,TrueAnswer):
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





    # 여기서부터 Neural Network 적용
    # TrainNum, Dim = ArrayMat_Train.shape
    # NNData = ClassificationDataSet(Dim, 1)
    #
    #
    # for idx in range(len(Y)):
    #     NNData.addSample(np.ravel(ArrayMat_Train[idx]), Y[idx])
    # NNData._convertToOneOfMany()
    # # NNData._convertToOneOfMany()
    # # HiddenNum = int(len(ArrayMat_Train)/ float(2 * (NNData.indim + NNData.outdim)))
    # HiddenNum = 2
    # print NNData.indim
    # print Dim
    # # print NNData.outdim
    # # print HiddenNum
    # NNNetwork = buildNetwork(NNData.indim, HiddenNum, NNData.outdim, outclass = GaussianLayer)
    # # NNNetwork = buildNetwork(NNData.indim, HiddenNum, NNData.outdim, hiddenclass=TanhLayer)
    # Trainer = BackpropTrainer(NNNetwork, dataset=NNData)
    #
    # NNTest = ClassificationDataSet(Dim, 1)
    #
    # for idx in range(len(TrueAnswer)):
    #     # print np.ravel(ArrayMat_Test[idx]), TrueAnswer[idx]
    #     NNTest.addSample(np.ravel(ArrayMat_Test[idx]), TrueAnswer[idx])
    # NNTest._convertToOneOfMany()
    # # trainer = BackpropTrainer( NNNetwork, dataset=NNData, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
    # trainer = BackpropTrainer( NNNetwork, dataset=NNData)
    # # trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )
    # trainer.trainUntilConvergence(verbose=True, maxEpochs = 500)
    # MyAnswer = trainer.testOnClassData(dataset=NNTest)
    #
    # NormalAsNormal = 0
    # NormalAsVEB = 0
    # VEBASVEB = 0
    # VEBASNormal = 0
    #
    # for a,b in zip(MyAnswer,TrueAnswer):
    #     if b ==0 and a == 0:
    #         NormalAsNormal+= 1
    #     elif b == 0 and a == 1:
    #         NormalAsVEB += 1
    #     elif a == 0 and b == 1:
    #         VEBASNormal += 1
    #     elif a == 1 and b == 1 :
    #         VEBASVEB += 1
    #
    # TN = NormalAsNormal
    # TP = VEBASVEB
    # FP = NormalAsVEB
    # FN = VEBASNormal
    #
    # print "NeuralNet"
    # print "Record", IntRecordNum
    # print "Normal(G) as Normal" , NormalAsNormal
    # print "Normal(G) as VEB" , NormalAsVEB
    # print "VEB(G) as Normal", VEBASNormal
    # print "VEB(G) as VEB", VEBASVEB
    # print "-" * 50
    #
    # print "Sensitivity", TP / float(TP + FN)
    # print "Specificity", TN / (float(TN + FP))
    # print "Positive Predictivity", TP / float(TP + FP)
    # print "False Positive Rate", FP / float(TN + FP)
    # print "Classification rate", (TP + TN) / float(TN + TP + FN + FP)





    # NNTest =
    # trainer.trainEpochs(1)
    # trainer.testOnData(dataset=ArrayMat_Test)


    # clf = svm.SVC()
    # clf.fit(ArrayMat_Train, Y)
    # # 0 : Normal
    # # 1 : PVC
    # SVMAnswer = clf.predict(ArrayMat_Test)
    #
    # NormalAsNormal = 0
    # NormalAsVEB = 0
    # VEBASVEB = 0
    # VEBASNormal = 0
    #
    # for a,b in zip(SVMAnswer,TrueAnswer):
    #     if b ==0 and a == 0:
    #         NormalAsNormal+= 1
    #     elif b == 0 and a == 1:
    #         NormalAsVEB += 1
    #     elif a == 0 and b == 1:
    #         VEBASNormal += 1
    #     elif a == 1 and b == 1 :
    #         VEBASVEB += 1
    #
    # print "Record", IntRecordNum
    # print "Normal(G) as Normal" , NormalAsNormal
    # print "Normal(G) as VEB" , NormalAsVEB
    # print "VEB(G) as Normal", VEBASNormal
    # print "VEB(G) as VEB", VEBASVEB

