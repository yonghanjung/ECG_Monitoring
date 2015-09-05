# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
from InControlConstruction2 import InControlConstruction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Old.Compute_Fisher_Score.HansFisherScore import Fisher_Score_Compute
from Old.Compute_Fisher_Score.HansFisherLDA import FisherLDA
from Old.Compute_Fisher_Score.TwoClassFisherLDA import TwoClassFisherLDA
from Class_SparseLDA import SparseLDA
from Old.Compute_Fisher_Score import NewtonMethodLDA_Ver1

''' Function or Class '''


class FeatureSelector(InControlConstruction):
    def __init__(self, RecordNum , RecordType, Seconds, StrWaveletBasis, IntDecompLevel, LDAorNOT, Threshold):
        # InControl Data Load
        InControlConstruction.__init__(self, RecordNum, RecordType, Seconds, StrWaveletBasis, IntDecompLevel)
        self.Sampling_rate = 360
        self.SecondsToSample = Seconds * self.Sampling_rate
        self.LDAorNOT = LDAorNOT
        self.Str_WaveletBasis = StrWaveletBasis
        self.Int_DecompLevel = IntDecompLevel
        self.Float_Threshold = Threshold

        # COL #
        self.Int_NumTrainFeature = self.Int_NumTrainFeature
        # ROW #
        self.Int_NumTrainNormal = self.Int_NumTrainNormal


    ### LDA OFF ###
    ## 1. Fisher Score Compute
    ## 2. Feature Selection
    ## 3. Mean Compute
    ## 4. CoVar Compute

    def LDAOFF_FisherScoreCompute(self):
        DictArrayMatrix_ForLDA = dict() #0 : Normal, 1 : PVC
        DictArrayMatrix_ForLDA[0] = list()
        DictArrayMatrix_ForLDA[1] = list()

        for idx, key in enumerate(sorted(self.DictArray_TrainWCNormal)):
            DictArrayMatrix_ForLDA[0].append(self.DictArray_TrainWCNormal[key])
        for idx, key in enumerate(sorted(self.DictArray_TrainWCPVC)):
            DictArrayMatrix_ForLDA[1].append(self.DictArray_TrainWCPVC[key])
        # print "LengthofPVC", len(DictArrayMatrix_ForLDA[1])
        # print "LengthofNormal", len(DictArrayMatrix_ForLDA[0])

        ObjFisherCompute = Fisher_Score_Compute(Training=DictArrayMatrix_ForLDA)
        List_FisherScoreForFeatures = ObjFisherCompute.Fisher_Score()
        Int_NumFeatureSelected = ObjFisherCompute.HowMany(self.Float_Threshold)
        List_IdxFeatureSelected, _ = ObjFisherCompute.FeatureSelector(Int_NumFeatureSelected)

        return List_FisherScoreForFeatures, Int_NumFeatureSelected, List_IdxFeatureSelected

    def LDAOFF_ReducedMeanComputation(self):
        List_FisherScoreForFeatures, Int_NumFeatures, List_IdxSelected = self.LDAOFF_FisherScoreCompute()
        Array_MeanReduced = np.array([float(0)] * Int_NumFeatures)
        for idx, key in enumerate(sorted(self.DictArray_TrainWCNormal)):
            Array_Reduced = self.DictArray_TrainWCNormal[key][List_IdxSelected[:Int_NumFeatures]]
            Array_MeanReduced += Array_Reduced
        Array_MeanReduced /= self.Int_NumTrainNormal
        return Array_MeanReduced

    def LDAOFF_ReducedCoVarComputation(self):
        List_FisherScoreForFeatures, Int_NumFeatures, List_IdxSelected = self.LDAOFF_FisherScoreCompute()
        ArrayMat = list()
        for idx, key in enumerate(sorted(self.DictArray_TrainWCNormal)):
            ArrayMat.append(self.DictArray_TrainWCNormal[key][List_IdxSelected[:Int_NumFeatures]])
        ArrayMat = np.array(ArrayMat)
        return np.diag(np.var(ArrayMat, axis=0))

    def TEMP_LDAOFF_CoVarComputation(self):
        ArrayMat = list()
        for idx, key in enumerate(sorted(self.DictArray_TrainWCNormal)):
            ArrayMat.append(self.DictArray_TrainWCNormal[key])
        ArrayMat = np.array(ArrayMat)
        return np.diag(np.var(ArrayMat, axis=0))






    ### LDA ON ###
    ## 1. LDA Operator Construction
    ## 2. Construct New Train Feature (for computing mean, cov)
    ## 3. Feature Selection
    ## 3. Mean Compute
    ## 4. CoVar Compute

    def LDAON_LDAOperatorConstruction(self):
        # Compute LDA Coefficeints
        DictArrayMatrix_ForLDA = dict() #0 : Normal, 1 : PVC
        DictArrayMatrix_ForLDA[0] = list()
        DictArrayMatrix_ForLDA[1] = list()

        for idx, key in enumerate(sorted(self.DictArray_TrainWCNormal)):
            DictArrayMatrix_ForLDA[0].append(self.DictArray_TrainWCNormal[key])
        for idx, key in enumerate(sorted(self.DictArray_TrainWCPVC)):
            DictArrayMatrix_ForLDA[1].append(self.DictArray_TrainWCPVC[key])
        DictArrayMatrix_ForLDA[0] = np.array(DictArrayMatrix_ForLDA[0])
        DictArrayMatrix_ForLDA[1] = np.array(DictArrayMatrix_ForLDA[1])
        # ObjLDA = FisherLDA(TrainingData=DictArrayMatrix_ForLDA)
        # TwoClass
        # ObjLDA = TwoClassFisherLDA(TrainingData=DictArrayMatrix_ForLDA)

        ### TRIAL ###
        ObjLDA = SparseLDA(Dict_TrainingData=DictArrayMatrix_ForLDA, Flt_Lambda=0.3, Flt_L1=0.1)

        # ObjectNewtonMethod = NewtonMethodLDA_Ver1.NewtonLDA(DictArrayMatrix_ForLDA)
        # Mat_LDAOperator = ObjLDA.LDAOperator()

        # Mat_LDAOperator = ObjLDA.ConstructW()
        Mat_LDAOperator = ObjLDA.B

        # Mat_LDAOperator = ObjectNewtonMethod.GradientDescent()
        return Mat_LDAOperator

    def LDAON_TrainWCNormalConstruction(self):
        DictArray_TrainWCNormal_LDA = dict()

        # 이게 W
        Matrix_LDAOpeartor = self.LDAON_LDAOperatorConstruction()

        # Key : Record , Value : Train Normal WC 마다
        for idx, key in enumerate(sorted(self.DictArray_TrainWCNormal)):
            # 1 by 64
            Val = np.reshape(self.DictArray_TrainWCNormal[key], (len(self.DictArray_TrainWCNormal[key]),1 ))

            # w*X
            NewVal = np.dot(np.transpose(Matrix_LDAOpeartor), Val)
            NewVal = np.squeeze(np.asarray(NewVal))
            NewVal = float(NewVal)
            DictArray_TrainWCNormal_LDA[key] = NewVal
        return DictArray_TrainWCNormal_LDA

    def LDAON_TrainWCPVCConstruction(self):
        DictArray_TrainWCPVC_LDA = dict()
        Matrix_LDAOpeartor = self.LDAON_LDAOperatorConstruction()
        for idx, key in enumerate(sorted(self.DictArray_TrainWCPVC)):
            # 1 by 64
            Val = np.reshape(self.DictArray_TrainWCPVC[key], (len(self.DictArray_TrainWCPVC[key]),1 ))

            # w*X
            NewVal = np.dot(np.transpose(Matrix_LDAOpeartor), Val)
            NewVal = np.squeeze(np.asarray(NewVal))
            NewVal = float(NewVal)

            DictArray_TrainWCPVC_LDA[key] = NewVal
        return DictArray_TrainWCPVC_LDA

    def LDAON_TrainWCConstruction(self):
        DictArray_TrainWCNormal_LDA = dict()
        List_Stat = list()
        Matrix_LDAOpeartor = self.LDAON_LDAOperatorConstruction()
        for idx, key in enumerate(sorted(self.DictArray_TrainWC)):
            # 각 비트의 vector를 재조정
            Val = np.reshape(self.DictArray_TrainWC[key], (len(self.DictArray_TrainWC[key]),1 ))

            # w*X
            NewVal = np.dot(np.transpose(Matrix_LDAOpeartor), Val)
            NewVal = np.squeeze(np.asarray(NewVal))
            NewVal = float(NewVal)
            DictArray_TrainWCNormal_LDA[key] = NewVal
            List_Stat.append(NewVal)
        return DictArray_TrainWCNormal_LDA

    def LDAON_TestWCConstruction(self):
        DictArray_TestWCNormal_LDA = dict()
        Matrix_LDAOpeartor = self.LDAON_LDAOperatorConstruction()
        for idx, key in enumerate(sorted(self.DictArray_TestWC)):
            # 각 비트의 vector를 재조정
            Val = np.reshape(self.DictArray_TestWC[key], (len(self.DictArray_TestWC[key]),1 ))

            # w*X
            NewVal = np.dot(np.transpose(Matrix_LDAOpeartor), Val)
            NewVal = np.squeeze(np.asarray(NewVal))
            NewVal = float(NewVal)
            DictArray_TestWCNormal_LDA[key] = NewVal
        return DictArray_TestWCNormal_LDA

    def LDAON_FisherScoreComputation(self):
        # 각각 key : record, val : 1dim num (wX)
        DictArray_TrainWCNormal_LDA = self.LDAON_TrainWCNormalConstruction()
        DictArray_TrainWCPVC_LDA = self.LDAON_TrainWCPVCConstruction()

        # Key : Record, Val : Array (wX 들이 각 record 마다)
        DictArrayMatrix_ForLDA = dict() #0 : Normal, 1 : PVC
        DictArrayMatrix_ForLDA[0] = list()
        DictArrayMatrix_ForLDA[1] = list()

        for idx, key in enumerate(sorted(DictArray_TrainWCNormal_LDA)):
            # Key : Record, Val : Float (wX)
            DictArrayMatrix_ForLDA[0].append(DictArray_TrainWCNormal_LDA[key])
        for idx, key in enumerate(sorted(DictArray_TrainWCPVC_LDA)):
            # Key : Record, Val : Float (wX)
            DictArrayMatrix_ForLDA[1].append(DictArray_TrainWCPVC_LDA[key])

        ObjFisherCompute = Fisher_Score_Compute(Training=DictArrayMatrix_ForLDA)
        List_FisherScoreForFeatures = ObjFisherCompute.Fisher_Score()
        Int_NumFeatureSelected = ObjFisherCompute.HowMany(self.Float_Threshold)
        List_IdxFeatureSelected, _ = ObjFisherCompute.FeatureSelector(Int_NumFeatureSelected)

        return List_FisherScoreForFeatures, Int_NumFeatureSelected, List_IdxFeatureSelected

    def LDAON_ReducedMeanComputation(self):
        # 이거 뭐 하는거지.
        Array_TotalSum = np.zeros((1,64))
        Array_LDACoef = self.LDAON_LDAOperatorConstruction()
        Idx = 1
        for idx, key in enumerate(sorted(self.DictArray_TrainWCNormal)):
            Array_EachWCofBeat = self.DictArray_TrainWCNormal[key]
            Array_TotalSum += Array_EachWCofBeat
            Idx = idx
        Array_AVG = Array_TotalSum / (Idx+1)
        # print Array_LDACoef.shape
        return np.ravel(np.dot(Array_AVG, Array_LDACoef))





        # List_FisherScoreForFeatures, Int_NumFeatures, List_IdxSelected = self.LDAON_FisherScoreComputation()
        # DictArray_TrainWCNormal_LDA = self.LDAON_TrainWCNormalConstruction()
        # Array_MeanReduced = np.array([float(0)] * Int_NumFeatures)
        # for idx, key in enumerate(sorted(DictArray_TrainWCNormal_LDA)):
        #     Array_Reduced = DictArray_TrainWCNormal_LDA[key][List_IdxSelected[:Int_NumFeatures]]
        #     Array_MeanReduced += Array_Reduced
        # Array_MeanReduced /= self.Int_NumTrainNormal
        # return Array_MeanReduced

    def LDAON_ReducedCoVarComputation(self):
        # Construct_Original Cov

        # 1. Construct Array Matrix (300 by 64)
        ArrayMat = list()
        for idx, key in enumerate(sorted(self.DictArray_TrainWCNormal)):
            Array_EachBeatWC = self.DictArray_TrainWCNormal[key]
            ArrayMat.append(Array_EachBeatWC)
        ArrayMat = np.array(ArrayMat)
        Array_VarElem = np.var(ArrayMat, axis=0)
        ArrayMat_COV = np.diag(Array_VarElem)

        # 2. Fisher Coef (64,1)
        Array_FisherCoef = self.LDAON_LDAOperatorConstruction() # 64,1
        FisherTransformedCOV = np.dot(np.dot(np.transpose(Array_FisherCoef), ArrayMat_COV), Array_FisherCoef)
        return np.ravel(FisherTransformedCOV)


        # List_FisherScoreForFeatures, Int_NumFeatures, List_IdxSelected = self.LDAON_FisherScoreComputation()
        # _, Int_TempNumFeatures, _ = self.LDAOFF_FisherScoreCompute()
        # DictArray_TrainWCNormal_LDA = self.LDAON_TrainWCNormalConstruction()
        # ArrayMat = list()
        # for idx, key in enumerate(sorted(DictArray_TrainWCNormal_LDA)):
        #     ArrayMat.append(DictArray_TrainWCNormal_LDA[key][List_IdxSelected[:Int_NumFeatures]])
        #     # ArrayMat.append(DictArray_TrainWCNormal_LDA[key][List_IdxSelected[:Int_TempNumFeatures]])
        # ArrayMat = np.array(ArrayMat)
        # return np.diag(np.var(ArrayMat, axis=0))





if __name__ == "__main__":
    IntRecordNum = 119
    IntRecordType = 0
    IntSeconds = 300

    StrWaveletBasis = 'db6'
    IntDecompLevel = 4
    BoolLDAorNOT = True
    Threshold = 0.5

    ObjFeatureSelector = FeatureSelector(RecordType=IntRecordType, RecordNum= IntRecordNum, Seconds= IntSeconds, StrWaveletBasis=StrWaveletBasis, IntDecompLevel = IntDecompLevel, LDAorNOT=BoolLDAorNOT, Threshold=Threshold)
    DictArray_TrainWCNormal = ObjFeatureSelector.DictArray_TrainWCNormal
    Matrix_LDAOpeartor = ObjFeatureSelector.LDAON_LDAOperatorConstruction()
    LDAON_TrainNormal = ObjFeatureSelector.LDAON_TrainWCNormalConstruction()
    LDAON_TrainPVC = ObjFeatureSelector.LDAON_TrainWCPVCConstruction()

    print ObjFeatureSelector.LDAON_ReducedCoVarComputation()

    # LDAOFF_Cov = ObjFeatureSelector.LDAOFF_ReducedCoVarComputation()
    # LDAON_Cov = ObjFeatureSelector.LDAON_ReducedCoVarComputation()
    # Temp_Cov = ObjFeatureSelector.TEMP_LDAOFF_CoVarComputation()
