# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
from Data_Preparation import Data_Preparation
from Old.Wavelet_Analysis.Wavelet_Analysis import Wavelet_Coefficient_Extractor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from KalmanFilter_NoiseReduction_Ver0 import HansKalmanFilter
''' Function or Class '''


class DataConstruction(Data_Preparation):
    def __init__(self, RecordNum, RecordType, Seconds, StrWaveletBasis, IntDecompLevel):
        ## Data Load
        Data_Preparation.__init__(self, RecordNum, RecordType)
        self.Sampling_rate = 360
        self.SecondsToSample = Seconds * self.Sampling_rate
        self.StrWaveletBasis = StrWaveletBasis
        self.IntDecompLevel = IntDecompLevel

    ### ECG Segmentizing ###


    def DataSegmentation(self):
        DictECGSegment, DictECGSegmentLabel = Data_Preparation.Segment(self)
        DictArrayECG = dict()
        for idx, key in enumerate(sorted(DictECGSegment)):
            DictArrayECG[key] = np.array(DictECGSegment[key])
        # Key : ECGSegmentIdx, Val : ListEachBeat
        return DictArrayECG,DictECGSegmentLabel

    def TrainDataConstruction(self):
        DictArrayECGBeat, DictECGLabel = self.DataSegmentation()
        Int_NoiseLength = len(DictArrayECGBeat[DictArrayECGBeat.keys()[0]])
        Array_Noise = np.random.normal(0,10,Int_NoiseLength)
        Array_Noise = np.zeros(Int_NoiseLength)
        DictArrayTrainBeat \
            = {RIdx : DictArrayECGBeat[RIdx] + Array_Noise for RIdx in DictArrayECGBeat.keys() if RIdx < self.SecondsToSample}
        DictArrayTrainBeatNormal \
            = {RIdx : DictArrayECGBeat[RIdx] + Array_Noise for RIdx in DictArrayECGBeat.keys() if RIdx < self.SecondsToSample and (DictECGLabel[RIdx] == 'N' or DictECGLabel[RIdx] == 'L' or DictECGLabel[RIdx] == 'R' or DictECGLabel[RIdx] == 'e' or DictECGLabel[RIdx] == 'j')}
        DictArrayTrainBeatPVC \
            = {RIdx : DictArrayECGBeat[RIdx] + Array_Noise for RIdx in DictArrayECGBeat.keys() if RIdx < self.SecondsToSample and (DictECGLabel[RIdx] == 'A' or DictECGLabel[RIdx] == 'a' or DictECGLabel[RIdx] == 'S' or DictECGLabel[RIdx] == 'V' or DictECGLabel[RIdx] == 'E') }
        DictTrainLabel = {RIdx : DictECGLabel[RIdx] for RIdx in DictECGLabel.keys() if RIdx < self.SecondsToSample}

        return DictArrayTrainBeat, DictArrayTrainBeatNormal, DictArrayTrainBeatPVC, DictTrainLabel

    def TestDataConstruction(self):
        KalmanIter = 3
        KalmanTrue = False
        DictArrayECGBeat, DictECGLabel = self.DataSegmentation()
        DictArrayTrainBeat, DictArrayTrainBeatNormal, DictArrayTrainBeatPVC, DictTrainLabel = self.TrainDataConstruction()
        Matrix_Train= DictArrayTrainBeat.values()

        Matrix_Train = np.matrix(Matrix_Train).T
        print Matrix_Train.shape

        List_DictARrayECGBeatKey = DictArrayECGBeat.keys()

        DictArrayTestBeat = dict()
        DictArrayTestBeatNormal = dict()
        DictArrayTestBeatPVC = dict()

        DictArrayTestLabel = {RIdx : DictECGLabel[RIdx] for RIdx in DictECGLabel.keys() if RIdx > self.SecondsToSample}


        for RIdx in sorted(List_DictARrayECGBeatKey):
            if RIdx > self.SecondsToSample:
                TargetSignal = DictArrayECGBeat[RIdx]
                if KalmanTrue == True:
                    key = RIdx
                    # if DictArrayTestLabel[key] == 'N' or DictArrayTestLabel[key] == 'L' or DictArrayTestLabel[key] == 'R' or DictArrayTestLabel[key] == 'e' or DictArrayTestLabel[key] =='j':
                    TargetSignal = np.reshape(TargetSignal,(len(TargetSignal),1))
                    # KALMAN with Normal
                    Object_KalmanFilter = HansKalmanFilter(Vector_Obs=TargetSignal, Matrix_Train=Matrix_Train)
                    (NewSignal, _) = Object_KalmanFilter.Execute_Kalman(KalmanIter)
                    NewSignal = np.reshape(NewSignal,(len(NewSignal) ,1))
                    NewSignal = np.squeeze(np.asarray(NewSignal))
                    # else:
                    #     NewSignal = TargetSignal
                    #     pass
                else:
                    NewSignal = TargetSignal
                DictArrayTestBeat[RIdx] = NewSignal

                # Normal 들
                if (DictECGLabel[RIdx] == 'N' or DictECGLabel[RIdx] == 'L' or DictECGLabel[RIdx] == 'R' or DictECGLabel[RIdx] == 'e' or DictECGLabel[RIdx] == 'j') :
                    DictArrayTestBeatNormal[RIdx] = NewSignal

                # VEB + SVEB
                # elif RIdx > self.SecondsToSample and (DictECGLabel[RIdx] == 'A' or DictECGLabel[RIdx] == 'a' or DictECGLabel[RIdx] == 'S' or DictECGLabel[RIdx] == 'V' or DictECGLabel[RIdx] == 'E'):
                elif RIdx > self.SecondsToSample and (DictECGLabel[RIdx] == 'V'):
                    DictArrayTestBeatPVC[RIdx] = NewSignal

                # if len(DictArrayTestBeat) > 50:
                #     break

            else:
                pass

        # DictArrayTestBeat \
        #     = {RIdx : DictArrayECGBeat[RIdx] for RIdx in DictArrayECGBeat.keys() if RIdx > self.SecondsToSample}
        # DictArrayTestBeatNormal \
        #     = {RIdx : DictArrayECGBeat[RIdx] for RIdx in DictArrayECGBeat.keys() if RIdx > self.SecondsToSample and (DictECGLabel[RIdx] == 'N' or DictECGLabel[RIdx] == 'L' or DictECGLabel[RIdx] == 'R' or DictECGLabel[RIdx] == 'e' or DictECGLabel[RIdx] == 'j')}
        # DictArrayTestBeatPVC \
        #     = {RIdx : DictArrayECGBeat[RIdx] for RIdx in DictArrayECGBeat.keys() if RIdx > self.SecondsToSample and (DictECGLabel[RIdx] == 'A' or DictECGLabel[RIdx] == 'a' or DictECGLabel[RIdx] == 'S' or DictECGLabel[RIdx] == 'V' or DictECGLabel[RIdx] == 'E') }


        return DictArrayTestBeat, DictArrayTestBeatNormal, DictArrayTestBeatPVC, DictArrayTestLabel

    def TestSegmentLabel(self):
        PdECGSegment, PdECGSegmentLabel = self.SegmentLoad()
        DictTestSegmentLabel \
            = {RIdx : PdECGSegmentLabel[RIdx] for RIdx in PdECGSegmentLabel.keys() if RIdx > self.SecondsToSample}
        # key : ECG Segment Idx
        PdTestSegment = pd.DataFrame.from_dict(DictTestSegmentLabel, orient = 'index')
        return PdTestSegment

    ### Wavelet Coefficient Costruction ###
    ## Normal for InControl (Phase I)
    ## PVC for OutControl (Phase II)
    def TrainWCConstruction(self):
        DictArray_TrainBeat, DictArray_TrainBeatNormal, DictArray_TrainBeatPVC, DictTrainLabel = self.TrainDataConstruction()

        DictArray_TrainWC = dict()
        DictArray_TrainWCNormal = dict()
        DictArray_TrainWCPVC = dict()
        Dict_TrainWCLabel = DictTrainLabel

        for idx, key in enumerate(sorted(DictArray_TrainBeatNormal)):
            ObjWCs = Wavelet_Coefficient_Extractor(DictArray_TrainBeatNormal[key], Wavelet_Basis_Fun=self.StrWaveletBasis,Level=self.IntDecompLevel)
            ListWCs = ObjWCs.WaveDec()
            DictArray_TrainWCNormal[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])
            DictArray_TrainWC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])

        for idx, key in enumerate(sorted(DictArray_TrainBeatPVC)):
            ObjWCs = Wavelet_Coefficient_Extractor(DictArray_TrainBeatPVC[key], Wavelet_Basis_Fun=self.StrWaveletBasis,Level=self.IntDecompLevel)
            ListWCs = ObjWCs.WaveDec()
            DictArray_TrainWCPVC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])
            DictArray_TrainWC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])

        return DictArray_TrainWC, DictArray_TrainWCNormal, DictArray_TrainWCPVC, Dict_TrainWCLabel

    def TestWCConstruction(self):
        DictArray_TestBeat, _, _, Dict_TestLabel = self.TestDataConstruction()

        DictArray_TestWC = dict()

        for idx, key in enumerate(sorted(DictArray_TestBeat)):
            TargetSignal = np.squeeze(np.asarray(DictArray_TestBeat[key]))
            print TargetSignal.shape
            ObjWCs = Wavelet_Coefficient_Extractor(TargetSignal, Wavelet_Basis_Fun=self.StrWaveletBasis,Level=self.IntDecompLevel)
            ListWCs = ObjWCs.WaveDec()
            DictArray_TestWC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])

        return DictArray_TestWC, Dict_TestLabel






if __name__ == "__main__":
    IntRecordNum = 223
    IntRecordType = 0
    IntSeconds = 120

    StrWaveletBasis = 'db8'
    IntDecompLevel = 4

    ObjDataConstruction = DataConstruction(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds, StrWaveletBasis=StrWaveletBasis, IntDecompLevel=IntDecompLevel)
    DictArrayTrainBeat, DictArrayTrainBeatNormal, DictArrayTrainBeatPVC, DictTrainLabel = ObjDataConstruction.TrainDataConstruction()
    DictArray_TrainWC, DictArray_TrainWCNormal, DictArray_TrainWCPVC, Dict_TrainWCLabel = ObjDataConstruction.TrainWCConstruction()
    DictArrayTestBeat, DictArrayTestBeatNormal, DictArrayTestBeatPVC, DictArrayTestLabel = ObjDataConstruction.TestDataConstruction()

    plt.figure()
    plt.title(str(IntRecordNum)+ " False")
    for idx, key in enumerate(sorted(DictArrayTrainBeatNormal)):
        plt.plot(DictArrayTrainBeatNormal[key],'g')

    # plt.figure()

    for idx,key in enumerate(sorted(DictArrayTestBeat)):
        if DictArrayTestLabel[key] == 'N' or DictArrayTestLabel[key] == 'L' or DictArrayTestLabel[key] == 'R' or DictArrayTestLabel[key] == 'e' or DictArrayTestLabel[key] =='j' :
            plt.plot(DictArrayTestBeat[key], 'b')
        elif DictArrayTestLabel[key] == 'A' or DictArrayTestLabel[key] == 'a' or DictArrayTestLabel[key] == 'S' or DictArrayTestLabel[key] == 'V' or DictArrayTestLabel[key] =='E':
            plt.plot(DictArrayTestBeat[key], 'r')
    plt.show()

    # DictArrayECG, DictArrayLabel = ObjDataConstruction.DataSegmentation()
    # print ObjDataConstruction.Index_Loading()
    # print DictArrayLabel.values()

