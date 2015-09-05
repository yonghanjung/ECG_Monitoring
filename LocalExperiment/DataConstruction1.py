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
        # print Matrix_Train.shape

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

                # VEBSVEB
                # VEB Label
                elif RIdx > self.SecondsToSample and (DictECGLabel[RIdx] == 'V' or DictECGLabel[RIdx] == 'E'):
                # SVEB Label
                # elif RIdx > self.SecondsToSample and (DictECGLabel[RIdx] == 'A' or DictECGLabel[RIdx] == 'a' or DictECGLabel[RIdx] == 'S' or DictECGLabel[RIdx] == 'J'):
                    DictArrayTestBeatPVC[RIdx] = NewSignal

                # if len(DictArrayTestBeat) > 50:
                #     break

            else:
                pass


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
            # DictArray_TrainWCNormal[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])
            # DictArray_TrainWC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])

            # MAD 추정치
            WaveletThreshold = np.sqrt(2 * np.log(256)) * (np.median(np.abs(np.array(ListWCs[4]) - np.median(ListWCs[4]))) / 0.6745)

            WaveletCoefs = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2], ListWCs[3], ListWCs[4]])
            for idx in range(len(WaveletCoefs)):
                if WaveletCoefs[idx] < WaveletThreshold:
                    WaveletCoefs[idx] = 0.0
            DictArray_TrainWCNormal[key] = WaveletCoefs
            DictArray_TrainWC[key] = WaveletCoefs

            # DictArray_TrainWC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2], ListWCs[3], ListWCs[4]])

        for idx, key in enumerate(sorted(DictArray_TrainBeatPVC)):
            ObjWCs = Wavelet_Coefficient_Extractor(DictArray_TrainBeatPVC[key], Wavelet_Basis_Fun=self.StrWaveletBasis,Level=self.IntDecompLevel)
            ListWCs = ObjWCs.WaveDec()
            # DictArray_TrainWCPVC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])
            # DictArray_TrainWC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])
            WaveletThreshold = np.sqrt(2 * np.log(256)) * (np.median(np.abs(np.array(ListWCs[4]) - np.median(ListWCs[4]))) / 0.6745)
            WaveletCoefs = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2], ListWCs[3], ListWCs[4]])
            for idx in range(len(WaveletCoefs)):
                if WaveletCoefs[idx] < WaveletThreshold:
                    WaveletCoefs[idx] = 0.0
            DictArray_TrainWCPVC[key] = WaveletCoefs
            DictArray_TrainWC[key] = WaveletCoefs
            # DictArray_TrainWCPVC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2], ListWCs[3], ListWCs[4]])
            # DictArray_TrainWC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2], ListWCs[3], ListWCs[4]])

        return DictArray_TrainWC, DictArray_TrainWCNormal, DictArray_TrainWCPVC, Dict_TrainWCLabel

    def TestWCConstruction(self):
        DictArray_TestBeat, _, _, Dict_TestLabel = self.TestDataConstruction()

        DictArray_TestWC = dict()

        for idx, key in enumerate(sorted(DictArray_TestBeat)):
            TargetSignal = np.squeeze(np.asarray(DictArray_TestBeat[key]))
            # print TargetSignal.shape
            ObjWCs = Wavelet_Coefficient_Extractor(TargetSignal, Wavelet_Basis_Fun=self.StrWaveletBasis,Level=self.IntDecompLevel)
            ListWCs = ObjWCs.WaveDec()
            WaveletThreshold = np.sqrt(2 * np.log(256)) * (np.median(np.abs(np.array(ListWCs[4]) - np.median(ListWCs[4]))) / 0.6745)
            WaveletCoefs = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2], ListWCs[3], ListWCs[4]])
            for idx in range(len(WaveletCoefs)):
                if WaveletCoefs[idx] < WaveletThreshold:
                    WaveletCoefs[idx] = 0.0
            # DictArray_TestWC[key] = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2], ListWCs[3], ListWCs[4]])
            DictArray_TestWC[key] = WaveletCoefs

        return DictArray_TestWC, Dict_TestLabel



# List_Test = [100, 103, 105, 111, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 233, 234]


if __name__ == "__main__":
    IntRecordNum = 200
    IntRecordType = 0
    IntSeconds = 300

    StrWaveletBasis = 'db8'
    IntDecompLevel = 4

    ObjDataConstruction = DataConstruction(RecordNum=IntRecordNum, RecordType=IntRecordType, Seconds=IntSeconds, StrWaveletBasis=StrWaveletBasis, IntDecompLevel=IntDecompLevel)
    DictArrayTrainBeat, DictArrayTrainBeatNormal, DictArrayTrainBeatPVC, DictTrainLabel = ObjDataConstruction.TrainDataConstruction()
    DictArray_TrainWC, DictArray_TrainWCNormal, DictArray_TrainWCPVC, Dict_TrainWCLabel = ObjDataConstruction.TrainWCConstruction()
    DictArrayTestBeat, DictArrayTestBeatNormal, DictArrayTestBeatPVC, DictArrayTestLabel = ObjDataConstruction.TestDataConstruction()

    plt.figure()
    plt.title(str(IntRecordNum)+ " False")
    # for idx, key in enumerate(sorted(DictArrayTrainBeatNormal)):
    #     plt.plot(DictArrayTrainBeatNormal[key],'g')

    # plt.figure()

    for idx,key in enumerate(sorted(DictArrayTestBeat)):
        if DictArrayTestLabel[key] == 'N' or DictArrayTestLabel[key] == 'L' or DictArrayTestLabel[key] == 'R' or DictArrayTestLabel[key] == 'e' or DictArrayTestLabel[key] =='j' :
            plt.plot(DictArrayTestBeat[key], 'b')
        elif DictArrayTestLabel[key] == 'A' or DictArrayTestLabel[key] == 'a' or DictArrayTestLabel[key] == 'S':
        # elif DictArrayTestLabel[key] == 'V' or DictArrayTestLabel[key] == 'E':
            plt.plot(DictArrayTestBeat[key], 'r')
        elif DictArrayTestLabel[key] == 'V' or DictArrayTestLabel[key] == 'E':
            plt.plot(DictArrayTestBeat[key], 'orange')
    # plt.legend()
    plt.show()

    # DictArrayECG, DictArrayLabel = ObjDataConstruction.DataSegmentation()
    # print ObjDataConstruction.Index_Loading()
    # print DictArrayLabel.values()

