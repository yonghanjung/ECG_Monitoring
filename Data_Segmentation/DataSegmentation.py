# -*- coding: utf-8 -*-
'''
Goal : Dataset 을 Global Trainset과 Local Trainset 으로 나누어서 저장하고 싶었다.
Author : Yonghan Jung, ISyE, KAIST 
Date : 150730
Comment
순서는 다음과 같다.
1. Raw 데이터와 annotation 파일을 로딩한다.
2. 데이터를 segmenting 한다.
3. Training 함수를 구성한다.
==> Training 은 Normal, VEB, SVEB 를 모두 뱉어낸다.
==> Test 는 Normal, VEB, SVEB 를 모두 뱉어낸다.
'''

''' Library '''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io.matlab
import pywt

''' Function or Class '''

class DataSegmentation:
    ''' __init__ 은 메인함수의 역할을 수행한다. '''
    def __init__(self):
        # Global Traindata 를 구성한다.
        def Construct_GE(List_TrainDataNum, Int_LeadNum):
            # 이 함수 하나로 Global Train, Global Test 전부 구성가능
            Dict_WCNormal = dict()
            Dict_WCVEB = dict()
            Dict_WCSVEB = dict()

            for Int_RecordNum in List_TrainDataNum:
                # Data record 와 annotation 파일을 로딩
                Array_ECGRecord = self.Load_MatData(Int_RecordNum=Int_RecordNum, Int_LeadNum=Int_LeadNum)
                Dict_Anno = self.Load_AnnoFile(Int_RecordNum=Int_RecordNum)
                Array_RIdx = Dict_Anno['Sample']
                Array_Type = Dict_Anno['Type']

                # Segmentation
                # 1. Normal
                ArrayMat_Normal = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="Normal" )
                ArrayMat_VEB = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="VEB" )
                ArrayMat_SVEB = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="SVEB" )

                Dict_WCNormal[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_Normal)
                Dict_WCVEB[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_VEB)
                Dict_WCSVEB[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_SVEB)

            return Dict_WCNormal, Dict_WCVEB, Dict_WCSVEB

        def Construct_LocalTrain(List_TestDataNum, Int_LeadNum):
            # 각 Test 데이터 레코드의 5분간 wavelet coefficients 들을 모은다.
            Dict_WCNormal = dict()
            Dict_WCVEB = dict()
            Dict_WCSVEB = dict()

            for Int_RecordNum in List_TestDataNum:
                Array_ECGRecord = self.Load_MatData(Int_RecordNum=Int_RecordNum, Int_LeadNum=Int_LeadNum)
                Dict_Anno = self.Load_AnnoFile(Int_RecordNum=Int_RecordNum)
                Array_RIdx = Dict_Anno['Sample']
                Array_Type = Dict_Anno['Type']

                ArrayMat_Normal_LTrain = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="Normal", Option="LocalTrain" )
                ArrayMat_VEB_LTrain = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="VEB", Option = "LocalTrain" )
                ArrayMat_SVEB_LTrain = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="SVEB", Option = "LocalTrain" )


                Dict_WCNormal[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_Normal_LTrain)
                Dict_WCVEB[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_VEB_LTrain)
                Dict_WCSVEB[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_SVEB_LTrain)

            return Dict_WCNormal, Dict_WCVEB, Dict_WCSVEB


        def Construct_LocalTest(List_TestDataNum, Int_LeadNum):
            # 각 Test 데이터 레코드의 5분간 wavelet coefficients 들을 모은다.
            Dict_WCNormal = dict()
            Dict_WCVEB = dict()
            Dict_WCSVEB = dict()

            for Int_RecordNum in List_TestDataNum:
                Array_ECGRecord = self.Load_MatData(Int_RecordNum=Int_RecordNum, Int_LeadNum=Int_LeadNum)
                Dict_Anno = self.Load_AnnoFile(Int_RecordNum=Int_RecordNum)
                Array_RIdx = Dict_Anno['Sample']
                Array_Type = Dict_Anno['Type']

                ArrayMat_Normal_LTest = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="Normal", Option="LocalTest" )
                ArrayMat_VEB_LTest = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="VEB", Option = "LocalTest" )
                ArrayMat_SVEB_Lest = self.Segment_MatECG(Mat_ECGRecord=Array_ECGRecord, Array_RPeakIdx=Array_RIdx, Array_Type=Array_Type, Type_youwant="SVEB", Option = "LocalTest" )


                Dict_WCNormal[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_Normal_LTest)
                Dict_WCVEB[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_VEB_LTest)
                Dict_WCSVEB[Int_RecordNum] = self.Wavelet_for_Segment(ArrayMat_Record=ArrayMat_SVEB_Lest)

            return Dict_WCNormal, Dict_WCVEB, Dict_WCSVEB

        ''' 설정값 '''
        self.List_TrainDataNum = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 220]
        self.List_TestDataNum = [100, 103, 105, 111, 105, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 233, 234]
        self.Frequency = 360
        self.Int_LeadNum = 0

        ''' Output '''
        self.SetDict_GlobalTrain = Construct_GE(List_TrainDataNum=self.List_TrainDataNum, Int_LeadNum=self.Int_LeadNum)
        self.SetDict_GlobalTest = Construct_GE(List_TrainDataNum=self.List_TestDataNum, Int_LeadNum=self.Int_LeadNum)
        self.SetDict_LocalTrain = Construct_LocalTrain(List_TestDataNum=self.List_TestDataNum, Int_LeadNum=self.Int_LeadNum)
        self.SetDict_LocalTest = Construct_LocalTest(List_TestDataNum=self.List_TestDataNum, Int_LeadNum=self.Int_LeadNum)

    ''' Raw 데이터를 불러들인다. '''
    def Load_MatData(self, Int_RecordNum, Int_LeadNum):
        Str_Path = "../Data/" + str(Int_RecordNum) + "_file.mat"
        A = scipy.io.loadmat(Str_Path)
        A = A['val']
        A = A[Int_LeadNum]
        return np.squeeze(np.asarray(A))

    def Load_AnnoFile(self, Int_RecordNum):
        Str_Path = "../Data/" + str(Int_RecordNum) + "_anno.txt"
        File_Anno = open(Str_Path)
        Index_dict = {}
        Index_dict['Time'] = []
        Index_dict['Sample'] = []
        Index_dict['Type'] = []

        for each_line in File_Anno.readlines():
            try :
                A = each_line.split(" ")
                b = [elem for elem in A if elem != ""]
                # b[0] time, b[1] sample idx, b[2] Type
                Index_dict['Time'].append(b[0])
                Index_dict['Sample'].append(int(b[1]))
                Index_dict['Type'].append(b[2])
                    # Index_dict.update({'Time' : b[0], 'Sample' : int(b[1]), 'Type' : b[2]})
            except:
                pass

        return Index_dict

    ''' ECG record 를 beat 별로 segmenting 하여 원하는 타입의 데이터만 불러들인다. '''
    def Segment_MatECG(self, Mat_ECGRecord, Array_RPeakIdx, Array_Type, Type_youwant, Option = "GlobalTrain" ):
        MatList_SegmentedECG = list()
        Int_IterIdx = 0
        for Each_RIdx in Array_RPeakIdx:
            # 윈도우가 데이터 길이에 맞게
            if Each_RIdx > 128 and Each_RIdx + 128 < len(Mat_ECGRecord):
                # Normal 을 원할 경우
                if Type_youwant == "Normal":
                    if Array_Type[Int_IterIdx] == 'N' or Array_Type[Int_IterIdx] == 'L' or \
                        Array_Type[Int_IterIdx] == 'R' or Array_Type[Int_IterIdx] == 'e' or \
                        Array_Type[Int_IterIdx] == 'j':
                        if Option == "GlobalTrain":
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)
                        elif Option == "LocalTrain" :
                            if Each_RIdx + 128 > self.Frequency * 300 :
                                break
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)

                        elif Option == "LocalTest":
                            if Each_RIdx + 128 < self.Frequency * 300 :
                                continue
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)

                elif Type_youwant == "SVEB":
                    if Array_Type[Int_IterIdx] == 'A' or Array_Type[Int_IterIdx] == 'a' or \
                        Array_Type[Int_IterIdx] == 'S' :
                        if Option == "GlobalTrain":
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)
                        elif Option == "LocalTrain" :
                            if Each_RIdx + 128 > self.Frequency * 300 :
                                break
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)

                        elif Option == "LocalTest":
                            if Each_RIdx + 128 < self.Frequency * 300 :
                                continue
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)

                elif Type_youwant == "VEB":
                    if Array_Type[Int_IterIdx] == 'V':
                        if Option == "GlobalTrain":
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)
                        elif Option == "LocalTrain" :
                            if Each_RIdx + 128 > self.Frequency * 300 :
                                break
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)

                        elif Option == "LocalTest":
                            if Each_RIdx + 128 < self.Frequency * 300 :
                                continue
                            Array_EachSegment = Mat_ECGRecord[range(Each_RIdx - 128, Each_RIdx + 128)]
                            MatList_SegmentedECG.append(Array_EachSegment)

            Int_IterIdx += 1
        Array_SegmentedECG = np.array(MatList_SegmentedECG)
        return Array_SegmentedECG

    ''' Wavelet Transformation '''
    def Wavelet_Transformation(self, Array_Record, Char_WaveletBasis, Int_Level):
        return pywt.wavedec(data = Array_Record, mode = 'per' , wavelet = Char_WaveletBasis, level = Int_Level)

    # 각 record 별, segmented ECG 를 segmented WC 로 바꾸어줌.
    def Wavelet_for_Segment(self, ArrayMat_Record, Char_WaveletBasis="db6", Int_Level=4 ):
        ArrayMat_WCs = list()
        for EachRecord in ArrayMat_Record:
            ListWCs = self.Wavelet_Transformation(Array_Record=EachRecord, Char_WaveletBasis=Char_WaveletBasis, Int_Level=Int_Level)
            WCs = np.concatenate([ListWCs[0], ListWCs[1], ListWCs[2]])
            ArrayMat_WCs.append(WCs)
        return np.array(ArrayMat_WCs)





if __name__ == "__main__":
    Int_RecordNum = 200
    Type_Want= "SVEB"

    Obj_DataSegment = DataSegmentation()
    print len(Obj_DataSegment.SetDict_GlobalTest)
