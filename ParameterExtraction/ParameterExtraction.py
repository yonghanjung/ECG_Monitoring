# -*- coding: utf-8 -*-
'''
Goal : Parameter Extractor
1. Global Traindata 에서 Mu와 Sigma 를 뽑아낸다.
2. Local Train 데이터에서 Mu와 Sigma 를 뽑아낸다.

Author : Yonghan Jung, ISyE, KAIST 
Date : 150730
WorkFlow
1. Global TrainData를 불러온다.
2. Global TrainData 로 Statistics 를 구성한다.
3. Local Train 데이터로 Stat을 구성한다.

'''

''' Library '''
from Data_Segmentation.DataSegmentation import DataSegmentation
import matplotlib.pyplot as plt
import numpy as np


''' Function or Class '''

class ParameterExtraction(DataSegmentation):
    def __init__(self):
        # 이건 메인함수의 역할을 수행한다.
        ObjDataSeg = DataSegmentation()
        self.Dict_WCNormal_GlobalTrain = ObjDataSeg.SetDict_GlobalTrain[0]
        self.Dict_WCVEB_GlobalTrain = ObjDataSeg.SetDict_GlobalTrain[1]
        self.Dict_WCSVEB_GlobalTrain =  ObjDataSeg.SetDict_GlobalTrain[2]

        self.Dict_WCNormal_GlobalTest = ObjDataSeg.SetDict_GlobalTest[0]
        self.Dict_WCVEB_GlobalTest = ObjDataSeg.SetDict_GlobalTest[1]
        self.Dict_WCSVEB_GlobalTest = ObjDataSeg.SetDict_GlobalTest[2]

        self.Dict_WCNormal_LocalTrain = ObjDataSeg.SetDict_LocalTrain[0]
        self.Dict_WCVEB_LocalTrain = ObjDataSeg.SetDict_LocalTrain[1]
        self.Dict_WCSVEB_LocalTrain = ObjDataSeg.SetDict_LocalTrain[2]

        self.Dict_WCNormal_LocalTest = ObjDataSeg.SetDict_LocalTest[0]
        self.Dict_WCVEB_LocalTest = ObjDataSeg.SetDict_LocalTest[1]
        self.Dict_WCSVEB_LocalTest = ObjDataSeg.SetDict_LocalTest[2]


    # Global Traindata 에 대해서 Mean 과 Cov 를 계산했다.
    # COV 는 독립이니까 대각행렬을 가정했다.
    def Extract_GlobalParameter(self):
        Dict_Normal_Param = dict()
        Dict_VEB_Param = dict()
        Dict_SVEB_Param = dict()

        # 먼저 Dictionary 를 다 모으는 Array를 만들자
        ArrayMat_Normal = np.empty(shape=(1,64))
        for Idx, Key in enumerate(sorted(self.Dict_WCNormal_GlobalTrain)):
            # 약 1800개의 beat 가 각 64개씩 : 1800 by 64
            Each_WCNormal = self.Dict_WCNormal_GlobalTrain[Key]
            ArrayMat_Normal = np.append(ArrayMat_Normal,Each_WCNormal, axis=0)
        Dict_Normal_Param['Mean'] = np.mean(ArrayMat_Normal, axis=0)
        Dict_Normal_Param['Var'] = np.diag(np.var(ArrayMat_Normal,axis=0))

        ArrayMat_VEB = np.empty(shape=(1,64))
        for Idx, Key in enumerate(sorted(self.Dict_WCVEB_GlobalTrain)):
            # 약 1800개의 beat 가 각 64개씩 : 1800 by 64
            Each_WCVEB = self.Dict_WCVEB_GlobalTrain[Key]
            if len(Each_WCVEB) == 0:
                continue
            ArrayMat_VEB = np.append(ArrayMat_VEB,Each_WCVEB, axis=0)
        Dict_VEB_Param['Mean'] = np.mean(ArrayMat_VEB, axis=0)
        Dict_VEB_Param['Var'] = np.diag(np.var(ArrayMat_VEB,axis=0))

        ArrayMat_SVEB = np.empty(shape=(1,64))
        for Idx, Key in enumerate(sorted(self.Dict_WCSVEB_GlobalTrain)):
            # 약 1800개의 beat 가 각 64개씩 : 1800 by 64
            Each_WCSVEB = self.Dict_WCSVEB_GlobalTrain[Key]
            if len(Each_WCSVEB) == 0:
                continue
            ArrayMat_SVEB = np.append(ArrayMat_VEB,Each_WCSVEB, axis=0)
        Dict_SVEB_Param['Mean'] = np.mean(ArrayMat_SVEB, axis=0)
        Dict_SVEB_Param['Var'] = np.diag(np.var(ArrayMat_SVEB,axis=0))

        return Dict_Normal_Param, Dict_VEB_Param, Dict_SVEB_Param

    def Extract_LocalParameter(self):
        Dict_Normal_Mean = dict()
        Dict_Normal_Cov = dict()

        Dict_VEB_Mean = dict()
        Dict_VEB_Cov = dict()

        Dict_SVEB_Mean = dict()
        Dict_SVEB_Cov = dict()
        for idx, key in enumerate(sorted(self.Dict_WCNormal_LocalTrain)):
            Each_RecordData = self.Dict_WCNormal_LocalTrain[key]
            Dict_Normal_Mean[key] = np.mean(Each_RecordData, axis=0)
            Dict_Normal_Cov[key] = np.diag(np.var(Each_RecordData, axis=0))

        for idx, key in enumerate(sorted(self.Dict_WCVEB_LocalTrain)):
            Each_RecordData = self.Dict_WCVEB_LocalTrain[key]
            if len(Each_RecordData) == 0:
                continue
            Dict_VEB_Mean[key] = np.mean(Each_RecordData, axis=0)
            Dict_VEB_Cov[key] = np.diag(np.var(Each_RecordData, axis=0))

        for idx, key in enumerate(sorted(self.Dict_WCSVEB_LocalTrain)):
            Each_RecordData = self.Dict_WCSVEB_LocalTrain[key]
            if len(Each_RecordData) == 0:
                continue
            Dict_SVEB_Mean[key] = np.mean(Each_RecordData, axis=0)
            Dict_SVEB_Cov[key] = np.diag(np.var(Each_RecordData, axis=0))

        return Dict_Normal_Mean, Dict_Normal_Cov, Dict_VEB_Mean, Dict_VEB_Cov, Dict_SVEB_Mean, Dict_SVEB_Cov



if __name__ == "__main__":
    Obj = ParameterExtraction()
    Dict = Obj.Dict_WCVEB_GlobalTrain
    Obj.Extract_LocalParameter()



