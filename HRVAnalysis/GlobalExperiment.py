# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
from FisherLDA import TwoClassFisherLDA
from ParameterExtraction.ParameterExtraction import ParameterExtraction
import numpy as np
import matplotlib.pyplot as plt
''' Function or Class '''


class FeatureExtraction(ParameterExtraction):
    def __init__(self):
        ObjParam = ParameterExtraction()

        # 필요한 것들 다 불러오기
        Dict_Normal_Param, Dict_VEB_Param, Dict_SVEB_Param = ObjParam.Extract_GlobalParameter()
        # Two Class Data
        # Key : recordnum
        # Value : Matrix
        ArrayMat_Global_NormalWC = self.Construc_DictMatrix_for_FisherLDA(ObjParam.Dict_WCNormal_GlobalTrain)
        ArrayMat_Global_VEBWC = self.Construc_DictMatrix_for_FisherLDA(ObjParam.Dict_WCVEB_GlobalTrain)
        ArrayMat_Global_SVEBWC = self.Construc_DictMatrix_for_FisherLDA(ObjParam.Dict_WCSVEB_GlobalTrain)


        Dict_FisherInput_for_FisherLDA = dict()
        Dict_FisherInput_for_FisherLDA['Normal'] = ArrayMat_Global_NormalWC
        Dict_FisherInput_for_FisherLDA['VEB'] = ArrayMat_Global_VEBWC

        ObjLDA = TwoClassFisherLDA(Dict_DataMatrix=Dict_FisherInput_for_FisherLDA)

        # W : 64 by 1 vector
        self.W_Normal_VEB = ObjLDA.W

        # Global Parameter Stat
        self.Dict_GlobalTestStat = self.Compute_Statistics_GlobalTestMatrix(Dict_GlobalTest_Normal=ObjParam.Dict_WCNormal_GlobalTest, Dict_GlobalTest_VEB=ObjParam.Dict_WCVEB_GlobalTest, Dict_GlobalTest_SVEB=ObjParam.Dict_WCSVEB_GlobalTest,Mu_Type=Dict_Normal_Param['Mean'], Cov_Type=Dict_Normal_Param['Var'], WType=self.W_Normal_VEB)


    def Construc_DictMatrix_for_FisherLDA(self, Dict_KeyRecord_ValArrayMatrix):
        Dimension = Dict_KeyRecord_ValArrayMatrix[Dict_KeyRecord_ValArrayMatrix.keys()[0]].shape[1]
        print Dimension
        ArrayMat = np.array([]).reshape(0,Dimension)
        for idx, key in enumerate(sorted(Dict_KeyRecord_ValArrayMatrix)):
            ArrayMat_EachRecord = Dict_KeyRecord_ValArrayMatrix[key]
            ArrayMat_EachRecord = np.squeeze(np.asarray(ArrayMat_EachRecord))
            if len(ArrayMat_EachRecord) == 0:
                continue
            ArrayMat = np.vstack((ArrayMat, ArrayMat_EachRecord))
        return ArrayMat

    # Local Stat
    # def Compute_Statistics_LocalTestMatrix(self):



    # Global Stat
    def Compute_Statistics_GlobalTestMatrix(self, Dict_GlobalTest_Normal, Dict_GlobalTest_VEB, Dict_GlobalTest_SVEB, Mu_Type, Cov_Type, WType):
        Mu_Type = np.reshape(Mu_Type, (len(Mu_Type),1))
        MuShape = Mu_Type.shape
        CovShape = Cov_Type.shape
        print "Mu shape", Mu_Type.shape, "Wshape", WType.shape
        MuW = np.dot(np.transpose(WType), Mu_Type)
        print "MuWShape", MuW.shape
        CovW = np.dot(np.transpose(WType), np.dot(Cov_Type,WType))
        Flt_CovW = np.squeeze(np.asarray(CovW))
        print "COVW", CovW
        print "COVSHape", Cov_Type.shape
        print "COVW shape", CovW.shape
        Dict_Stat = dict()
        Dict_Stat["Normal"] = list()
        Dict_Stat["VEB"] = list()
        Dict_Stat["SVEB"] = list()

        for idx, key in enumerate(sorted(Dict_GlobalTest_Normal)):
            ArrayMat_EachRecord = Dict_GlobalTest_Normal[key]
            for each_beat in ArrayMat_EachRecord:
                each_beat = np.reshape(each_beat, MuShape)
                each_beat = np.dot(np.transpose(WType), each_beat) # 1 by 1
                Flt_each_beat = np.squeeze(np.asarray(each_beat)) # float
                Flt_Stat = Flt_each_beat * (Flt_CovW **(-1)) * Flt_each_beat
                # Float_Stat = np.dot(np.dot(np.transpose(each_beat - MuW), np.linalg.inv(CovW)), each_beat - MuW)
                # Float_Stat = np.squeeze(np.asarray(Float_Stat))
                # print "Normal STAT", Float_Stat
                Dict_Stat["Normal"].append(Flt_Stat)
                # if len(Dict_Stat['Normal']) > 40:
                #     break

        for idx, key in enumerate(sorted(Dict_GlobalTest_VEB)):
            ArrayMat_EachRecord = Dict_GlobalTest_VEB[key]
            if len(ArrayMat_EachRecord) == 0:
                continue
            for each_beat in ArrayMat_EachRecord:
                each_beat = np.reshape(each_beat, MuShape)
                each_beat = np.dot(np.transpose(WType), each_beat)
                Flt_each_beat = np.squeeze(np.asarray(each_beat)) # float
                Flt_Stat = Flt_each_beat * (Flt_CovW **(-1)) * Flt_each_beat
                Float_Stat = np.dot(np.dot(np.transpose(each_beat - MuW), np.linalg.inv(CovW)), each_beat - MuW)
                Float_Stat = np.squeeze(np.asarray(Float_Stat))
                # print "VEB STAT", Float_Stat
                Dict_Stat["VEB"].append(Flt_Stat)
                # if len(Dict_Stat['VEB']) > 40:
                #     break

        for idx, key in enumerate(sorted(Dict_GlobalTest_SVEB)):
            ArrayMat_EachRecord = Dict_GlobalTest_SVEB[key]
            if len(ArrayMat_EachRecord) == 0:
                continue
            for each_beat in ArrayMat_EachRecord:
                each_beat = np.reshape(each_beat, MuShape)
                each_beat = np.dot(np.transpose(WType), each_beat)
                Flt_each_beat = np.squeeze(np.asarray(each_beat)) # float
                Flt_Stat = Flt_each_beat * (Flt_CovW **(-1)) * Flt_each_beat
                Float_Stat = np.dot(np.dot(np.transpose(each_beat - MuW), np.linalg.inv(CovW)), each_beat - MuW)
                Float_Stat = np.squeeze(np.asarray(Float_Stat))
                # print "SVEB STAT", Float_Stat
                Dict_Stat["SVEB"].append(Flt_Stat)
                # if len(Dict_Stat['SVEB']) > 40:
                #     break

        return Dict_Stat







if __name__ == "__main__":
    Obj = FeatureExtraction()
    DictStat = Obj.Dict_GlobalTestStat

    plt.figure()
    plt.grid()
    plt.plot(DictStat['Normal'], 'bo', label = "Normal")
    plt.plot(DictStat['VEB'], 'ro', label="VEB")
    # plt.plot(DictStat['SVEB'], 'go', label = "SVEB")
    plt.legend()
    plt.show()
