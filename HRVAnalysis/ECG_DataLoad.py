# -*- coding: utf-8 -*-
'''
Goal : RR interval 이 어떻게 분포되어있나 확인해보자
Author : Yonghan Jung, ISyE, KAIST 
Date : 150818
Comment 
- 이 프로그램에서는 Locally 어떻게 분포되어 있나 확인해보자.

'''

''' Library '''
from Data_Load import Data_Loading
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

''' Function or Class '''


class Data_Preparation:
    def __init__(self, record_num, Record_Type=0):
        # Declaration of variables
            ## Type : 0 or 1
        self.record_num = record_num
        self.Sampling_rate = 360
        self.DyadLength = 2 ** (int(np.log2(self.Sampling_rate * 60 * 30)))
        self.show_record = Record_Type

        self.To_the_left = 128
        self.To_the_right = 128

        scale = 4
        record_name = ""

    def Data_Load(self):
        Time_domain, ECG_Dyadic_Sample, Dyad_length = \
            Data_Loading(self.record_num,record_type=self.show_record, Sampling_rate=self.Sampling_rate).Data_Loading()
        return Time_domain, ECG_Dyadic_Sample, Dyad_length

    def Index_Loading(self):
        '''
        Output : Dictionary for each record frame
        '''
        try:
            index_file = open('Data/'+str(self.record_num)+'_anno.txt','rb')
        except:
            index_file = open('../Data/'+str(self.record_num)+'_anno.txt','rb')
        Index_dict = {}
        Index_dict['Time'] = []
        Index_dict['Sample'] = []
        Index_dict['Type'] = []

        for each_line in index_file.readlines():
            try :
                A = each_line.split(" ")
                b = [elem for elem in A if elem != ""]
                # b[0] time, b[1] sample idx, b[2] Type
                if int(b[1]) <= self.DyadLength:
                    Index_dict['Time'].append(b[0])
                    Index_dict['Sample'].append(int(b[1]))
                    Index_dict['Type'].append(b[2])
                    # Index_dict.update({'Time' : b[0], 'Sample' : int(b[1]), 'Type' : b[2]})
            except:
                pass

        return Index_dict

    def Segment(self):
        '''
        Output
        - ECG_segment : Dictionary (key : r-index, value : list)
        - ECG_segment_type : Dictionary (Key : r-index, value : type)
        '''
        iter_idx = 0
        Time_domain, ECG_Dyadic_Sample, Dyad_length = self.Data_Load()
        ECG_Segment = {}
        ECG_Segment_Type = {}
        Index_dict = self.Index_Loading()
        self.Each_Type = Index_dict['Type']
        self.R_Locations_Index = Index_dict['Sample']
        for each_r in self.R_Locations_Index:
            if each_r > self.To_the_left and \
                                    each_r + self.To_the_right < Dyad_length:
                # N,V,L,R,a,A
                # if self.Each_Type[iter_idx] == 'N' or self.Each_Type[iter_idx] == 'V' or self.Each_Type[iter_idx] == 'a' or self.Each_Type[iter_idx] == 'L' or self.Each_Type[iter_idx] == 'R' or self.Each_Type[iter_idx] == 'A' or self.Each_Type[iter_idx] == 'F' :
                # if self.Each_Type[iter_idx] == 'N' or self.Each_Type == 'N':
                ECG_Segment.update({each_r: ECG_Dyadic_Sample[range(each_r - self.To_the_left, each_r + self.To_the_right)]})
                ECG_Segment_Type.update({each_r : self.Each_Type[iter_idx]})
            iter_idx += 1

        return ECG_Segment, ECG_Segment_Type

    def KeyNum_Type(self, type):
        '''
        Input
        - Type : N,V,L,R,a,A
        Output
        - R-index key number list of each types
        '''
        Test_Segment_Num = []
        for idx in range(len(self.R_Locations_Index)):
            if self.Each_Type[idx] == type \
                    and self.R_Locations_Index[idx] > 128 \
                    and self.R_Locations_Index[idx] < self.DyadLength-128:
                Test_Segment_Num.append(self.R_Locations_Index[idx])

        return Test_Segment_Num

if __name__ == "__main__":
    RecordNum = 105
    RecordType = 0
    Obj = Data_Preparation(record_num=RecordNum, Record_Type=RecordType)
    Dict_KeyRIdx_ValBeat, Dict_KeyRIdx_ValType = Obj.Segment()
    Dict_Anno = Obj.Index_Loading()

    # RR interval Dict 를 만들자.
    Dict_KeyRIdx_ValRR = dict()
    Dict_KeyRIdx_ValRatio = dict()


    for idx, key in sorted(enumerate(Dict_KeyRIdx_ValType)):
        if idx > 0 :
            # print Dict_Anno['Sample'][idx]
            RRInterval = Dict_Anno['Sample'][idx] - Dict_Anno['Sample'][idx-1]
            Post_RRInterval = Dict_Anno['Sample'][idx+1] - Dict_Anno['Sample'][idx]
            Dict_KeyRIdx_ValRR[key] = RRInterval / float(360)
            Pre_RR = RRInterval / float(360)
            Post_RR = Post_RRInterval / float(360)
            Dict_KeyRIdx_ValRatio[key] = Pre_RR / Post_RR



    # Normal, PVC 를 모아볼까?
    List_Normal = list()
    List_VEB = list()
    List_NormalRatio = list()
    List_VEBRatio = list()

    for idx, key in enumerate(sorted(Dict_KeyRIdx_ValRR)):
        print key, Dict_KeyRIdx_ValRR[key], Dict_KeyRIdx_ValType[key]
        if Dict_KeyRIdx_ValType[key] == 'N':
            List_Normal.append(Dict_KeyRIdx_ValRR[key])
            List_NormalRatio.append(Dict_KeyRIdx_ValRatio[key])
        elif Dict_KeyRIdx_ValType[key] == 'V':
            List_VEB.append(Dict_KeyRIdx_ValRR[key])
            List_VEBRatio.append(Dict_KeyRIdx_ValRatio[key])

    Density_Normal = gaussian_kde(List_Normal)
    Density_VEB = gaussian_kde(List_VEB)
    Density_NormalRatio = gaussian_kde(List_NormalRatio)
    Density_VEBRatio = gaussian_kde(List_VEBRatio)

    Domain = np.linspace(-1,2,200)
    plt.figure()
    plt.plot(Domain, Density_Normal(Domain), 'b', label="Normal")
    plt.plot(Domain, Density_VEB(Domain),'r', label="VEB")
    plt.legend()

    plt.figure()
    plt.plot(Domain, Density_NormalRatio(Domain), 'b', label="Normal")
    plt.plot(Domain, Density_VEBRatio(Domain), 'r', label="VEB")
    # plt.hist(List_NormalRatio)
    # plt.hist(List_VEBRatio)
    plt.legend()


    plt.show()



