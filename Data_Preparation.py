'''
150501
Goal
- Library for data preparation 

Possible records are =
[105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 217, 219, 221, 223, 228, 233]
'''

from Data_Load import Data_Loading
import numpy as np

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

    # def Segment(self,TimeSeries, StepData):
    #     for i in range(len(StepData)):
    #         StepStatus = StepData[i]


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
                if self.Each_Type[iter_idx] == 'N' or self.Each_Type[iter_idx] == 'L' or \
                                self.Each_Type[iter_idx] == 'R' or self.Each_Type[iter_idx] == 'e' or \
                                self.Each_Type[iter_idx] == 'j' or self.Each_Type[iter_idx] == 'A' or \
                                self.Each_Type[iter_idx] == 'a' or self.Each_Type[iter_idx] == 'S' or \
                                self.Each_Type[iter_idx] == 'V' or self.Each_Type[iter_idx] == 'E' or \
                                self.Each_Type[iter_idx] == 'F' :
                    ECG_Segment.update({each_r: ECG_Dyadic_Sample[range(each_r - self.To_the_left, each_r + self.To_the_right)]})
                    ECG_Segment_Type.update({each_r : self.Each_Type[iter_idx]})
            iter_idx += 1

        return ECG_Segment, ECG_Segment_Type

    def KeyNum_Type(self, type):
        '''
        Input
        - Type : N, V
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




