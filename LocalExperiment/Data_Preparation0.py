'''
150501
Goal
- Library for data preparation 

Possible records are =
[105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 217, 219, 221, 223, 228, 233]
'''

# from Data_Load import Data_Loading
from Data_Load import Data_Loading
# import Data_Load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Data_Preparation:
    def __init__(self, record_num, Record_Type=0):
        # Declaration of variables
            ## Type : 0 or 1
        self.record_num = record_num # File name
        # self.Sampling_rate = 360
        self.Sampling_rate =  50
        self.DyadLength = 2 ** (int(np.log2(self.Sampling_rate * 60 * 30)))
        self.show_record = Record_Type

        self.To_the_left = 128
        self.To_the_right = 128

        scale = 4
        record_name = ""

        self.Time_domain, self.ECG_Dynamic_Sample, self.Dyad_length = Data_Loading(self.record_num,record_type=self.show_record, Sampling_rate=self.Sampling_rate).Data_Loading()


    def Data_Load(self):
        Time_domain, ECG_Dyadic_Sample, Dyad_length = \
            Data_Loading(self.record_num,record_type=self.show_record, Sampling_rate=self.Sampling_rate).Data_Loading()
        return Time_domain, ECG_Dyadic_Sample, Dyad_length

    # def Data_Loading(self):
    #     '''
    #     Goal : Load the ECG mat (MIT BIH)
    #     Output
    #     - Time domain : Array
    #     - ECG_Dyadic_Sample = np.array
    #     - DyadLenth = integer
    #     '''
    #     # try:
    #     #     mat = scipy.io.loadmat('Data/'+str(self.datanum)+'_file.mat')
    #     file = pd.read_csv(self.record_num)
    #     # except:
    #     #     # mat = scipy.io.loadmat('../Data/'+str(self.datanum)+'_file.mat')
    #     #     print "error"
    #
    #     ''' KKseo '''
    #     # "accelY","accelZ","accelMag","gyroX","risk","step"
    #     accelY = file["accelY"]
    #
    #     accelZ = file["accelZ"]
    #     accelMag = file["accelMag"]
    #     gyroX = file["gyroX"]
    #     risk = file["risk"]
    #     step = file["step"]
    #
    #     length = len(accelY)
    #     Time_domain = np.array([x / float(self.Sampling_rate) for x in range(len(accelY))])
    #     ############
    #
    #     # if self.record_type == 0:
    #     #     ECG = mat['val'][0]
    #     # else:
    #     #     ECG = mat['val'][1]
    #     # Dyad_length = 2**(int(np.log(len(ECG)) / np.log(2)))
    #     # ECG_Dyadic_Sample = np.array(ECG[:Dyad_length])
    #     # Time_domain = np.array([x / float(self.sampling_rate) for x in range(len(ECG_Dyadic_Sample))])
    #     return Time_domain, accelY,accelZ,accelMag,gyroX,risk,step

    # def Data_Load(self):
    #     # dataLoad = Data_Loading(self.record_num,self.show_record,self.Sampling_rate)
    #     dataLoad = Data_Load.Data_Loading(self.record_num,self.show_record,self.Sampling_rate)
    #     Time_domain, accelY, accelZ, accelMag, gyroX, risk, step = dataLoad.Data_Loading()
    #     return Time_domain, accelY, accelZ, accelMag, gyroX, risk, step

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
    #
    #
    # def KKSeo_Segment(self):
    #     Time_domain, accelY, accelZ, accelMag, gyroX, risk, step  = self.Data_Loading()
    #     # print accelY
    #     Segment = {}
    #     Segment_Type = {}
    #
    #     StepIdx = step[step==4].keys()
    #     StepStart = list()
    #     StepEnd = list()
    #     for i in range(len(StepIdx)):
    #         if np.mod(i,2) == 0:
    #             StepStart.append(StepIdx[i])
    #         else:
    #             StepEnd.append(StepIdx[i])
    #     print StepStart
    #     print StepEnd
    #
    #
    #     # TimeSeriesList = [accelY,accelZ,accelMag,gyroX]
    #
    #     StepCount = 0
    #     StepSwitch = False
    #
    #
    # # for timeseries in TimeSeriesList
    #     for i in range(len(accelY)):
    #         if i in StepStart:
    #             StepSwitch = True
    #             StepCount += 1
    #             Segment[StepCount] = list( )
    #             Segment_Type[StepCount] = list()
    #         if i in StepEnd:
    #             StepSwitch = False
    #
    #         if StepSwitch:
    #             # print accelY[i]
    #             Segment[StepCount].append(accelY[i])
    #             Segment_Type[StepCount].append(risk[i])
    #
    #         # if StepStatus == 4 and StepSwitch == False: # Step start!
    #         #     StepSwitch = True
    #         #     StepCount += 1
    #         #     Segment[StepCount] = list()
    #         #     Segment_Type[StepCount] = list()
    #         # if StepStatus == 4 and np.mod(StepCount,2) == 1: # Step end!
    #         #     StepSwitch = False
    #         #     pass
    #         #
    #         # if StepStatus != 4 and np.mod(StepCount,2) == 1:
    #         #     Segment[StepCount].append(accelY[i])
    #         #     Segment_Type[StepCount].append(risk[i])
    #
    #     return Segment,Segment_Type




    def Segment(self):
        '''
        Output
        - ECG_segment : Dictionary (key : r-index, value : list)
        - ECG_segment_type : Dictionary (Key : r-index, value : type)
        '''
        iter_idx = 0
        ECG_Segment = {}
        ECG_Segment_Type = {}
        Index_dict = self.Index_Loading()
        self.Each_Type = Index_dict['Type']
        self.R_Locations_Index = Index_dict['Sample']
        for each_r in self.R_Locations_Index:
            if each_r > self.To_the_left and \
                                    each_r + self.To_the_right < self.Dyad_length:
                # N,V,L,R,a,A
                if self.Each_Type[iter_idx] == 'N' or self.Each_Type[iter_idx] == 'V' or self.Each_Type[iter_idx] == 'a' or self.Each_Type[iter_idx] == 'L' or self.Each_Type[iter_idx] == 'R' or self.Each_Type[iter_idx] == 'A' or self.Each_Type[iter_idx] == 'F' :
                # if self.Each_Type[iter_idx] == 'N' or self.Each_Type == 'N':
                    ECG_Segment.update({each_r: self.ECG_Dyadic_Sample[range(each_r - self.To_the_left, each_r + self.To_the_right)]})
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
    FileName = "zone3_downhill_hengame_labeled.csv"
    dataPrep = Data_Preparation(FileName)
    Seg,SegType = dataPrep.KKSeo_Segment()
    for idx, key in enumerate(sorted(Seg)):
        print len(Seg[key]), SegType[key]

    # RecordNum = 119
    # RecordType = 0
    # OBJ = Data_Preparation(record_num = RecordNum, Record_Type = RecordType)
    # Time_domain, ECG_Dyadic_Sample, Dyad_length = OBJ.Data_Load()
    #
    # fig, ax = plt.subplots()
    # # plt.title("ECG beats")
    # ax.plot(Time_domain[5800:6250], ECG_Dyadic_Sample[5800:6250],'b', label="Normal")
    # plt.setp(ax.get_yticklabels(), visible = False)
    # plt.setp(ax.get_xticklabels(), visible = False)
    # # plt.xlabel(visible=False)
    # #
    # # fig, ax = plt.subplots(2,1,figsize=(10,5))
    # # fig.canvas.set_window_title('Performance comparison')
    # # plt.subplots_adjust(left=0.075, right = 0.95, top = 0.95, bottom = 0.05)
    # # # ax[0].plot(Time_domain[2100:2429],ECG_Dyadic_Sample[2100:2429],'b', label="Normal")
    # # # ax[0].plot(Time_domain[2430:2860],ECG_Dyadic_Sample[2430:2860],'r--', label="PVC")
    # # # ax[0].plot(Time_domain[2861:3200],ECG_Dyadic_Sample[2861:3200],'b', label="Normal")
    # #
    # # # ax[1].plot(Time_domain[5800:6690], ECG_Dyadic_Sample[5800:6690],'b', label="Normal")
    # # ax[1].plot(Time_domain[5800:6100], ECG_Dyadic_Sample[5800:6100],'b', label="Normal")
    # #
    # # # plt.setp(ax[0].get_yticklabels(),visible=False)
    # # plt.setp(ax[1].get_yticklabels(),visible=False)
    # #
    # # plt.tight_layout()
    # # ax[0].legend(loc="best")
    # # ax[1].legend(loc="best")
    #
    # plt.show()



