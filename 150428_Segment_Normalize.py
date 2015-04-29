# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io
import pywt
import numpy as np

''' Control variable (Global) '''
PVC_Data_Num = [105, 106, 116, 119, 200, 201, 203, 208, 210, 213, 215, 217, 219, 221, 223, 228, 233]
Sampling_rate = 360

datanum = 105
show_record = 1

# INDEX = True
INDEX = False
# PLOT = True
PLOT = False

To_the_left = 128
To_the_right = 128

scale = 4
record_name = ""

if show_record == 0:
    record_name = "MLII"
elif show_record == 1:
    record_name = "V1"

def Data_Loading(datanum):
    mat = scipy.io.loadmat('Data/'+str(datanum)+'_file.mat')
    if show_record == 0:
        ECG = mat['val'][0]
    else:
        ECG = mat['val'][1]
    Dyad_length = 2**(int(np.log(len(ECG)) / np.log(2)))
    ECG_Dyadic_Sample = np.array(ECG[:Dyad_length])
    Time_domain = np.array([x / float(Sampling_rate) for x in range(len(ECG_Dyadic_Sample))])
    return Time_domain, ECG_Dyadic_Sample, Dyad_length

def Index_Loading(datanum, Dyad_length):
    index_file = open('Data/'+str(datanum)+'_anno.txt','rb')
    Index_dict = {}
    Index_dict['Time'] = []
    Index_dict['Sample'] = []
    Index_dict['Type'] = []

    for each_line in index_file.readlines():
        try :
            A = each_line.split(" ")
            b = [elem for elem in A if elem != ""]
            # b[0] time, b[1] sample idx, b[2] Type
            if int(b[1]) <= Dyad_length:
                Index_dict['Time'].append(b[0])
                Index_dict['Sample'].append(int(b[1]))
                Index_dict['Type'].append(b[2])
                # Index_dict.update({'Time' : b[0], 'Sample' : int(b[1]), 'Type' : b[2]})
        except:
            pass

    return Index_dict

def DWT(data, wavelet, scale):
    db8 = pywt.Wavelet(wavelet)
    DWT_result = pywt.wavedec(data=data, mode='per', wavelet=db8, level=scale)
    return DWT_result

def Detail_Coef_Plot_Except_Show(ECG_Segment, ECG_Segment_Type, Test_Segment_num):
    Test_ECG_Segment = ECG_Segment[Test_Segment_num]
    Test_ECG_Segment_Type = ECG_Segment_Type[Test_Segment_num]
    DWT_WCs = DWT(data=Test_ECG_Segment, wavelet='db8', scale=scale)

    if Test_ECG_Segment_Type == 'V':
        return [DWT_WCs[0], DWT_WCs[1], DWT_WCs[2]]
    else:
        return -1

    # DWT_Details = np.concatenate((DWT_WCs[1], DWT_WCs[2]))
    # DWT_Details = np.concatenate((DWT_WCs[0], DWT_Details))
    # # DWT_Details = DWT_WCs[1]
    #
    # if Test_ECG_Segment_Type == 'N':
    #     plt.plot(DWT_Details,'b')
    # elif Test_ECG_Segment_Type == 'V':
    #     plt.plot(DWT_Details,'r')
    #
    # return None

def main():
    ''' Data file and Index file loading '''
    Time_domain, ECG, Dyad_length = Data_Loading(datanum=datanum)
    Index_dict = Index_Loading(datanum=datanum, Dyad_length=Dyad_length)
    R_Locations_Index = Index_dict['Sample']
    Each_Type = Index_dict['Type']
    ECG_Segment = {}
    ECG_Segment_Type = {}
    iter_idx = 0
    for each_r in R_Locations_Index:
        if each_r > To_the_left and each_r + To_the_right < Dyad_length:
            ECG_Segment.update({each_r: ECG[range(each_r - To_the_left, each_r + To_the_right)]})
            ECG_Segment_Type.update({each_r : Each_Type[iter_idx]})
        iter_idx += 1

    Test_Segment_Num = []
    for idx in range(len(R_Locations_Index)) :
        if Each_Type[idx] == 'V' and R_Locations_Index[idx] > 128 and R_Locations_Index[idx] < Dyad_length-128:
            Test_Segment_Num.append(R_Locations_Index[idx])

    DWT_BOX = []
    for test_seg in Test_Segment_Num:
        Keunseo = Detail_Coef_Plot_Except_Show(ECG_Segment=ECG_Segment, ECG_Segment_Type=ECG_Segment_Type, Test_Segment_num=test_seg)
        if Keunseo != -1:
            DWT_BOX.append(Keunseo)

    ''' Normalize '''
    DWT_BOX = np.matrix(DWT_BOX).T
    DWT_BOX = np.asarray(DWT_BOX)

    APPROX_BOX = DWT_BOX[0]
    DWT4_BOX = DWT_BOX[1]
    DWT3_BOX = DWT_BOX[2]

    APPROX_Col_Std = np.std(APPROX_BOX, axis=0)
    DWT4_Col_Std = np.std(DWT4_BOX, axis=0)
    DWT3_Col_Std = np.std(DWT3_BOX, axis=0)

    APPROX_Mean = np.mean(APPROX_BOX)
    DWT4_Mean = np.mean(DWT4_BOX)
    DWT3_Mean = np.mean(DWT3_BOX)





    # for idx in range(len(APPROX_BOX)):
    #     APPROX_BOX[idx] -= APPROX_Mean[idx]
    #     APPROX_BOX[idx] /= APPROX_Col_Std[idx]
    #
    #     DWT4_BOX[idx] -= DWT4_Mean[idx]
    #     DWT4_BOX[idx] /= DWT4_Col_Std[idx]
    #
    #     DWT3_BOX[idx] -= DWT3_Mean[idx]
    #     DWT3_BOX[idx] /= DWT3_Col_Std[idx]
    #
    # Normalized_DWT_BOX = \
    #     np.concatenate(APPROX_BOX, DWT4_BOX)
    # Normalized_DWT_BOX = \
    #     np.concatenate(Normalized_DWT_BOX, DWT3_BOX)
    #
    # plt.plot(Normalized_DWT_BOX)
    # plt.show()




    # DWT_BOX = list(DWT_BOX)
    # print len(DWT_BOX)

if __name__ == "__main__":
    main()
