# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import scipy.io
import pywt
import numpy as np

''' Control variable (Global) '''
PVC_Data_Num = [105, 106, 116, 119, 201, 203, 208, 210, 213, 215, 217, 219, 221, 223, 228, 233]
Sampling_rate = 360

datanum = 119
show_record = 0
wavelet_basis = 'db8'

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
    mat = scipy.io.loadmat('../Data/'+str(datanum)+'_file.mat')
    if show_record == 0:
        ECG = mat['val'][0]
    else:
        ECG = mat['val'][1]
    Dyad_length = 2**(int(np.log(len(ECG)) / np.log(2)))
    ECG_Dyadic_Sample = np.array(ECG[:Dyad_length])
    Time_domain = np.array([x / float(Sampling_rate) for x in range(len(ECG_Dyadic_Sample))])
    return Time_domain, ECG_Dyadic_Sample, Dyad_length

def Index_Loading(datanum, Dyad_length):
    index_file = open('../Data/'+str(datanum)+'_anno.txt','rb')
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

def Orig_Plot_Except_Show(ECG_Segment, ECG_Segment_Type, Test_Segment_num):
    Test_ECG_Segment = ECG_Segment[Test_Segment_num]
    Test_ECG_Segment_Type = ECG_Segment_Type[Test_Segment_num]
    DWT_WCs = DWT(data=Test_ECG_Segment, wavelet=wavelet_basis, scale=scale)
    # DWT_Details = np.concatenate((DWT_WCs[1], DWT_WCs[2]))
    # if Test_ECG_Segment_Type == 'N':
        # plt.plot(Test_ECG_Segment, 'b')
    # elif Test_ECG_Segment_Type == 'V':
        # plt.plot(Test_ECG_Segment, 'r')

    return None

def Detail_Coef_Plot_Except_Show(ECG_Segment, ECG_Segment_Type, Test_Segment_num):
    Test_ECG_Segment = ECG_Segment[Test_Segment_num]
    Test_ECG_Segment_Type = ECG_Segment_Type[Test_Segment_num]
    DWT_WCs = DWT(data=Test_ECG_Segment, wavelet=wavelet_basis, scale=scale)
    DWT_Details = np.concatenate((DWT_WCs[1], DWT_WCs[2]))
    DWT_Details = np.concatenate((DWT_WCs[0], DWT_Details))
    # DWT_Details = DWT_WCs[1]

    if Test_ECG_Segment_Type == 'N':
        plt.plot(DWT_Details,'bo')
    elif Test_ECG_Segment_Type == 'V':
        plt.plot(DWT_Details,'ro')

    return None

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
    for idx in range(len(R_Locations_Index)):
        if Each_Type[idx] == 'N' and R_Locations_Index[idx] > 128 and R_Locations_Index[idx] < Dyad_length-128:
            Test_Segment_Num.append(R_Locations_Index[idx])

    # print Test_Segment_Num
    plt.figure()
    plt.grid()
    plt.title(str(datanum) + " " + record_name +  " Original Plot, Blue : Normal vs Red : PVC")

    for test_seg in Test_Segment_Num:
        Orig_Plot_Except_Show(ECG_Segment=ECG_Segment, ECG_Segment_Type=ECG_Segment_Type, Test_Segment_num=test_seg)

    Test_Segment_Num = []
    for idx in range(len(R_Locations_Index)) :
        if Each_Type[idx] == 'V' and R_Locations_Index[idx] > 128 and R_Locations_Index[idx] < Dyad_length-128:
            Test_Segment_Num.append(R_Locations_Index[idx])

    for test_seg in Test_Segment_Num:
        Orig_Plot_Except_Show(ECG_Segment=ECG_Segment, ECG_Segment_Type=ECG_Segment_Type, Test_Segment_num=test_seg)

    plt.figure()
    plt.grid()
    plt.title(str(datanum) + " " + record_name + " DWT coefs plot, Blue : Normal vs Red : PVC")

    Test_Segment_Num = []
    for idx in range(len(R_Locations_Index)):
        if Each_Type[idx] == 'N' and R_Locations_Index[idx] > 128 and R_Locations_Index[idx] < Dyad_length-128:
            Test_Segment_Num.append(R_Locations_Index[idx])

    for test_seg in Test_Segment_Num:
        Detail_Coef_Plot_Except_Show(ECG_Segment=ECG_Segment, ECG_Segment_Type=ECG_Segment_Type, Test_Segment_num=test_seg)

    Test_Segment_Num = []
    for idx in range(len(R_Locations_Index)):
        if Each_Type[idx] == 'V' and R_Locations_Index[idx] > 128 and R_Locations_Index[idx] < Dyad_length-128:
            Test_Segment_Num.append(R_Locations_Index[idx])

    for test_seg in Test_Segment_Num:
        Detail_Coef_Plot_Except_Show(ECG_Segment=ECG_Segment, ECG_Segment_Type=ECG_Segment_Type, Test_Segment_num=test_seg)

    plt.show()


    ''' Index Dict '''
    if INDEX:
        for tp, idx in zip(Index_dict['Sample'], Index_dict['Type']):
            print tp, idx



    ''' PLOT '''
    if PLOT:
        plt.figure()
        plt.title(str(Test_Segment_Num))
        plt.plot(ECG_Segment[Test_Segment_Num])
        print len(ECG_Segment[Test_Segment_Num])
        # plt.plot(ECG)
        # plt.plot(Index_dict['Sample'],ECG[Index_dict['Sample']], 'ro')
        plt.show()




if __name__ == "__main__":
    main()
