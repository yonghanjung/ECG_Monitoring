import numpy as np

from methods_noSDA import Loading_ECG
from methods_noSDA import Loading_R_Peak_and_Label
from methods_noSDA import Segmenting_ECG_Beat
from methods_noSDA import Wavelet_Transformation
from methods_noSDA import Computing_Cov
from methods_noSDA import Constructing_T2_Stat
from methods_noSDA import Computing_UCL
from methods_noSDA import Evaluating_Performance_SPM

''' ECG record number '''
LongTerm_idx = [14046, 14134, 14149, 14157, 14172, 14184, 15814] # sampling rate = 128.0
INCART_idx = range(301,376) # 301 - 375
MITBIH_idx = [105, 106, 108, 109, 114, 118, 119, 200, 202, 203, 205, 208, 210, 213, 214, 215, 219, 221, 223, 228, 233] # 22

class Main:
    def __init__(self, record_idx, alpha, SDA_L1_penalty, SDA_L2_penalty):
        ''' ECG label '''
        AAMI_Normal = ['N','L','R','e','j'] # Those labels in MIT-BIH are considered as normal in AAMI recommended practice
        AAMI_PVC = ['V','E'] # Those label in MIT-BIH are considered as PVC in AAMI recommended practice

        ''' Control variables '''
        # alpha = 1-0.999999998026825 # 6-sigma under normal distribution assumption
        # alpha = 1-0.9927 # 3-sigma under normal distribution assumption
        # alpha = 1-0.99 # 3-sigma under normal distribution assumption
        # alpha = 0.005
        time_training = 300 # seconds (= Initial 5 minutes)

        # SDA_L1_penalty = 4.
        # SDA_L2_penalty = 1.


        ''' 1. Loading ECG records and segmenting by beats '''
        sampling_rate_MITBIH = 360. # MIT BIH
        sampling_rate_INCART = 257. # INCART
        sampling_rate_LongTerm = 128. # LONG
        if record_idx in MITBIH_idx:
            data_name = 'MITBIH'
            sampling_rate = sampling_rate_MITBIH
        elif record_idx in INCART_idx:
            sampling_rate = sampling_rate_INCART
            data_name = 'INCART'
        elif record_idx in LongTerm_idx:
            data_name = 'LongTerm'
            sampling_rate = sampling_rate_LongTerm
        print ""
        print("1. Loading ECG record " + str(record_idx) + " from " + data_name + "...")

        time_domain, ECG_record = Loading_ECG(record_idx,sampling_rate)
        R_peak_and_label = Loading_R_Peak_and_Label(record_idx)
        dict_ECG_beat_segmented, dict_ECG_beat_label = Segmenting_ECG_Beat(ECG_record,R_peak_and_label) # key: R peak index

        ''' 2. Constructing training set (initial 5min are segmented as training set) '''
        print("2. Constructing training set...")
        dict_train_ECG = {RIdx : dict_ECG_beat_segmented[RIdx] for RIdx in dict_ECG_beat_segmented.keys() if RIdx < sampling_rate * time_training}
        dict_train_ECG_normal = {RIdx : dict_ECG_beat_segmented[RIdx] for RIdx in dict_ECG_beat_segmented.keys() if RIdx < sampling_rate * time_training and dict_ECG_beat_label[RIdx] in AAMI_Normal}
        dict_train_ECG_PVC = {RIdx : dict_ECG_beat_segmented[RIdx] for RIdx in dict_ECG_beat_segmented.keys() if RIdx < sampling_rate * time_training and dict_ECG_beat_label[RIdx] in AAMI_PVC}
        dict_train_label = {RIdx : dict_ECG_beat_label[RIdx] for RIdx in dict_ECG_beat_label.keys() if RIdx < sampling_rate * time_training}
        dict_test_ECG = {RIdx : dict_ECG_beat_segmented[RIdx] for RIdx in dict_ECG_beat_label.keys() if RIdx > sampling_rate * time_training}
        dict_test_label = {RIdx : dict_ECG_beat_label[RIdx] for RIdx in dict_ECG_beat_label.keys() if RIdx > sampling_rate * time_training}

        ''' 3. Applying wavelet transformation to each ECG beats in training and test set '''
        print("3. Applying wavelet transformation...")
        dict_train_wc = Wavelet_Transformation(dict_train_ECG)
        dict_train_wc_normal = Wavelet_Transformation(dict_train_ECG_normal)
        dict_train_wc_PVC = Wavelet_Transformation(dict_train_ECG_PVC)
        dict_test_wc = Wavelet_Transformation(dict_test_ECG)

        # Computing average of wavelet coefficients of normal ECG beats in training set
        average_train_wc_normal = np.zeros(256)
        number_train_normal = 0
        for idx, key in enumerate(sorted(dict_train_wc_normal)):
            average_train_wc_normal += np.array(dict_train_wc_normal[key])
            number_train_normal += 1
        average_train_wc_normal /= float(number_train_normal)
        average_train_wc_normal = np.reshape(average_train_wc_normal, (1,len(average_train_wc_normal)))


        ''' 4. Constructing covariance matrix '''
        print("4. Constructing Covariance matrix...")
        cov_matrix = Computing_Cov(dict_train_wc_normal)

        ''' 5. Computing T2 statistics '''
        print("5. Computing T2 statistics...")
        dict_test_T2stat = Constructing_T2_Stat(average_train_wc_normal,cov_matrix,dict_test_wc)
        UCL = Computing_UCL(len(dict_train_wc_normal),256,alpha)
        print len(dict_train_wc_normal), UCL

        ''' 6. Evaluating accuracy by counting right and wrongly classified beats '''
        print("Evaluating of wavelet-based SPM for record number " + str(record_idx) + " in " + data_name)
        DictInt_Accuracy = Evaluating_Performance_SPM(dict_test_T2stat, dict_test_label,UCL,AAMI_Normal,AAMI_PVC)
            # for idx,key in enumerate(sorted(DictInt_Accuracy)):
            #     print key, DictInt_Accuracy[key]
            # print len(non_zero_elem)
        self.DictInt_Accuracy = DictInt_Accuracy
