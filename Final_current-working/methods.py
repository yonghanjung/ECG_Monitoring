import numpy as np
import scipy.io
import pywt
from scipy.stats import f
from scipy.stats import t
from Class.Class_SDA import SDA

def Loading_ECG(ECG_record_number,sampling_rate):
    '''
    Loading the ECG record
    :param: record_number
    :return: time_domain (seconds) / ECG_record
    '''
    ECG_file_name = '../Data/'+str(ECG_record_number)+'_file.mat'
    ECG_file = scipy.io.loadmat(ECG_file_name)
    time_index = np.array([x / float(sampling_rate) for x in range(len(ECG_file['val'][0]))])
    ECG_record = ECG_file['val'][0]
    return time_index, ECG_record

def Loading_R_Peak_and_Label(ECG_record_number):
    '''
    Loading annotation file (R_peak_index and labels of ECG beats are in annotation file)
    :param: record_index
    :return: dictionary (key: 'Time','Sample','Type' / val: time_(sec)_R_peak, sample_index_R_peak, label_beat_R_peak
    '''
    try:
        annotation_file = open('Data/'+str(ECG_record_number)+'_anno.txt','rb')
    except:
        annotation_file = open('../Data/'+str(ECG_record_number)+'_anno.txt','rb')
    dict_annotation = dict()
    dict_annotation['Time'] = list() # time index (sec) for R_peak
    dict_annotation['Sample'] = list() # sample index for R_peak
    dict_annotation['Type'] = list() # label for ECG beat containing R_peak

    for each_line in annotation_file.readlines():
        try :
            A = each_line.split(" ")
            b = [elem for elem in A if elem != ""] # b[0] time, b[1] sample idx, b[2] Type
            # if int(b[1]) <= DyadLength:
            dict_annotation['Time'].append(b[0])
            dict_annotation['Sample'].append(int(b[1]))
            dict_annotation['Type'].append(b[2])
        except:
            pass
    return dict_annotation

def SoftThreshold(X, Threshold):
    '''
    Implementing the soft threshold for the universal thresholding
    :param X: Signal to be thresheld
    :param thresholding_value
    :return: thesheld signal X
    '''
    ResultList = list()
    for elem in X:
        if elem < 0:
            sgn = -1
        else:
            sgn = 1
        val = np.abs(elem) - Threshold
        # val *= sgn
        if val > 0:
            val *= sgn
            ResultList.append(val)
        else:
            # print "ho"
            ResultList.append(0.0)
    return np.asarray(ResultList)

def Segmenting_ECG_Beat(ECG_record, Index_dict):
    '''
    Segment ECG record beat by beat
    :param ECG_record: ECG record
    :param Index_dict: dictionary providing information about ECG beat
    :return dict_ECG_beat (key: R_peak_index, val: ECG_beat)
    :return dict_label_beat (key: R_peak_index, val: labe_ECG_beat)
    '''
    # one ECG beat = 128 points to the left and right from R_peak
    distance_to_left = 128 # 128 points to the left from R_peak
    distance_to_right = 128 # 128 points to the right from R_peak

    AAMI_Normal = ['N','L','R','e','j'] # Those label in MIT-BIH are considered as Normal in AAMI
    AAMI_PVC = ['V','E'] # Those label in MIT-BIH are considered as Normal in AAMI
    AAMI_Others = ['A','a','J','S']
    AAMI_total_label = AAMI_Normal + AAMI_PVC + AAMI_Others

    dict_ECG_beat = dict()
    dict_label_beat = dict()
    list_label = Index_dict['Type']
    list_R_peak_index = Index_dict['Sample']

    iter_idx = 0
    for each_r in list_R_peak_index:
        if each_r > distance_to_left and each_r + distance_to_right < len(ECG_record):
            ECG_beat = ECG_record[range(each_r - distance_to_left, each_r + distance_to_right)]
            if list_label[iter_idx] in AAMI_total_label:
                dict_ECG_beat[each_r] = ECG_beat
                label_of_beat = list_label[iter_idx]
                if label_of_beat in AAMI_Normal:
                    AAMI_label = 'N'
                elif label_of_beat in AAMI_PVC:
                    AAMI_label = 'V'
                elif label_of_beat in AAMI_Others:
                    AAMI_label = 'S'
                dict_label_beat[each_r] = AAMI_label
        iter_idx += 1
    return dict_ECG_beat, dict_label_beat

def Wavelet_Transformation(dict_ECG_beat):
    '''
    Implementing discrete wavelet transformation to each ECG beat after denoising using universal threshold.
    :param dict_ECG_beat: dictionary of ECG beat (key: R_peak_index)
    :return denoised_coefs: dictionary containing denoised wavelet coefficients of each ECG beat
    '''
    dict_wc = dict()
    WaveletBasis = 'db8'
    DecompLevel = 4
    for idx,key in enumerate(sorted(dict_ECG_beat)):
        # wavelet transformation
        dict_wc[key] = pywt.wavedec(data=dict_ECG_beat[key],wavelet=pywt.Wavelet(WaveletBasis),mode='per',level=DecompLevel)
        # universal thresholding
        WaveletThreshold = np.sqrt(2 * np.log(256)) * (np.median(np.abs(np.array(dict_wc[key][DecompLevel]) - np.median(dict_wc[key][DecompLevel]))) / 0.6745)
        wavelet_coefs = np.concatenate([dict_wc[key][0],dict_wc[key][1],dict_wc[key][2],dict_wc[key][3],dict_wc[key][4]])
        wavelet_coefs = SoftThreshold(wavelet_coefs, WaveletThreshold)
        dict_wc[key] = wavelet_coefs
    return dict_wc

def Constructing_SDA_Vector(DictArray_TrainWCNormal,DictArray_TrainWCPVC,a,b):
    '''
    Constructing sparse discriminant vector as suggested in Sparse Discriminant Analysis (Technometrics) by Clemmensen (2012)
    :param DictArray_TrainWCNormal: wavelet coefficients of normal ECG beat in training set
    :param DictArray_TrainWCPVC: wavelet coefficients of PVC ECG beat in training set
    :param a: L1 penalty parameter of SDA algorithm
    :param b: L2 penalty parameter of SDA algorithm
    :return: sparse discriminant vector
    '''

    DictArrayMatrix_for_SDA = dict() #0: Normal, 1: PVC
    DictArrayMatrix_for_SDA[0] = list()
    DictArrayMatrix_for_SDA[1] = list()

    for idx, key in enumerate(sorted(DictArray_TrainWCNormal)):
        DictArrayMatrix_for_SDA[0].append(DictArray_TrainWCNormal[key])
    for idx, key in enumerate(sorted(DictArray_TrainWCPVC)):
        DictArrayMatrix_for_SDA[1].append(DictArray_TrainWCPVC[key])
    DictArrayMatrix_for_SDA[0] = np.array(DictArrayMatrix_for_SDA[0])
    DictArrayMatrix_for_SDA[1] = np.array(DictArrayMatrix_for_SDA[1])

    alpha = a+b
    try:
        L1Ratio = a / (a+b)
    except:
        L1Ratio = 0.0

    obj_SDA = SDA(dict_train=DictArrayMatrix_for_SDA, Flt_Lambda=alpha, Flt_L1=L1Ratio)
    sparse_discriminant_vector = obj_SDA.sparse_discriminant_vector
    nonzero_elem = obj_SDA.non_zero_elem
    return sparse_discriminant_vector, nonzero_elem

def Projecting_Lower_Dimensional_Vec(sparse_discriminant_vector,dict_wc):
    '''
    Implementing low dimensional projection of vector using sparse discriminant vector
    :param sparse_discriminant_vector: sparse discriminant vector constructed from 'Constructing_SDA_vector' function
    :param dict_wc: dictionary containing wavelet coefficients of each ECG beats (key: R_peak_index)
    :return: dictionary of low dimensional projected wavelet coefficients of each ECG beats (key: R_peak_index)
    '''
    dict_low_dim_projected = dict()

    for idx, key in enumerate(sorted(dict_wc)):
        # 1 by 64
        wavelet_coefs = np.reshape(dict_wc[key], (len(dict_wc[key]),1 ))
        low_dim_projected = np.dot(np.transpose(sparse_discriminant_vector), wavelet_coefs)
        low_dim_projected = np.squeeze(np.asarray(low_dim_projected))
        low_dim_projected = float(low_dim_projected)
        dict_low_dim_projected[key] = low_dim_projected
    return dict_low_dim_projected

def Projecting_Low_Dimensional_Cov(sparse_discriminant_vector, dict_train_normal_wc):
    '''
    Implementing low dimensional projection of covariance matrix using sparse discriminant vector
    :param sparse_discriminant_vector: sparse discriminant vector constructed from 'Constructing_SDA_vector' function
    :param dict_train_normal_wc: dictionary containing wavelet coefficients of ECG beats in training set (key: R_peak_index)
    :return:
    '''
    # 1. Computing covariance matrix of wavelet coefficients of normal ECG beats in training set
    mat_wc_normal = list()
    for idx, key in enumerate(sorted(dict_train_normal_wc)):
        wc_train_normal = dict_train_normal_wc[key]
        mat_wc_normal.append(wc_train_normal)
    mat_wc_normal = np.array(mat_wc_normal)
    mat_Cov = np.var(mat_wc_normal, axis=0)
    mat_Cov = np.diag(mat_Cov)
    sparse_discriminant_vector = np.reshape(sparse_discriminant_vector,(len(sparse_discriminant_vector),1))
    low_dim_projected_Cov = np.dot(np.dot(np.transpose(sparse_discriminant_vector), mat_Cov), sparse_discriminant_vector)
    return np.ravel(low_dim_projected_Cov)

def Constructing_T2_Stat(projected_average_wc_normal, projected_Cov_wc_normal, dict_test_projected_wc):
    '''
    Compute T2 statistics from low dimensional projected wavelet coefficients, using projected average and covariance matrix
    :param projected_average_wc_normal: low dimensional projected average of wavelet coefficients in normal ECG beats in training set
    :param projected_Cov_wc_normal: low dimensional projected covariance matrix of wavelet coefficients in normal ECG beats in training set
    :param dict_test_projected_wc: low dimensional projected vector of wavelet coefficients in test dataset
    :return: dictionary of T2 statistics corresponding to ECG beats in test set (key: R_peak)
    '''

    dict_test_T2stat = dict()
    for idx, key in sorted(enumerate(dict_test_projected_wc)):
        wc_test_projected = dict_test_projected_wc[key] # 1 Dim
        wc_test_projected = np.array(wc_test_projected)
        wc_test_centered_projected = np.array(wc_test_projected - projected_average_wc_normal)

        T2_stat = wc_test_centered_projected * (projected_Cov_wc_normal**(-1)) * wc_test_centered_projected.T
        T2_stat = np.squeeze(np.asarray(T2_stat))
        # T2_stat = np.sqrt(T2_stat)
        dict_test_T2stat[key] = T2_stat
    return dict_test_T2stat

def Computing_UCL(num_train_beats, alpha):
    '''
    Computing upper control limit (UCL)
    :param num_train_beats: number of ECG beats in training set
    :param alpha: predefined alpha level (0.01 in the paper)
    :return: UCL value
    '''
    dim_projected = 1

    # print num_train_beats, t.ppf(1-alpha,num_train_beats-1), np.sqrt((dim_projected*((num_train_beats-1)*(num_train_beats+1))  / (num_train_beats*(num_train_beats-dim_projected)) * f.ppf(1-alpha, dim_projected, num_train_beats-dim_projected)))
    # return t.ppf(1-alpha,num_train_beats-1)
    return (dim_projected*((num_train_beats-1)**2) * f.ppf(1-alpha, dim_projected, num_train_beats-dim_projected)) / (num_train_beats*(num_train_beats-dim_projected))


def Evaluating_Performance_SPM(dict_test_T2stat, dict_test_label, UCL_val, AAMI_Normal, AAMI_PVC):
    '''
    Counting right and wrong classification result for evaluating performance of wavelet-based SPM
    :param dict_test_T2stat: dictionary of T2 statistics corresponding to test set (key: R_peak_index)
    :param dict_test_label: dictionary of label of ECG beats in test set (key: R_peak_index)
    :param UCL_val: UCL value computed from the function 'Computing_UCL'
    :param AAMI_Normal: list of normal label ['N','L','R','e','j'] # Those label in MIT-BIH are considered as Normal in AAMI
    :param AAMI_PVC: list of PVC label # AAMI_PVC = ['V','E'] # Those label in MIT-BIH are considered as Normal in AAMI
    :return: dictionary of counting results
    '''
    counting_tested_points = 0
    counting_normal_as_PVC = 0
    counting_normal_as_normal = 0
    counting_PVC_as_normal = 0
    counting_PVC_as_PVC = 0

    dict_evaluating_performance = dict()

    for idx, key in enumerate(sorted(dict_test_T2stat)):
        counting_tested_points += 1
        if dict_test_label[key] in AAMI_Normal:
            if dict_test_T2stat[key] < UCL_val:
                counting_normal_as_normal += 1
            elif dict_test_T2stat[key] > UCL_val:
                counting_normal_as_PVC += 1

        elif dict_test_label[key] in AAMI_PVC :
            if dict_test_T2stat[key] < UCL_val:
                counting_PVC_as_normal += 1
            elif dict_test_T2stat[key] > UCL_val:
                counting_PVC_as_PVC += 1

    dict_evaluating_performance['Normal as PVC'] = counting_normal_as_PVC
    dict_evaluating_performance['Normal as Normal'] = counting_normal_as_normal
    dict_evaluating_performance['PVC as Normal'] = counting_PVC_as_normal
    dict_evaluating_performance['PVC as PVC'] = counting_PVC_as_PVC
    return dict_evaluating_performance


def Evaluating_Performance_SVM_NN(labeled_by_classifiers, dict_test_label):
    '''
    Counting right and wrong classification result for evaluating performance of SVM and neural network
    :param labeled_by_classifiers: label determined by classifiers (SVM, NN)
    :param dict_test_label: dictionary of test label
    :return: dictionary of counting results
    '''
    dict_evaluating_performance = dict()
    dict_evaluating_performance['Normal as PVC'] = 0
    dict_evaluating_performance['Normal as Normal'] = 0
    dict_evaluating_performance['PVC as Normal'] = 0
    dict_evaluating_performance['PVC as PVC'] = 0

    for idx,key in enumerate(sorted(labeled_by_classifiers)):
        if dict_test_label[key] == 'N' and labeled_by_classifiers[key] == 'N':
            dict_evaluating_performance['Normal as Normal'] += 1
        elif dict_test_label[key] == 'N' and labeled_by_classifiers[key] == 'V':
            dict_evaluating_performance['Normal as PVC'] += 1
        elif dict_test_label[key] == 'V' and labeled_by_classifiers[key] == 'N':
            dict_evaluating_performance['PVC as Normal'] += 1
        elif dict_test_label[key] == 'V' and labeled_by_classifiers[key] == 'V':
            dict_evaluating_performance['PVC as PVC'] += 1

    return dict_evaluating_performance