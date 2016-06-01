import numpy as np
import scipy.io
import pywt
from scipy.stats import f
from Final.Class.Class_SparseLDA import SparseLDA

def Loading_ECG(recordNum):
    '''
    Loading the ECG record
    :param: record_number
    :return: time_domain (seconds) / ECG_record
    '''
    FileName = '../Data/'+str(recordNum)+'_file.mat'
    mat = scipy.io.loadmat(FileName)
    SamplingRate = 360.0
    Time_domain = np.array([x / float(SamplingRate) for x in range(len(mat['val'][0]))])
    ECG = mat['val'][0]
    return Time_domain, ECG

def Loading_R_Peak_and_Label(recordNum):
    '''
    Loading R peak index and label of the ECG record
    :param: record_index
    :return: dictionary (key: 'Time','Sample','Type' / val: time_(sec)_R_peak, sample_index_R_peak, label_beat_R_peak
    '''
    try:
        index_file = open('Data/'+str(recordNum)+'_anno.txt','rb')
    except:
        index_file = open('../Data/'+str(recordNum)+'_anno.txt','rb')
    Index_dict = {}
    Index_dict['Time'] = []
    Index_dict['Sample'] = []
    Index_dict['Type'] = []

    for each_line in index_file.readlines():
        try :
            A = each_line.split(" ")
            b = [elem for elem in A if elem != ""] # b[0] time, b[1] sample idx, b[2] Type
            # if int(b[1]) <= DyadLength:
            Index_dict['Time'].append(b[0])
            Index_dict['Sample'].append(int(b[1]))
            Index_dict['Type'].append(b[2])
        except:
            pass
    return Index_dict

def SoftThreshold(X, Threshold):
    '''
    Implementing the soft threshold for the universal thresholding (Donoho, 1994)
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

def Segmenting_ECG_Beat(ECG_record, Index_dict,AAMI_total_label,AAMI_normal,AAMI_PVC):
    '''
    Segment ECG record beat by beat
    :param ECG_record: ECG record
    :param Index_dict: dictionary providing information about ECG beat
    :param AAMI_total_label: Total label that should be imported
    :return dict_ECG_record (key: R_peak_index, val: ECG_beat)
    :return dict_ECG_label (key: R_peak_index, val: labe_ECG_beat)
    '''
    To_the_left = 128
    To_the_right = 128
    ECG_Segment = dict()
    ECG_Segment_Type = dict()
    Each_Type = Index_dict['Type']
    R_Locations_Index = Index_dict['Sample']

    ECG_segment_box = list()
    ECG_label_box = list()

    iter_idx = 0
    for each_r in R_Locations_Index:
        # if each_r > To_the_left and each_r + To_the_right < Dyad_length:
        if each_r > To_the_left and each_r + To_the_right < len(ECG_record):
            ECG_beat = ECG_record[range(each_r - To_the_left, each_r + To_the_right)]
            if Each_Type[iter_idx] in AAMI_total_label:
                ECG_Segment.update({each_r: ECG_record[range(each_r - To_the_left, each_r + To_the_right)]})
                ECG_segment_box.append(ECG_beat)
                Label = Each_Type[iter_idx]
                if Label in AAMI_normal:
                    AAMI_label = 'N'
                elif Label in AAMI_PVC:
                    AAMI_label = 'V'
                elif Label not in AAMI_normal and Label not in AAMI_PVC and Label in AAMI_total_label:
                    AAMI_label = 'S'
                ECG_label_box.append(AAMI_label)
                ECG_Segment_Type.update({each_r : AAMI_label})
        iter_idx += 1
    return ECG_Segment, ECG_Segment_Type

def Wavelet_Transformation(ECG_dict):
    Wav_dict = dict()
    WaveletBasis = 'db8'
    DecompLevel = 4
    for idx,key in enumerate(sorted(ECG_dict)):
        Wav_dict[key] = pywt.wavedec(data=ECG_dict[key],wavelet=pywt.Wavelet(WaveletBasis),mode='per',level=DecompLevel)
        # print len(Wav_dict[key][DecompLevel])
        WaveletThreshold = np.sqrt(2 * np.log(256)) * (np.median(np.abs(np.array(Wav_dict[key][DecompLevel]) - np.median(Wav_dict[key][DecompLevel]))) / 0.6745)
        DetailCoefs = np.concatenate([Wav_dict[key][0],Wav_dict[key][1],Wav_dict[key][2],Wav_dict[key][3],Wav_dict[key][4]])
        NoiseRemoved = SoftThreshold(DetailCoefs, WaveletThreshold)
        Wav_dict[key] = NoiseRemoved
    return Wav_dict

def Constructing_SDA_Vector(DictArray_TrainWCNormal,DictArray_TrainWCPVC,a,b):
    # Compute LDA Coefficeints
    DictArrayMatrix_ForLDA = dict() #0 : Normal, 1 : PVC
    DictArrayMatrix_ForLDA[0] = list()
    DictArrayMatrix_ForLDA[1] = list()

    for idx, key in enumerate(sorted(DictArray_TrainWCNormal)):
        DictArrayMatrix_ForLDA[0].append(DictArray_TrainWCNormal[key])
    for idx, key in enumerate(sorted(DictArray_TrainWCPVC)):
        DictArrayMatrix_ForLDA[1].append(DictArray_TrainWCPVC[key])
    DictArrayMatrix_ForLDA[0] = np.array(DictArrayMatrix_ForLDA[0])
    DictArrayMatrix_ForLDA[1] = np.array(DictArrayMatrix_ForLDA[1])

    alpha = a+b
    try:
        L1Ratio = a / (a+b)
    except:
        L1Ratio = 0.0

    ObjLDA = SparseLDA(Dict_TrainingData=DictArrayMatrix_ForLDA, Flt_Lambda=alpha, Flt_L1=L1Ratio)
    Mat_LDAOperator = ObjLDA.B
    return Mat_LDAOperator

# Apply LDA to the Data
def Projecting_Lower_Dimensional_Vec(Matrix_LDAOperator,DictArray_TrainWCNormal):
    DictArray_TrainWCNormal_LDA = dict()

    # Matrix_LDAOpeartor = self.LDAON_LDAOperatorConstruction()

    # Key : Record , Value : Train Normal WC
    for idx, key in enumerate(sorted(DictArray_TrainWCNormal)):
        # 1 by 64
        Val = np.reshape(DictArray_TrainWCNormal[key], (len(DictArray_TrainWCNormal[key]),1 ))

        # w*X
        NewVal = np.dot(np.transpose(Matrix_LDAOperator), Val)
        NewVal = np.squeeze(np.asarray(NewVal))
        NewVal = float(NewVal)
        DictArray_TrainWCNormal_LDA[key] = NewVal
    return DictArray_TrainWCNormal_LDA

# Compute Cov after applying LDA
def Projecting_Low_Dimensional_Cov(LDAOperator, DictArray_TrainWCNormal):
    # Construct_Original Cov

    # 1. Construct Array Matrix (300 by 64)
    ArrayMat = list()
    for idx, key in enumerate(sorted(DictArray_TrainWCNormal)):
        Array_EachBeatWC = DictArray_TrainWCNormal[key]
        ArrayMat.append(Array_EachBeatWC)
    ArrayMat = np.array(ArrayMat)
    Array_VarElem = np.var(ArrayMat, axis=0)
    ArrayMat_COV = np.diag(Array_VarElem)

    # 2. Fisher Coef (64,1)
    Array_FisherCoef = LDAOperator
    Array_FisherCoef = np.reshape(Array_FisherCoef,(len(Array_FisherCoef),1))
    FisherTransformedCOV = np.dot(np.dot(np.transpose(Array_FisherCoef), ArrayMat_COV), Array_FisherCoef)
    return np.ravel(FisherTransformedCOV)

# Construct Stat
def Constructing_T2_Stat(Reduced_Mean, Reduced_Cov, RedWavSegment_Test):
    DictFloat_Stat = dict()
    Array_ReducedMean = Reduced_Mean
    Matrix_ReducedCov = Reduced_Cov

    # Key : record, Val : 1dim
    DictArray_TestWC_LDA = RedWavSegment_Test

    for idx, key in sorted(enumerate(DictArray_TestWC_LDA)):
        Val = DictArray_TestWC_LDA[key] # 1 Dim
        Val = np.array(Val)
        Val = np.array(Val - Array_ReducedMean)

        NewVal = Val * (Matrix_ReducedCov**(-1)) * Val.T
        Stat = np.squeeze(np.asarray(NewVal))
        # print "Stat", self.Dict_TestLabel[key], NewVal
        DictFloat_Stat[key] = Stat
    return DictFloat_Stat


def Computing_UCL(N, alpha):
    S = 1
    return (S*((N-1)**2) * f.ppf(1-alpha, S, N-S)) / (N*(N-S))


def Evaluating_Performance_SPM(Stat_dict, Label_dict, UCL_val,AAMI_Normal,AAMI_PVC):
    Int_TotalTestPoint = 0
    Int_Type1_Error = 0
    Int_Type1_correct = 0
    Int_Type2_Error = 0
    Int_Type2_correct = 0

    DictInt_Accuracy = dict()

    for idx, key in enumerate(sorted(Stat_dict)):
        Int_TotalTestPoint += 1
        if Label_dict[key] in AAMI_Normal:
            if Stat_dict[key] < UCL_val: # Normal In Control
                Int_Type1_correct += 1 # Normal as Normal
            elif Stat_dict[key] > UCL_val: # Normal Out Control
                # print self.Dict_TestLabel[key], DictFloat_Stat[key],  UCLVAL
                Int_Type1_Error += 1 # Normal as VEB

        elif Label_dict[key] in AAMI_PVC :
            if Stat_dict[key] < UCL_val: # PVC In Control
                Int_Type2_Error += 1 # VEB as Normal
            elif Stat_dict[key] > UCL_val: # PVC Out Control
                Int_Type2_correct += 1 # VEB as VEB

    DictInt_Accuracy['Normal as PVC'] = Int_Type1_Error
    DictInt_Accuracy['Normal as Normal'] = Int_Type1_correct
    DictInt_Accuracy['PVC as Normal'] = Int_Type2_Error
    DictInt_Accuracy['PVC as PVC'] = Int_Type2_correct
    return DictInt_Accuracy


def Evaluating_Performance_SVM_NN(NN_answer, Label):
    DictInt_Accuracy = dict()
    DictInt_Accuracy['Normal as PVC'] = 0
    DictInt_Accuracy['Normal as Normal'] = 0
    DictInt_Accuracy['PVC as Normal'] = 0
    DictInt_Accuracy['PVC as PVC'] = 0

    for idx,key in enumerate(sorted(NN_answer)):
        if Label[key] == 'N' and NN_answer[key] == 'N':
            DictInt_Accuracy['Normal as Normal'] += 1
        elif Label[key] == 'N' and NN_answer[key] == 'V':
            DictInt_Accuracy['Normal as VEB'] += 1
        elif Label[key] == 'V' and NN_answer[key] == 'N':
            DictInt_Accuracy['VEB as Normal'] += 1
        elif Label[key] == 'V' and NN_answer[key] == 'V':
            DictInt_Accuracy['VEB as VEB'] += 1

    return DictInt_Accuracy
