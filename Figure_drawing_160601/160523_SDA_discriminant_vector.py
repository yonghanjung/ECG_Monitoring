import numpy as np
import scipy.io

import pywt
from scipy.stats import f
from Class_SparseLDA import SparseLDA
from HansFisherLDA import FisherLDA
import matplotlib.pyplot as plt


def ECG_Load(recordNum):
    FileName = '../Data/'+str(recordNum)+'_file.mat'
    mat = scipy.io.loadmat(FileName)
    SamplingRate = 360.0
    Time_domain = np.array([x / float(SamplingRate) for x in range(len(mat['val'][0]))])
    ECG = mat['val'][0]
    return Time_domain, ECG

def Index_Loading(recordNum):
    '''
    Output : Dictionary for each record frame
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

def Segment(ECG_Dyadic_Sample, Index_dict):
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
        if each_r > To_the_left and each_r + To_the_right < len(ECG_Dyadic_Sample):
            ECG_beat = ECG_Dyadic_Sample[range(each_r - To_the_left, each_r + To_the_right)]
            if Each_Type[iter_idx] == 'N' or Each_Type[iter_idx] == 'L' or Each_Type[iter_idx] == 'R' or Each_Type[iter_idx] == 'e' or Each_Type[iter_idx] == 'j' or \
                Each_Type[iter_idx] == 'V' or Each_Type[iter_idx] == 'E' \
                or Each_Type[iter_idx] == 'A' or Each_Type[iter_idx] == 'a' or Each_Type[iter_idx] == 'J' or Each_Type[iter_idx] == 'S' or \
                Each_Type[iter_idx] == 'F' or \
                Each_Type[iter_idx] == '/' or Each_Type[iter_idx] == 'f' or Each_Type[iter_idx] == 'Q':
                ECG_Segment.update({each_r: ECG_Dyadic_Sample[range(each_r - To_the_left, each_r + To_the_right)]})

                ECG_segment_box.append(ECG_beat)

                # MIH-BIH to AAMI
                Label = Each_Type[iter_idx]
                if Label == 'N' or Label == 'L' or Label == 'R' or Label == 'e' or Label == 'j':
                    AAMI_label = 'N'
                elif Label == 'V' or Label == 'E':
                    AAMI_label = 'V'
                elif Label == 'A' or Label == 'a' or Label == 'J' or Label == 'S':
                    AAMI_label = 'S'
                elif Label == 'F':
                    AAMI_label = 'F'
                elif Label == '/' or Label == 'f' or Label == 'Q':
                    AAMI_label = 'Q'
                ECG_label_box.append(AAMI_label)
                ECG_Segment_Type.update({each_r : AAMI_label})
        iter_idx += 1

    return ECG_Segment, ECG_Segment_Type

def ECG_to_Wav(ECG_dict):
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

def SDA_Operation(DictArray_TrainWCNormal,DictArray_TrainWCPVC,a,b):
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

    ### TRIAL ###
    alpha = a+b
    try:
        L1Ratio = a / (a+b)
    except:
        L1Ratio = 0.
    ObjLDA = SparseLDA(Dict_TrainingData=DictArrayMatrix_ForLDA, Flt_Lambda=alpha, Flt_L1=L1Ratio)
    # ObjLDA = SparseLDA(Dict_TrainingData=DictArrayMatrix_ForLDA, Flt_Lambda=0.1, Flt_L1=0.5)
    Mat_LDAOperator = ObjLDA.B
    return Mat_LDAOperator

def LDA_operation_trial(DictArray_TrainWCNormal,DictArray_TrainWCPVC):
    DictArrayMatrix_ForLDA = dict() #0 : Normal, 1 : PVC
    DictArrayMatrix_ForLDA[0] = list()
    DictArrayMatrix_ForLDA[1] = list()

    for idx, key in enumerate(sorted(DictArray_TrainWCNormal)):
        DictArrayMatrix_ForLDA[0].append(DictArray_TrainWCNormal[key])
    for idx, key in enumerate(sorted(DictArray_TrainWCPVC)):
        DictArrayMatrix_ForLDA[1].append(DictArray_TrainWCPVC[key])
    DictArrayMatrix_ForLDA[0] = np.array(DictArrayMatrix_ForLDA[0])
    DictArrayMatrix_ForLDA[1] = np.array(DictArrayMatrix_ForLDA[1])

    FisherLDA_obj = FisherLDA(DictArrayMatrix_ForLDA)
    return FisherLDA_obj.LDAOperator()

# Apply LDA to the Data
def LDA_Application(Matrix_LDAOperator,DictArray_TrainWCNormal):
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
def LDAON_ReducedCoVarComputation(LDAOperator, DictArray_TrainWCNormal):
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
def StatisticsConstruction(Reduced_Mean, Reduced_Cov, RedWavSegment_Test):
    DictFloat_Stat = dict()
    # List_FisherScoreForFeatures, Int_NumFeatureSelected, List_IdxFeatureSelected = self.LDAON_FisherScoreComputation()
    # Array_ReducedMean = self.LDAON_ReducedMeanComputation() # MuW
    # Array_ReducedMean = Array_ReducedMean[0]

    # Matrix_ReducedCov = self.LDAON_ReducedCoVarComputation() # wTAW
    # Matrix_ReducedCov = Matrix_ReducedCov[0]
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


def UCL_Computation(N, alpha):
    S = 1
    return (S*((N-1)**2) * f.ppf(1-alpha, S, N-S)) / (N*(N-S))


def AccuracyComputation(Stat_dict, Label_dict, UCL_val):
    # TYPE1 Error : Blue (Normal) Out Control
    # TYPE2 Error : Red (PVC) In Control
    # TYPE1 Duzi : Blue In Control
    # Type2 Duzi : Red Out Control

    Int_TotalTestPoint = 0
    Int_Type1_Error = 0
    Int_Type1_Duzi = 0
    Int_Type2_Error = 0
    Int_Type2_Duzi = 0

    DictInt_Accuracy = dict()
    DictFloat_Accuracy = dict()

    for idx, key in enumerate(sorted(Stat_dict)):
        # VEBSVEB
        Int_TotalTestPoint += 1
        if Label_dict[key] == 'N' or Label_dict[key] == 'L' or Label_dict[key] == 'R' or Label_dict[key] == 'e' or Label_dict[key] == 'j' :
            if Stat_dict[key] < UCL_val: # Normal In Control
                Int_Type1_Duzi += 1 # Normal as Normal
            elif Stat_dict[key] > UCL_val: # Normal Out Control
                # print self.Dict_TestLabel[key], DictFloat_Stat[key],  UCLVAL
                Int_Type1_Error += 1 # Normal as VEB
        # SVEB
        # elif self.Dict_TestLabel[key] == 'A' or self.Dict_TestLabel[key] == 'a' or self.Dict_TestLabel[key] == 'S' or self.Dict_TestLabel[key] == 'J':
        # VEB
        elif Label_dict[key] == 'V' or Label_dict[key] == 'E' :
            if Stat_dict[key] < UCL_val: # PVC In Control
                Int_Type2_Error += 1 # VEB as Normal
            elif Stat_dict[key] > UCL_val: # PVC Out Control
                Int_Type2_Duzi += 1 # VEB as VEB

    DictInt_Accuracy['Normal(G) as VEB'] = Int_Type1_Error
    DictInt_Accuracy['Normal(G) as Normal'] = Int_Type1_Duzi
    DictInt_Accuracy['VEB(G) as Normal'] = Int_Type2_Error
    DictInt_Accuracy['VEB(G) as VEB'] = Int_Type2_Duzi
    # DictInt_Accuracy['TotalBeat'] = Int_TotalTestPoint
    # DictInt_Accuracy['TotalError'] = Int_Type1_Error + Int_Type2_Error

    Int_TP = Int_Type2_Duzi # Correctly detected beat (Normal as Normal)
    Int_TN = Int_Type1_Duzi # Correctly rejected beat (PVC as PVC)
    Int_FP = Int_Type1_Error # Falsely detected (Normal as PVC)
    Int_FN = Int_Type2_Error # Misssed beat (PVC as Normal)

    # # Acc = NT - NE (=NC) / NT
    # DictFloat_Accuracy['Accuracy'] = float(Int_TotalTestPoint - (Int_Type1_Error+ Int_Type2_Error)) / float(Int_TotalTestPoint)
    # # Se = TP / TP + FN, ratio
    # DictFloat_Accuracy['Sensitivity'] = float(Int_TP) / float(Int_TP + Int_FN)
    # # Positive Predictivity = TP / TP + FP, ratio of correctly detected beat to the total num of beats
    # DictFloat_Accuracy['Positive_Predictivity'] = float(Int_TP) / float(Int_TP + Int_FP)
    # # Sp = TN / TN + FP , Correctly rejected to num of nonevent
    # DictFloat_Accuracy['Specificity'] = float(Int_Type2_Duzi) / float(Int_Type2_Duzi + Int_Type1_Error)

    return DictInt_Accuracy


def NN_accuracy(NN_answer, Label):
    DictInt_Accuracy = dict()
    DictInt_Accuracy['Normal as VEB'] = 0
    DictInt_Accuracy['Normal as Normal'] = 0
    DictInt_Accuracy['VEB as Normal'] = 0
    DictInt_Accuracy['VEB as VEB'] = 0

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




##############################################################################################################################
##############################################################################################################################


np.random.seed(123)

# List_Train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 220] # DS1
# List_Test = [100, 103, 105, 111, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 233, 234] # DS2

# Long_Term = [14046, 14134, 14149, 14157, 14172, 14184, 15814] # sampling rate = 128.0
INCART = range(301,376) # 301 - 375
HansTrain = [105, 106, 108, 109, 114, 118, 119, 200, 202, 203, 205, 208, 209, 210, 213, 214, 215, 219, 221, 223, 228, 233]

Normal = ['N','L','R','e','j']
# VEB = ['V','E']
VEB = ['V','E','A','a','J','S'] # VEB and SVEB by AAMI procedure

''' Reading data and segmenting by beats '''
RecordNum = 119
Sampling_rate = 360. # MIT BIH
# Sampling_rate = 257. # INCART
# Sampling_rate = 128. # LONG
Time_domain, ECG = ECG_Load(RecordNum)
Index_dict = Index_Loading(RecordNum)
ECG_segment_dict, ECG_label_dict = Segment(ECG,Index_dict)
print("Data reading..." + str(RecordNum) + " and " + str(len(ECG)))


''' Control variables '''
alphaList = [0.5, 0.25, 0.1, 0.05, 0.01, 0.0023]
alpha = 0.01

Training_list = [240,180,120,60]
Training_seconds = 300

a = 0.7 # SDA, for L1
b = 0.5 # SDA, for L2

# a0 = 0.0# SDA, for L1
# b0 = 0.0 # SDA, for L2
#
# a1 = 1e-16
# b1 = 1-a1 #L1 = 0.01 / L2 = 1
#
# a2 = 1e-4
# b2 = 1-a2 # L1=0.1 / L2 = 1.
#
# a3 = 1e-2
# b3 = 1-a3 # L1=0.5 / L2 = 1.
#
# a4 = 0.1
# b4 = 0.9 # L1=0.9 / L2 = 1.
#
# a5 = 0.99
# b5 = 0.01 # L1 = 0.99 / L2 = 1.
#
# a = 0.7
# b = 0.5

Acc_SPC = True
# Acc_SPC = False

# Acc_NN = True
Acc_NN = False

# Acc_SVM = True
Acc_SVM = False

# plot_YN = True
plot_YN = False


# for Training_seconds in Training_list:
# for alpha in alphaList:
''' Constructing training set '''
print("Training set construction...")

ECG_train_dict = {RIdx : ECG_segment_dict[RIdx] for RIdx in ECG_segment_dict.keys() if RIdx < Sampling_rate * Training_seconds}
# ECG_train_N_dict = {RIdx : ECG_segment_dict[RIdx] for RIdx in ECG_segment_dict.keys() if RIdx < Sampling_rate * Training_seconds and (ECG_label_dict[RIdx] == 'N' or ECG_label_dict[RIdx] == 'L' or ECG_label_dict[RIdx] == 'R' or ECG_label_dict[RIdx] == 'e' or ECG_label_dict[RIdx] == 'j')}
ECG_train_N_dict = {RIdx : ECG_segment_dict[RIdx] for RIdx in ECG_segment_dict.keys() if RIdx < Sampling_rate * Training_seconds and ECG_label_dict[RIdx] in Normal}
ECG_train_V_dict = {RIdx : ECG_segment_dict[RIdx] for RIdx in ECG_segment_dict.keys() if RIdx < Sampling_rate * Training_seconds and ECG_label_dict[RIdx] in VEB}
# ECG_train_V_dict = {RIdx : ECG_segment_dict[RIdx] for RIdx in ECG_segment_dict.keys() if RIdx < Sampling_rate * Training_seconds and (ECG_label_dict[RIdx] == 'V' or ECG_label_dict[RIdx] == 'E' )}
ECG_train_label_dict = {RIdx : ECG_label_dict[RIdx] for RIdx in ECG_label_dict.keys() if RIdx < Sampling_rate * Training_seconds}
ECG_test_dict = {RIdx : ECG_segment_dict[RIdx] for RIdx in ECG_segment_dict.keys() if RIdx > Sampling_rate * Training_seconds}
ECG_test_label_dict = {RIdx : ECG_label_dict[RIdx] for RIdx in ECG_label_dict.keys() if RIdx > Sampling_rate * Training_seconds}


''' Applying wavelet transformation '''
print("Wavelet construction...")
Wav_train_dict = ECG_to_Wav(ECG_train_dict)
Wav_train_N_dict = ECG_to_Wav(ECG_train_N_dict)
Wav_train_V_dict = ECG_to_Wav(ECG_train_V_dict)
Wav_test_dict = ECG_to_Wav(ECG_test_dict)


''' Computing the average of normal wavelet coefficients in training set '''
print("Normal mean computation...")
Wav_train_N_mean = np.zeros(256)
Count_Safe = 0
for idx, key in enumerate(sorted(Wav_train_N_dict)):
    # if Dict_Danger[key] == False:
    Wav_train_N_mean += np.array(Wav_train_N_dict[key])
    Count_Safe += 1
Wav_train_N_mean /= float(Count_Safe)


''' Constructing SDA operator'''

# SDA_original = LDA_operation_trial(Wav_train_N_dict,Wav_train_V_dict)
# SDA_orig = LDA_operation_trial(Wav_train_N_dict,Wav_train_V_dict)
# SDA_orig = SDA_Operation(Wav_train_N_dict,Wav_train_V_dict,a0,b0)
SDA_1 = SDA_Operation(Wav_train_N_dict,Wav_train_V_dict,a,b)
# SDA_2 = SDA_Operation(Wav_train_N_dict,Wav_train_V_dict,a2,b2) # More sparse
# SDA_3 = SDA_Operation(Wav_train_N_dict,Wav_train_V_dict,a3,b3) # More sparse
# SDA_4 = SDA_Operation(Wav_train_N_dict,Wav_train_V_dict,a4,b4) # More sparse
# SDA_5 = SDA_Operation(Wav_train_N_dict,Wav_train_V_dict,a5,b5) # More sparse


for idx in range(len(SDA_1)):
    if SDA_1[idx] != 0:
        print idx, SDA_1[idx]




#
# domain = range(1,257)
# plt.figure(1,figsize=(20,10))
#
# ax = plt.subplot(111)
# ax.vlines(domain,[0],SDA_1)
# ax.yaxis.set_visible(False)
# plt.axhline(y=0.0, c='black')
#
# plt.tight_layout()
# plt.savefig('Fig_SDA_discriminant_vector.eps',format='eps')
# plt.show()

# ax_orig = plt.subplot(511)
# # ax_orig.plot(SDA_orig, label="original")
# ax_orig.vlines(domain,[0],SDA_orig,label="Discriminant vector by Fisher LDA")
# plt.ylim(-0.001,0.001)
# plt.xlim(0,256)
# plt.legend()
# plt.axhline(y=0.0, c='black')
# ax_orig.yaxis.set_visible(False)
#
# ax_SDA1= plt.subplot(512)
# # ax_SDA1.plot(SDA_1,'r', label='a='+str(a1)+', b='+str(b1))
# ax_SDA1.vlines(domain,[0],SDA_1,label='Discriminant vector by SDA (a)')
# plt.ylim(-0.001,0.001)
# plt.xlim(0,256)
# ax_SDA1.yaxis.set_visible(False)
# plt.axhline(y=0.0, c='black')
# plt.legend()
#
# ax_SDA2 = plt.subplot(513)
# ax_SDA2.vlines(domain,[0],SDA_2,label='Discriminant vector by SDA (b)')
# plt.ylim(-0.001,0.001)
# plt.xlim(0,256)
# ax_SDA2.yaxis.set_visible(False)
# plt.axhline(y=0.0, c='black')
# plt.legend()
#
# ax_SDA3= plt.subplot(514)
# ax_SDA3.vlines(domain,[0],SDA_3,label='Discriminant vector by SDA (c)')
# plt.ylim(-0.001,0.001)
# plt.xlim(0,256)
# ax_SDA3.yaxis.set_visible(False)
# plt.axhline(y=0.0, c='black')
# plt.legend()
#
# ax_SDA4= plt.subplot(515)
# ax_SDA4.vlines(domain,[0],SDA_4,label='Discriminant vector by SDA (d)')
# plt.ylim(-0.001,0.001)
# plt.xlim(0,256)
# ax_SDA4.yaxis.set_visible(False)
# plt.axhline(y=0.0, c='black')
# plt.legend()
# plt.xlabel("The element of a sparse discriminant vector", fontsize=20)
# #
# # ax_SDA5= plt.subplot(616)
# # ax_SDA5.vlines(domain,[0],SDA_5,label='a='+str(a)+', b='+str(b))
# # plt.ylim(-0.001,0.001)
# # plt.xlim(0,256)
# # plt.legend()
# # ax_SDA5.yaxis.set_visible(False)
# # plt.axhline(y=0.0, c='black')
# # plt.xlabel("The element of a sparse discriminant vector")
#
# # plt.savefig('Fig_SDA.eps',format='eps')
#
#
# # ax_a1b1 = plt.subplot(212)
# #
# # plt.legend()
# # ax.vlines(domain,[0],LDA_Operator)
#
# plt.show()


