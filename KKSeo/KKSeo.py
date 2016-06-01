__author__ = 'jeong-yonghan'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io
from scipy.signal import resample
import pywt
from Class_SparseLDA import SparseLDA
from scipy.stats import f
import random

# Data Load
def ECG_Load(recordNum):
    FileName = '../Data/'+str(recordNum)+'_file.mat'
    mat = scipy.io.loadmat(FileName)
    SamplingRate = 360.0
    Time_domain = np.array([x / float(SamplingRate) for x in range(len(mat['val'][0]))])
    ECG = mat['val'][0]

    return Time_domain, mat['val'][0]

def Index_Loading(recordNum):
    index_file = open('../Data/'+str(recordNum)+'_anno.txt','rb')
    SamplingRate = 360.0
    DyadLength = 2 ** (int(np.log2(SamplingRate * 60 * 30)))
    Index_dict = {}
    Index_dict['Time'] = []
    Index_dict['Sample'] = []
    Index_dict['Type'] = []

    for each_line in index_file.readlines():
        try :
            A = each_line.split(" ")
            b = [elem for elem in A if elem != ""]
            # b[0] time, b[1] sample idx, b[2] Type
            if int(b[1]) <= DyadLength:
                Index_dict['Time'].append(b[0])
                Index_dict['Sample'].append(int(b[1]))
                Index_dict['Type'].append(b[2])
                # Index_dict.update({'Time' : b[0], 'Sample' : int(b[1]), 'Type' : b[2]})
        except:
            pass

    return Index_dict

def ECG_Segment(FileName):
    Time_Domain, ECGdata = ECG_Load(FileName)
    SamplingRate = 360.0

    To_the_left = 128
    To_the_right = 128
    '''
    Output
    - ECG_segment : Dictionary (key : r-index, value : list)
    - ECG_segment_type : Dictionary (Key : r-index, value : type)
    '''
    iter_idx = 0
    # Time_domain, ECG_Dyadic_Sample, Dyad_length = self.Data_Load()
    ECG_Segment = {}
    ECG_Segment_Type = {}
    Index_dict = Index_Loading(FileName)
    Each_Type = Index_dict['Type']
    R_Locations_Index = Index_dict['Sample']
    for each_r in R_Locations_Index:
        if each_r > To_the_left and \
                                each_r + To_the_right < len(ECGdata):
            if Each_Type[iter_idx] == 'N' or Each_Type[iter_idx] == 'L' or \
                            Each_Type[iter_idx] == 'R' or Each_Type[iter_idx] == 'e' or \
                            Each_Type[iter_idx] == 'j' or Each_Type[iter_idx] == 'A' or \
                            Each_Type[iter_idx] == 'a' or Each_Type[iter_idx] == 'S' or \
                            Each_Type[iter_idx] == 'V' or Each_Type[iter_idx] == 'E' or \
                            Each_Type[iter_idx] == 'F' :
                ECG_Segment.update({each_r: ECGdata[range(each_r - To_the_left, each_r + To_the_right)]})
                ECG_Segment_Type.update({each_r : Each_Type[iter_idx]})
        iter_idx += 1

    return ECG_Segment, ECG_Segment_Type


def Data_Loading(FileName):
    '''
    Goal : Load the ECG mat (MIT BIH)
    Output
    - Time domain : Array
    - ECG_Dyadic_Sample = np.array
    - DyadLenth = integer
    '''

    SamplingRate = 50.0
    # try:
        # mat = scipy.io.loadmat('Data/'+str(self.datanum)+'_file.mat')
    file = pd.read_csv(FileName)
    # except:
    #     # mat = scipy.io.loadmat('../Data/'+str(self.datanum)+'_file.mat')
    #     print "error"

    ''' KKseo '''
    # "accelY","accelZ","accelMag","gyroX","risk","step"
    accelY = file["accelY"]

    accelZ = file["accelZ"]
    accelMag = file["accelMag"]
    gyroX = file["gyroX"]
    risk = file["risk"]
    step = file["step"]

    Time_domain = np.array([x / float(SamplingRate) for x in range(len(accelY))])

    return Time_domain, accelY,accelZ,accelMag,gyroX,risk,step


# Segmentation
def KKSeo_Segment(FileName):
    Time_domain, accelY, accelZ, accelMag, gyroX, risk, step  = Data_Loading(FileName)
    # print accelY
    Segment = {}
    Segment_Type = {}

    StepIdx = step[step==4].keys()
    StepStart = list()
    StepEnd = list()
    for i in range(1,len(StepIdx)-1):
        # if np.mod(i,2) == 0:
        StepStart.append(StepIdx[i])
        # else:
        StepEnd.append(StepIdx[i])
    StepStart.append(0)
    StepEnd.append(len(StepIdx)-1)
    # print StepStart
    # print StepEnd

    StepCount = 0
    StepSwitch = False

    for i in range(len(gyroX)):
        if i in StepStart or i in StepEnd:
            StepSwitch = True
            StepCount += 1
            Segment[StepCount] = list( )
            Segment_Type[StepCount] = list()
        # if i in StepEnd:
        #     StepSwitch = False

        if StepSwitch:
            # print accelY[i]
            Segment[StepCount].append(accelY[i])
            Segment_Type[StepCount].append(risk[i])

    return Segment,Segment_Type

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



# Generate LDA Operator
def LDA_Operation(DictArray_TrainWCNormal,DictArray_TrainWCPVC,a,b):
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
    L1Ratio = a / (a+b)
    # a = L1
    # b = L2
    ObjLDA = SparseLDA(Dict_TrainingData=DictArrayMatrix_ForLDA, Flt_Lambda=alpha, Flt_L1=L1Ratio)


    Mat_LDAOperator = ObjLDA.B

    # Mat_LDAOperator = ObjectNewtonMethod.GradientDescent()
    # print Mat_LDAOperator
    return Mat_LDAOperator


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




#################################################################################

FileName = "zone1_downhill_hengame_labeled.csv"
RecordNum = 100
CutLength = 64
# Segment,SegmentType = ECG_Segment(100)
Segment,SegmentType = KKSeo_Segment(FileName)
Dict_Danger = dict()
WavSegment = dict()

''' Cut by cut length '''
for idx,key in enumerate(sorted(Segment)):
    # print len(Segment[key])
    Segment[key] = Segment[key][:CutLength]
    if 5 in SegmentType[key]:
        Dict_Danger[key] = True # Risky
    else:
        Dict_Danger[key] = False # Safe

''' Resampling for shorter than cut length '''
for idx,key in enumerate(sorted(Segment)):
    Segment[key] = resample(Segment[key],64)

''' Wavelet analysis '''
WaveletBasis = 'db6'
DecompLevel = 4

for idx,key in enumerate(sorted(Segment)):
    WavSegment[key] = pywt.wavedec(data=Segment[key],wavelet=pywt.Wavelet(WaveletBasis),mode='per',level=DecompLevel)
    WaveletThreshold = np.sqrt(2 * np.log(256)) * (np.median(np.abs(np.array(WavSegment[key][4]) - np.median(WavSegment[key][4]))) / 0.6745)
    DetailCoefs = np.concatenate([WavSegment[key][1],WavSegment[key][2],WavSegment[key][3],WavSegment[key][4]])
    NoiseRemoved = SoftThreshold(DetailCoefs, WaveletThreshold)
    WaveletCoefs = np.concatenate([WavSegment[key][0], NoiseRemoved])
    WavSegment[key] = WaveletCoefs

Num_Train = 20
Num_Train_Safe = 13
Num_Train_Danger = 7
WavSegment_Train = dict()
WavSegment_Train_Safe = dict()
WavSegment_Train_Danger = dict()

WavSegment_Test = dict()
Safe_Idx = list()
Danger_Idx = list()

''' Seperate by Train and Test '''
for idx,key in enumerate(sorted(WavSegment)):
    if Dict_Danger[key] == False: # Safe Step
        Safe_Idx.append(key)
    else:
        Danger_Idx.append(key)

random.seed(100)
Train_Idx = random.sample(Safe_Idx,Num_Train_Safe) + random.sample(Danger_Idx,Num_Train_Danger)
print Train_Idx

for idx,key in enumerate(sorted(WavSegment)):
    if key in Train_Idx:
        WavSegment_Train[key] = WavSegment[key]
        if Dict_Danger[key] == True:
            WavSegment_Train_Danger[key] = WavSegment[key]
        else:
            WavSegment_Train_Safe[key] = WavSegment[key]
    else:
        WavSegment_Test[key] = WavSegment[key]


# for idx, key in enumerate(sorted(WavSegment)):
    # # if key <= Train_Step:
    # WavSegment_Train[key] = WavSegment[key]
    #     if Dict_Danger[key] == True:
    #         WavSegment_Train_Danger[key] = WavSegment[key]
    #     else:
    #         WavSegment_Train_Safe[key] = WavSegment[key]
    # else:
    #     WavSegment_Test[key] = WavSegment[key]

''' Compute Normal Mean '''
WavSegment_Train_Safe_Mean = np.zeros(CutLength)
Count_Safe = 0
for idx, key in enumerate(sorted(WavSegment_Train_Safe)):
    # if Dict_Danger[key] == False:
    WavSegment_Train_Safe_Mean += np.array(WavSegment_Train_Safe[key])
    Count_Safe += 1
WavSegment_Train_Safe_Mean /= float(Count_Safe)


''' Feature extraction '''
a = 0.7 # L1
b = 0.5 # L2
LDA_Operator = LDA_Operation(WavSegment_Train_Safe,WavSegment_Train_Danger,a,b)
RedWavSegment_Train_Safe = LDA_Application(LDA_Operator,WavSegment_Train_Safe)
RedWavSegment_Train_Danger = LDA_Application(LDA_Operator,WavSegment_Train_Danger)
RedWavSegment_Test = LDA_Application(LDA_Operator,WavSegment_Test)

# Compute Mean and Cov
ReducedMean = np.dot(LDA_Operator,WavSegment_Train_Safe_Mean)
ReducedCov = LDAON_ReducedCoVarComputation(LDA_Operator,WavSegment_Train_Safe)
Statistics_Test = StatisticsConstruction(ReducedMean,ReducedCov,RedWavSegment_Test)

for idx,key in enumerate(sorted(Statistics_Test)):
    print key, Statistics_Test[key], Dict_Danger[key]


# print ReducedMean,ReducedCov

# print Dict_Danger[64]
#
# plt.plot(Segment[64],'b')
# plt.plot(Segment[62],'r')
# plt.show()

# for idx,key in enumerate(sorted(WavSegment_Train_Safe)):
#     plt.figure(key)
#     plt.title(key)
#     plt.plot(WavSegment_Train_Safe[key])
# plt.show()
#
# plt.figure()
# plt.plot(WavSegment[33])
#
# plt.figure()
# plt.plot(WavSegment[34])
#
# print Dict_Danger[33]
# print Dict_Danger[34]
#
# plt.show()
#
#
#
# print WavSegment_Train_Safe
# plt.plot(WavSegment_Train_Safe)
# plt.show()
# print Dict_Danger
#
#
#
