import numpy as np

from methods import Loading_ECG
from methods import Loading_R_Peak_and_Label
from methods import Segmenting_ECG_Beat
from methods import Wavelet_Transformation
from methods import Constructing_SDA_Vector
from methods import Projecting_Lower_Dimensional_Vec
from methods import Projecting_Low_Dimensional_Cov
from methods import Constructing_T2_Stat
from methods import Computing_UCL
from methods import Evaluating_Performance_SPM
from methods import Evaluating_Performance_SVM_NN

''' ECG record number '''
LongTerm_idx = [14046, 14134, 14149, 14157, 14172, 14184, 15814] # sampling rate = 128.0
INCART_idx = range(301,376) # 301 - 375
MITBIH_idx = [105, 106, 108, 109, 114, 118, 119, 200, 202, 203, 205, 208, 209, 210, 213, 214, 215, 219, 221, 223, 228, 233]

''' ECG labels '''
AAMI_Normal = ['N','L','R','e','j'] # Those label in MIT-BIH are considered as Normal in AAMI
AAMI_PVC = ['V','E'] # Those label in MIT-BIH are considered as Normal in AAMI
AAMI_NonNormal= ['V','E','A','a','J','S']
AAMI_Total_label = AAMI_Normal + AAMI_NonNormal

''' Control variables '''
record_idx = 119 # from LongTerm_idx, INCART_idx, or MITBIH_idx
alpha = 0.01  # [0.5,0.25,0.1,0.05,0.01,0.0023]
time_training = 300 # [240,180,120,60]

SDA_L1_penalty = 0.7
SDA_L2_penalty = 0.5

SPM_switch = True # if False, then SPM is not run in this program
NeuralNetwork_switch = False  # if False, then Neural network is not run in this program
SVM_switch = True # if False, then SVM is not run in this program
plot_switch = False # if False, the plot for SPM is not drawn

''' 1. Reading ECG records and segmenting by beats '''
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

time_domain, ECG_record = Loading_ECG(record_idx,sampling_rate)
R_peak_and_label = Loading_R_Peak_and_Label(record_idx)
dict_ECG_beat_segmented, dict_ECG_beat_label = Segmenting_ECG_Beat(ECG_record,R_peak_and_label,AAMI_Total_label,AAMI_Normal,AAMI_PVC) # key: R peak index

print("Reading ECG record " + str(record_idx) + " from " + data_name + "...")


''' 2. Constructing training set (initial 5min are segmented as training set) '''
print("Training set construction...")
dict_train_ECG = {RIdx : dict_ECG_beat_segmented[RIdx] for RIdx in dict_ECG_beat_segmented.keys() if RIdx < sampling_rate * time_training}
dict_train_ECG_normal = {RIdx : dict_ECG_beat_segmented[RIdx] for RIdx in dict_ECG_beat_segmented.keys() if RIdx < sampling_rate * time_training and dict_ECG_beat_label[RIdx] in AAMI_Normal}
dict_train_ECG_PVC = {RIdx : dict_ECG_beat_segmented[RIdx] for RIdx in dict_ECG_beat_segmented.keys() if RIdx < sampling_rate * time_training and dict_ECG_beat_label[RIdx] in AAMI_NonNormal}
dict_train_label = {RIdx : dict_ECG_beat_label[RIdx] for RIdx in dict_ECG_beat_label.keys() if RIdx < sampling_rate * time_training}
dict_test_ECG = {RIdx : dict_ECG_beat_segmented[RIdx] for RIdx in dict_ECG_beat_label.keys() if RIdx > sampling_rate * time_training}
dict_test_label = {RIdx : dict_ECG_beat_label[RIdx] for RIdx in dict_ECG_beat_label.keys() if RIdx > sampling_rate * time_training}


''' 3. Applying wavelet transformation to each ECG beats in training and test set '''
print("Wavelet construction...")
dict_train_wc = Wavelet_Transformation(dict_train_ECG)
dict_train_wc_normal = Wavelet_Transformation(dict_train_ECG_normal)
dict_train_wc_PVC = Wavelet_Transformation(dict_train_ECG_PVC)
dict_test_wc = Wavelet_Transformation(dict_test_ECG)

average_train_wc_normal = np.zeros(256)
number_train_normal = 0
for idx, key in enumerate(sorted(dict_train_wc_normal)):
    average_train_wc_normal += np.array(dict_train_wc_normal[key])
    number_train_normal += 1
average_train_wc_normal /= float(number_train_normal)


''' 4. Constructing sparse discriminant vector and projecting to the low dimensional space'''
print("Constructing sparse discriminant vector and projecting to the low dimensional space...")
SDA_vector \
    = Constructing_SDA_Vector(dict_train_wc_normal,dict_train_wc_PVC,SDA_L1_penalty,SDA_L2_penalty)

dict_train_projected_normal \
    = Projecting_Lower_Dimensional_Vec(SDA_vector,dict_train_wc_normal)
dict_train_projected_PVC \
    = Projecting_Lower_Dimensional_Vec(SDA_vector,dict_train_wc_PVC)
dict_train_projected \
    = Projecting_Lower_Dimensional_Vec(SDA_vector, dict_train_wc)
dict_test_projected \
    = Projecting_Lower_Dimensional_Vec(SDA_vector,dict_test_wc)

projected_average_train_wc_normal =\
    np.dot(SDA_vector,average_train_wc_normal)
projected_Cov_train_wc_normal = Projecting_Low_Dimensional_Cov(SDA_vector,dict_train_wc_normal)

''' 5. Computing T2 statistics '''
print("Computing T2 statistics...")
dict_test_T2stat \
    = Constructing_T2_Stat(projected_average_train_wc_normal,projected_Cov_train_wc_normal,dict_test_projected)
# Stat_dict_test = Constructing_T2_Stat(ReducedMean,ReducedCov,RedWavSegment_Test)
dict_train_T2stat = Constructing_T2_Stat(projected_average_train_wc_normal,projected_Cov_train_wc_normal,dict_train_projected)
UCL = Computing_UCL(len(dict_train_wc_normal),alpha)

''' 6. Evaluating accuracy by counting right and wrongly classified beats '''
if SPM_switch:
    print("")
    print("Evaluating of wavelet-based SPM for record number " + str(record_idx) + " in " + data_name)
    DictInt_Accuracy = Evaluating_Performance_SPM(dict_test_T2stat, dict_test_label,UCL,AAMI_Normal,AAMI_PVC)
    for idx,key in enumerate(sorted(DictInt_Accuracy)):
        print key, DictInt_Accuracy[key]

''' Competiting method: NN '''
if NeuralNetwork_switch:
    print ""
    print("Evaluating Neural network for record number " + str(record_idx) + " in " + data_name)
    from pybrain.datasets import SupervisedDataSet
    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.supervised.trainers import BackpropTrainer

    dim = 256
    numHidden = 256
    ds = SupervisedDataSet(dim,1)
    # net = buildNetwork(dim,numHidden,1,outclass=SoftmaxLayer, bias = True)
    # net = buildNetwork(dim,numHidden,1,outclass=GaussianLayer, bias = True)
    net = buildNetwork(dim,numHidden,1)

    # Dataset regeneration for pybrain
    Wav_train_list = list()
    Train_label_list = list()
    for idx,key in enumerate(sorted(dict_train_wc)):
        Label = dict_train_label[key]
        if Label in AAMI_Normal:
            ds.addSample(dict_train_wc[key],1)
        elif Label in AAMI_PVC:
            ds.addSample(dict_train_wc[key],0)

    trainer = BackpropTrainer(net, dataset=ds)
    trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)
    NN_ans_dict = dict()

    for idx, key in enumerate(sorted(dict_test_wc)):
        Label = dict_test_label[key]
        NN_ans = net.activate(dict_test_wc[key])
        if abs(NN_ans[0] - 1 ) < abs(NN_ans[0] - 0): # NN_ans : Normal
            NN_ans_dict[key] = 'N'
        else:
            NN_ans_dict[key] = 'V'

    NN_accracy_dict = Evaluating_Performance_SVM_NN(NN_ans_dict,dict_test_label)
    for idx,key in enumerate(sorted(NN_accracy_dict)):
        print key, NN_accracy_dict[key]


''' Competiting method: SVM '''
if SVM_switch:
    print("")
    print("Evaluating SVM for record number "+ str(record_idx) + " in " + data_name)
    from sklearn.svm import SVC
    X = list()
    y = list()

    for idx,key in enumerate(sorted(dict_train_T2stat)):
        Label = dict_train_label[key]
        if Label in AAMI_Normal:
            X.append(dict_train_T2stat[key])
            y.append(1)
        elif Label in AAMI_PVC:
            X.append(dict_train_T2stat[key])
            y.append(0)

    X = np.reshape(X, (len(X),1))
    y = np.array(y)

    clf = SVC(kernel='linear')
    clf.fit(X,y)
    SVM_ans_dict = dict()
    for idx, key in enumerate(sorted(dict_test_T2stat)):
        Label = dict_test_label[key]
        SVM_ans = clf.predict(dict_test_T2stat[key])
        if abs(SVM_ans[0] - 1 ) < abs(SVM_ans[0] - 0): # NN_ans : Normal
            SVM_ans_dict[key] = 'N'
        else:
            SVM_ans_dict[key] = 'V'

    SVM_accracy_dict = Evaluating_Performance_SVM_NN(SVM_ans_dict,dict_test_label)
    for idx,key in enumerate(sorted(SVM_accracy_dict)):
        print key, SVM_accracy_dict[key]



''' Compute accuracy '''


''' Plot '''
if plot_switch:
    import matplotlib.pyplot as plt
    plt.figure()
    for idx,key in enumerate(sorted(dict_test_wc)):
        plt.plot(idx,UCL,'m.')
        if dict_test_label[key] == 'N':
            plt.plot(idx,dict_test_T2stat[key],'bo')
        elif dict_test_label[key] == 'V':
            plt.plot(idx,dict_test_T2stat[key],'ro')
        else:
            plt.plot(idx,dict_test_T2stat[key],'go')

    plt.show()