import matplotlib.pyplot as plt
import scipy.io
import pywt
import numpy as np

def IDX_Peak(List, Signal, Length_signal, Search_seconds = 0.1, Sampling = 360):
    New_Peak_Idx = []
    Plus_Minus = int(Sampling_rate*Search_seconds)
    for each_idx in List:
        # Search Zone Construction
        Search_Zone = range(max(0,each_idx- Plus_Minus), min(Length_signal, each_idx + Plus_Minus))

        # Put all idx in Search Zone
        Signal_values_in_Search_zone = [Signal[x] for x in Search_Zone]

        # Find max
        Idx = Signal_values_in_Search_zone.index(max(Signal_values_in_Search_zone))
        New_Peak_Idx.append(Search_Zone[Idx])
    return New_Peak_Idx


''' Control Variables '''
PVC_Data_Num = [105, 106, 116, 119, 124, 200, 201, 203, 208, 210, 213, 215, 217, 219, 221, 223, 228, 233]
# all_sum = 0

datanum = 221
Sampling_rate = 360
scale = 4
Dyadic_length = 2**19

''' Data Loading and Preparation'''
mat = scipy.io.loadmat('Data/'+str(datanum)+'_file.mat')
ECG = mat['val'][0]
ECG_Dyadic_Sample = ECG[:Dyadic_length]
Time_domain = [x / float(Sampling_rate) for x in range(len(ECG_Dyadic_Sample))]
Time_domain = np.array(Time_domain)

Annotation = open('Data/' + str(datanum) + '_anno.txt','rb')
Anno_Box = []
R_loc_box = []
Peak_IDX = []
R_accurate_time = []
for x in Annotation.readlines():
    try :
        A = x.split(" ")
        b = [elem for elem in A if elem != ""]
        if int(b[1]) <= Dyadic_length:
            R_accurate_time.append(b[0])
            R_loc_box.append(int(b[1]))
            Anno_Box.append(b)
    except:
        pass

R_loc_time = [x / float(Sampling_rate) for x in R_loc_box]
Peak_dict = {}
Wavelet_Peak_IDX = []

# for x,y in zip(R_loc_time, R_accurate_time):
#     print x,y

for x in R_loc_box:
    Peak_dict.update({x/float(Sampling_rate):ECG_Dyadic_Sample[x]})
    # print x / float(360), ECG_Dyadic_Sample[x]
    Wavelet_Peak_IDX.append(x / (2**scale))

# print Wavelet_Peak_IDX
''' Wavelet for denoising '''
db8 = pywt.Wavelet('db8')
DWT_result = pywt.wavedec(data=ECG_Dyadic_Sample, mode='per', wavelet=db8, level=scale)
# DWT_result = pywt.wavedec(data=ECG_Dyadic_Sample, mode='sym', wavelet=db8, level=scale)
DWT_result = pywt.wavedec(data=ECG_Dyadic_Sample, wavelet=db8, level=scale)
# print type(DWT_result), len(DWT_result)

# Thresholding
DWT_Scale_Approx = DWT_result[0]
DWT_Noise_Coefficient = DWT_result[len(DWT_result)-1]
Noise_Dev = np.median(np.abs(DWT_Noise_Coefficient)) / float(0.6745)
Noise_Threshold = np.sqrt(2*np.log(len(ECG_Dyadic_Sample))) * Noise_Dev

for Each_idx in range(len(DWT_result)):
    for elem_idx in range(len(DWT_result[Each_idx])):
        if DWT_result[Each_idx][elem_idx] < Noise_Threshold:
            DWT_result[Each_idx][elem_idx] = 0

Reconstructed_Signal = pywt.waverec(DWT_result,wavelet=db8,mode='sym')
Reconst_R_amplitude = Reconstructed_Signal[R_loc_box]
# print len(Reconstructed_Signal), len(ECG_Dyadic_Sample)
#
# Scailing = Dyadic_length / len(DWT_Scale_Approx)
#
# DWT_Scale_Time_domain =  [(Scailing * x) / float(Sampling_rate) for x in range(len(DWT_Scale_Approx))]
#
# ''' Wavelet Peak Marking = "Where is Scaling peak?" '''
#
# Wavelet_R_loc = [x / (2**scale) for x in R_loc_box]
# Wavelet_R_loc = IDX_Peak(Wavelet_R_loc, DWT_Scale_Approx, len(DWT_Scale_Approx))
#

''' P point Search '''
# Preknowledge : PR interval is 0.12 - 0.20s
Min_zone = int(Sampling_rate * 0.12)
Max_zone = int(Sampling_rate * 0.2)
P_loc_box = []
for r_loc in R_loc_box:
    P_search_zone = range(max(0,r_loc - Max_zone), max(0,r_loc - Min_zone))
    P_search_amp = Reconstructed_Signal[P_search_zone]
    P_search_amp = list(P_search_amp)
    if len(P_search_amp) > 0 :
        Temp_IDX = P_search_amp.index(max(P_search_amp))
        P_loc_box.append(P_search_zone[Temp_IDX])
    else:
        pass

P_loc_time = Time_domain[P_loc_box]
Reconst_P_amplitude = Reconstructed_Signal[P_loc_box]


''' P* Point Search  '''
# Preknowledge : P wave duration

P_star_search_limit = int(Sampling_rate * 0.08)
P_star_loc_box = []

for p_loc in P_loc_box:
    P_star_search_zone = range(p_loc - P_star_search_limit, p_loc)
    P_star_search_amplitude = Reconstructed_Signal[P_star_search_zone]
    P_star_search_amplitude = list(P_star_search_amplitude)
    if len(P_star_search_amplitude) > 0:
        Temp_IDX = P_star_search_amplitude.index(min(P_star_search_amplitude))
        P_star_loc_box.append(P_star_search_zone[Temp_IDX])
    else:
        pass

Reconst_P_star_amplitude = Reconstructed_Signal[P_star_loc_box]
P_star_loc_time = Time_domain[P_star_loc_box]

Beat_Box = {}
''' CUT BY BEATS '''
for idx in range(len(P_star_loc_box)-1):
    each_beat = range(P_star_loc_box[idx],P_star_loc_box[idx+1])
    Beat_Box.update({idx:each_beat})


''''''''''' PLOT '''''''''''''''

# plt.figure()
# plt.grid()
# plt.plot(Time_domain, ECG_Dyadic_Sample)
# plt.plot(R_loc_time, Reconst_R_amplitude,'ro')
# plt.plot(P_loc_time, Reconst_P_amplitude,'go')
# plt.plot(P_star_loc_time, Reconst_P_star_amplitude,'bo')
# plt.plot(Peak_dict.keys(), Peak_dict.values(),"ro")

# plt.figure()
# plt.title(str(datanum) + " ECG records ")
# plt.grid()
# # plt.plot(Time_domain, Reconstructed_Signal)
# plt.plot(Time_domain, ECG_Dyadic_Sample)
# plt.plot(R_loc_time, Reconst_R_amplitude,'ro', label = "R point")
# # plt.plot(P_loc_time, Reconst_P_amplitude,'go')
# plt.plot(P_star_loc_time, Reconst_P_star_amplitude,'bo', label = "Start point")
# plt.legend()

# plt.plot(Reconstructed_Signal)
# plt.plot(Reconst_R_amplitude,'ro')
# plt.plot(Reconst_P_amplitude,'go')
# plt.plot(Reconst_P_star_amplitude,'bo')


# plt.figure()
# plt.title('Arbi beat')
# plt.grid()
# plt.plot(ECG_Dyadic_Sample[Beat_Box[4]])

# plt.show()





