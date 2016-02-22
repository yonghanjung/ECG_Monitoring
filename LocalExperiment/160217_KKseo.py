import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

import csv # To read csv

'''Data Load'''
FileName = "../Data/zone3_uphill_hengame_labeled.csv"
df = pd.read_csv(FileName)

accelY = df['accelY']
accelZ = df['accelZ']
accelMag = df['accelMag']
gyroX = df['gyroX']
risk = df['risk']
step = df['step']

sampling_rate = 50.0
Time_domain = np.array([x / float(sampling_rate) for x in range(len(accelY))])

KKseo_Sig = dict()
step_start = False
step_count = 0
newIdx=  0
Label = dict()
idx_save = True
idx_start = 0
idx_end = 0
for i in range(len(accelY)):
    step_status = step[i]
    if step_status == 1 and step_start == False:
        step_start = True
        step_count += 1
        if idx_save:
            idx_save = False
            idx_start = i
    elif step_status == 1 and step_start == True:
        step_start = False

    if np.mod(step_count,3) == 0:
        KKseo_Sig[newIdx] = accelY[range(idx_start,i)]
        newIdx += 1
        idx_save = True

print KKseo_Sig[2]


    #     if np.mod(step_count, 2) == 1: # Step count
    #         step_start = True
    #     else:
    #         step_start = False
    #
    # if step_start:
    #     KKseo_Sig[step_count] = accelY[i]
    #     Label[step_count] = risk[i]






# for idx,key in enumerate(sorted(Label)):
#     Label[key] = sp.signal.resample(Label[key],128)

# for idx,key in enumerate(sorted(Label)):
#     print idx,key, Label[key]

#
# plt.plot(accelY)
# plt.show()
# print KKseo_Sig.keys()
#
# plt.figure('YoungMin')
# plt.plot(KKseo_Sig[1])
#
# plt.figure('KKseo')
# plt.plot(KKseo_Sig[14])
#
# plt.show()
