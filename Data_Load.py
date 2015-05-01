'''
Date : 20150501
Author : Yonghan Jung, IE KAIST
Goal : Data Load
'''

import scipy.io
import numpy as np

class Data_Loading:
    def __init__(self, datanum, record_type = 0, Sampling_rate = 360):
        self.datanum = datanum
        self.record_type = record_type
        self.sampling_rate = Sampling_rate

    def Data_Loading(self):
        '''
        Goal : Load the ECG mat (MIT BIH)
        Output
        - Time domain : Array
        - ECG_Dyadic_Sample = np.array
        - DyadLenth = integer
        '''
        try:
            mat = scipy.io.loadmat('Data/'+str(self.datanum)+'_file.mat')
        except:
            mat = scipy.io.loadmat('../Data/'+str(self.datanum)+'_file.mat')
        if self.record_type == 0:
            ECG = mat['val'][0]
        else:
            ECG = mat['val'][1]
        Dyad_length = 2**(int(np.log(len(ECG)) / np.log(2)))
        ECG_Dyadic_Sample = np.array(ECG[:Dyad_length])
        Time_domain = np.array([x / float(self.sampling_rate) for x in range(len(ECG_Dyadic_Sample))])
        return Time_domain, ECG_Dyadic_Sample, Dyad_length