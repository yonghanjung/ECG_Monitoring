# -*- coding: utf-8 -*-
'''
Goal : Bootstrap 을 통한 SHT, CUSUM 분포추정
Author : Yonghan Jung, ISyE, KAIST 
Date : 150824
Comment 
- 

'''

''' Library '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import gaussian_kde

''' Function or Class '''


class CUSUM_HotellingT:
    def __init__(self, K):
        self.K = K
        self.CUSUM_distribution = self.Generate_CUSUM_Distribution()


    # CDF 를 구해줘야 한다.
    # 여기서 Quantile 을 계산할 수 있을까? 이 variable 이 몇번째 퀀타일에 있는지 알 수 있을까?
    def Compute_AlphaThreshold(self, Alpha):
        Array = self.CUSUM_distribution
        return np.percentile(Array,Alpha)

    def Compute_PValue(self,NewVal):
        Array = self.CUSUM_distribution
        List_Array = list(Array)
        if NewVal not in List_Array :
            List_Array.append(NewVal)
            LocIndex = len(List_Array)-1
        else:
            for idx in range(len(List_Array)):
                if List_Array[idx] == NewVal :
                    LocIndex = idx
                    break

        Rank = rankdata(List_Array)[LocIndex]
        Rank = np.squeeze(np.asarray(Rank))
        Pvalue = Rank / float(len(List_Array))

        return Pvalue

    def Generate_CUSUM_Distribution(self):
        SamplingIter = 10
        Bootstrap_Iter = 10000

        P = 1.0
        M = 375.0
        T = self.K
        ArrayMat_CUSUMVal = list()

        for SampleIdx in range(SamplingIter):
            SampleBox = list()
            # T SUM statistics 를 Bootstrap iter 만큼 만들자.
            for idx in range(Bootstrap_Iter):
                # T sample generation
                FSamples = np.random.f(P, M, T)
                # print FSamples
                TSamples = ((P * ((M-1) ** 2)) / (M * (M-P))) * FSamples
                NewSample = np.sum(TSamples)
                SampleBox.append(NewSample)
            ArrayMat_CUSUMVal.append(SampleBox)
        ArrayMat_CUSUMVal = np.array(ArrayMat_CUSUMVal)
        Array_CUSUMVal = np.mean(ArrayMat_CUSUMVal, axis= 0)
        # print np.percentile(Array_CUSUMVal, 99)
        # print np.max(Array_CUSUMVal)
        return Array_CUSUMVal







if __name__ == "__main__":
    Obj = CUSUM_HotellingT(K=5)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)
    print Obj.Compute_PValue(10)




