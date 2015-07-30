# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
from Statistics4 import ConstructStatistics
''' Function or Class '''

if __name__ == "__main__":

    List_RecordNum = [100, 105, 106, 108, 109, 113, 114, 116, 118, 119,
                      200, 201, 202, 203, 207, 208, 209, 210, 213, 214,
                      215, 219, 220, 221, 223, 228, 233]
    List_RecordType = [0,1]
    List_BoolLDAorNOT = [True, False]
    Alpha = [0.9973, 0.9999]
    IntDecompLevel = 4
    StrWaveletBasis = 'db6'
    TrainingSeconds = [300]


    print "RecordNum | RecordType | TrainingTime | Alpha | LDAorNOT | Threshold | Specificity | Sensitivity | Positive Predictivity | Accuracy"

    for IntRecordNum in List_RecordNum:
        for IntRecordType in List_RecordType:
            for alpha in Alpha:
                for BoolLDAorNOT in List_BoolLDAorNOT:
                    for IntSeconds in TrainingSeconds:
                        try :
                            if BoolLDAorNOT == True:
                                FltThreshold = 0.99
                            else:
                                FltThreshold = 0.8
                            ObjConstructStatistics = ConstructStatistics(IntRecordNum , IntRecordType, IntSeconds, StrWaveletBasis, IntDecompLevel, BoolLDAorNOT, FltThreshold, alpha)
                            _, DictFloat_Performance = ObjConstructStatistics.AccuracyComputation()
                            #print "RecordNum | RecordType | LDAorNOT | Threshold | Specificity | Sensitivity | Positive Predictivity | Accuracy"
                            if BoolLDAorNOT == True:
                                LDAORNOT = "True"
                            else:
                                LDAORNOT = "False"
                            print IntRecordNum,"|",IntRecordType,"|", IntSeconds, "|", alpha, "|"  ,LDAORNOT,"|",FltThreshold,"|",DictFloat_Performance['Specificity'], "|", DictFloat_Performance['Sensitivity'],"|",DictFloat_Performance["Positive_Predictivity"],"|", DictFloat_Performance['Accuracy']
                        except:
                            pass



