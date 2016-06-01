'''
150501
Goal
- For given matrix data for each classes, compute the fisher score for each features.

Input
- Dictionary Matrix (List of Matrix)

Output
- Fisher score array

'''

from itertools import combinations
import numpy as np
import pandas as pd

class Fisher_Score_Compute:
    def __init__(self, Training):
        '''
        Training record
        - Dict.key : label
        - Dict.value : Each features. (n1 by p matrix)

        Goal
        - Find the distinguishable column p'
        '''
        self.Training = Training
        self.List_Training_Key = Training.keys()
        self.FeatureNum = len(self.Training[self.List_Training_Key[0]][0])

    def Compute_each_pair(self):
        All_Class_Fisher_Score = []
        for KeySubset in combinations(self.List_Training_Key, 2):
            Mat_Training_A_T = np.matrix(np.array(self.Training[KeySubset[0]])).T
            Mat_Training_B_T = np.matrix(np.array(self.Training[KeySubset[1]])).T

            Each_Class_Fisher_score = []
            idx = 0

            for (Feature_A, Feature_B) in zip(Mat_Training_A_T, Mat_Training_B_T):
                # All_Feature = np.concatenate([Feature_A[0], Feature_B[0]])
                Within_Var_each_feature = \
                    (np.var(Feature_A) + np.var(Feature_B))
                Between_Var_each_feature = \
                    (np.mean(Feature_A) - np.mean(Feature_B)) ** 2
                Fisher_Score_each_feature = \
                    Between_Var_each_feature / Within_Var_each_feature
                Each_Class_Fisher_score.append(Fisher_Score_each_feature)
                # print idx, "A Var", np.var(Feature_A), "B Var", np.var(Feature_B), "A Mean", np.mean(Feature_A), "B Mean", np.mean(Feature_B), "Within", Within_Var_each_feature, "Between", Between_Var_each_feature
                # print ""
                idx += 1
            All_Class_Fisher_Score.append(Each_Class_Fisher_score)
        return All_Class_Fisher_Score

    def Fisher_Score(self):
        # row sum for each score
        AllClassFisherScore = self.Compute_each_pair()
        Each_Feature_Score = np.array([float(0)] * self.FeatureNum)
        for each_class_pair in AllClassFisherScore:
            Each_Feature_Score += each_class_pair
        return Each_Feature_Score

    def FisherRatio(self):
        MyFisherOrder = self.FeatureOrder()
        MyFisherRatio = self.Fisher_Score()
        SUM = np.sum(MyFisherRatio)
        NewRatio = np.array([float(x) / SUM for x in MyFisherRatio], dtype='float32')[MyFisherOrder]
        return NewRatio

    def HowMany(self, Threshold):
        MyRatio = self.FisherRatio()
        M = float(0)
        IDX = 1
        for x in MyRatio:
            M += x
            if M > Threshold:
                break
            IDX += 1
        return IDX

    def FeatureOrder(self):
        FeatureScore = self.Fisher_Score()
        idx = np.argsort(FeatureScore)[::-1]
        return idx

    def FeatureSelector(self, Num):
        FeatureScore = self.Fisher_Score()
        idx = np.argsort(FeatureScore)[::-1]
        AllData = np.concatenate([x for x in self.Training.values()])
        OrderAllData = AllData[:,idx]
        # print pd.DataFrame(AllData)
        return idx, OrderAllData[:,:Num]


