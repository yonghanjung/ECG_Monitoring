'''
150501
Goal
- For given matrix data for each classes, compute the fisher score for each features.

Input
- Dictionary Matrix (List of List)

Output
- Fisher score array

'''

from itertools import combinations
import numpy as np

class Fisher_Score:
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
            # For each class
            Mat_Training_A_T = np.matrix(self.Training[KeySubset[0]]).T
            Mat_Training_B_T = np.matrix(self.Training[KeySubset[1]]).T

            Each_Class_Fisher_score = []

            for (Feature_A, Feature_B) in zip(Mat_Training_A_T, Mat_Training_B_T):
                # Each Column is a feature.
                Within_Var_each_feature = \
                    np.var(Feature_A) + np.var(Feature_B)
                Between_Var_each_feature = \
                    np.abs(np.mean(Feature_A) - np.mean(Feature_B)) ** 2
                Fisher_Score_each_feature = \
                    Between_Var_each_feature / Within_Var_each_feature
                Each_Class_Fisher_score.append(Fisher_Score_each_feature)
            All_Class_Fisher_Score.append(Each_Class_Fisher_score)
        return All_Class_Fisher_Score

    def Fisher_Score(self):
        # row sum for each score
        AllClassFisherScore = self.Compute_each_pair()
        Each_Feature_Score = np.array([float(0)] * self.FeatureNum)
        for each_class_pair in AllClassFisherScore:
            Each_Feature_Score += each_class_pair
        return Each_Feature_Score

