from Fisher_Score_Computation import Fisher_Score
import numpy as np

Training = {}
Training[1] = [[1,2,3],[4,5,6],[7,8,9]]
Training[2] = [[5,100,3],[3,133,5],[1,223,6],[5,152,2],[12,233,1]]
Training[3] = [[2,1000,5],[3,2621,4]]

A = Fisher_Score(Training=Training).Fisher_Score()
print A
