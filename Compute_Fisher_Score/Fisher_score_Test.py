from Fisher_Score_Computation import Fisher_Score_Compute
import numpy as np

Training = {}
Training[1] = [[1,2,3],[3,4,5]]
Training[2] = [[5,100,3],[3,100,5],[1,100,6],[5,100,2],[12,100,1],[12,100,1],[12,100,1],[12,100,1]]
# Training[3] = [[2,1000,5],[3,2621,4]]

A = Fisher_Score_Compute(Training=Training).Fisher_Score()
print A
