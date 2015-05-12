from HansFisherScore import Fisher_Score_Compute
import numpy as np

Training = {}
Training[1] = [[10,2,3],[30,4,5],[50,12,11]]
Training[2] = [[5,100,3],[3,100,5],[1,100,6],[5,100,2],[12,100,1],[12,100,1],[12,100,1],[12,100,1]]
# Training[3] = [[2,1000,5],[3,2621,4]]

A = Fisher_Score_Compute(Training=Training)
print A.Fisher_Score()
for x in  A.FisherRatio():
    print x

print A.HowMany(0.9)
