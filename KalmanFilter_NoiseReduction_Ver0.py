# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 150712 Kalman Filter for Noise Reduction
Comment 
- 

'''

''' Library '''
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
''' Function or Class '''

class HansKalmanFilter:
    def __init__(self, Vector_Obs, Matrix_Train):
        def Compute_HatMatrix():
            try:
                CoefMatrix = np.linglg.inv(self.Matrix_Train.T * self.Matrix_Train) * self.Matrix_Train.T * self.Vector_Obs
                return CoefMatrix
            except:
                Matrix_Reg = np.matrix(np.eye(self.Int_NumTrain) * 1e-4)
                CoefMatrix = (self.Matrix_Train.T * self.Matrix_Train + Matrix_Reg).I * self.Matrix_Train.T * self.Vector_Obs
                return CoefMatrix

        def Estimate_ProcessNoise():
            Matrix_TempTrain = self.Matrix_Train.T
            return np.matrix(np.diag(np.var(Matrix_TempTrain, axis=0)))

        def Normalizing_Columnwise(Matrix):
            ColSum = np.sum(Matrix,axis=1)
            ColSum = [float(val) for val in ColSum]
            ColSum = np.reshape(ColSum, (len(ColSum),1))
            return Matrix/ColSum



        # Each Column : Record. Column1, Column2...
        self.Matrix_Train = np.matrix(Matrix_Train)

        # self.Matrix_Train = Normalizing_Columnwise(Matrix_Train)
        self.Vector_Obs = Vector_Obs


        self.Int_SignalLen = len(self.Matrix_Train)
        self.Int_NumTrain = len(self.Matrix_Train.T)

        self.Coefficient = Compute_HatMatrix()
        SHAPE = self.Coefficient.shape
        SUM = np.sum(self.Coefficient)
        self.Coefficient = np.reshape(np.asarray([val/SUM for val in self.Coefficient]), SHAPE)
        self.Matrix_Q = Estimate_ProcessNoise()

        print self.Coefficient.shape


    def KalmanPredict(self, Vector_MeanEst, Matrix_CovEst):
        # print len(Vector_MeanEst)
        # print Vector_MeanEst.shape
        Vector_MeanPredict = Vector_MeanEst
        Matrix_CovPredict = Matrix_CovEst + self.Matrix_Q
        return (Vector_MeanPredict, Matrix_CovPredict)

    def Compute_MatrixR(self):
        Vector_Obs = np.reshape(self.Vector_Obs, (len(self.Vector_Obs),1))
        Vector_ObsHat = self.Matrix_Train * self.Coefficient
        Vector_Residual = Vector_Obs - Vector_ObsHat
        Flt_MeasureNoise = np.squeeze(np.asarray(Vector_Residual.T * Vector_Residual)) / float(self.Int_SignalLen - self.Int_NumTrain)
        Matrix_R = np.matrix(np.eye(self.Int_SignalLen) * Flt_MeasureNoise)
        return Matrix_R


    def KalmanUpdate(self, Vector_MeanPredict, Matrix_CovPredict, Vector_MeanEst):
        Vector_Obs = self.Vector_Obs
        Matrix_R = self.Compute_MatrixR()
        Vector_Residual = Vector_Obs - Vector_MeanPredict
        Matrix_ResiaulCov = Matrix_CovPredict + Matrix_R
        try:
            Matrix_K = Matrix_CovPredict * (Matrix_ResiaulCov).I
        except:
            Matrix_Reg = np.matrix(np.eye(self.Int_SignalLen) * 1e-5)
            Matrix_K = Matrix_CovPredict * (Matrix_ResiaulCov + Matrix_Reg).I
        Vector_MeanEst = Vector_MeanEst + Matrix_K * Vector_Residual
        Matrix_CovEst = Matrix_CovPredict - Matrix_K * Matrix_ResiaulCov * Matrix_K.T

        return (Vector_MeanEst, Matrix_CovEst)

    def Execute_Kalman(self, Int_Niter):
        # Initializaiton
        # Vector_MeanEst = self.Matrix_HatMatrix * Vector_Obs
        Vector_Obs = self.Vector_Obs
        Vector_Obs = np.reshape(Vector_Obs,(len(Vector_Obs),1))
        # Vector_MeanEst = np.reshape(np.ones(self.Int_SignalLen), (self.Int_SignalLen,1))
        # print self.Matrix_HatMatrix.shape
        # print Vector_Obs.shape
        Vector_MeanEst = self.Matrix_Train * self.Coefficient
        # Vector_MeanEst = np.ones(self.Int_SignalLen)
        # print pd.DataFrame(self.Matrix_HatMatrix)
        Matrix_CovEst = np.matrix(np.eye(self.Int_SignalLen))

        for IntIdx in range(Int_Niter):
            print IntIdx
            (Vector_MeanPredict, Matrix_CovPredict) = self.KalmanPredict(Vector_MeanEst=Vector_MeanEst, Matrix_CovEst=Matrix_CovEst)
            (Vector_MeanEst, Matrix_CovEst) = self.KalmanUpdate(Matrix_CovPredict=Matrix_CovPredict, Vector_MeanPredict=Vector_MeanPredict, Vector_MeanEst = Vector_MeanEst)
        return (Vector_MeanEst, Matrix_CovEst)


def Generate_NoiseData(Int_SignalLength, Int_NumRecord, Matrix_Clean):
    Matrix_NoisyData = []
    for IntIdx in range(Int_NumRecord):
        Vector_Clean = np.array(Matrix_Clean.T[IntIdx])
        Noise = np.random.normal(10000,4000, Int_SignalLength)
        Vector_Obs = np.squeeze(np.asarray(np.reshape(Vector_Clean + Noise, (1, Int_SignalLength))))
        Matrix_NoisyData.append(Vector_Obs)
    Matrix_NoisyData = np.matrix(Matrix_NoisyData)
    Matrix_NoisyData = Matrix_NoisyData.T
    return Matrix_NoisyData

def Generate_CleanData(Int_SignalLength, Int_NumRecord):
    Matrix_CleanData = []

    # Initial Signal
    x = np.linspace(0,20,Int_SignalLength)
    y = 3000*np.cos(2*x) + 2000*np.sin(3*x)
    Vector_InitialSignal = y

    for IntIdx in range(Int_NumRecord):
        ProcessNoise = np.random.normal(0,5, Int_SignalLength)
        y += ProcessNoise
        Matrix_CleanData.append(Vector_InitialSignal)
    Matrix_CleanData = np.matrix(Matrix_CleanData)
    Matrix_CleanData = Matrix_CleanData.T
    return Matrix_CleanData

if __name__ == "__main__":
    # Generate Data
    Int_SigLen = 250
    Int_RecordNum = 40
    Int_KalmanIter = 100

    Matrix_Clean = Generate_CleanData(Int_SignalLength= Int_SigLen, Int_NumRecord=Int_RecordNum)
    Matrix_Obs = Generate_NoiseData(Int_SignalLength= Int_SigLen, Int_NumRecord=Int_RecordNum,Matrix_Clean= Matrix_Clean)

    VectorObs = Matrix_Obs.T[30].T
    print VectorObs.shape
    VectorClean = Matrix_Clean.T[30].T

    # Object
    Object_HansKalman = HansKalmanFilter(Vector_Obs=VectorObs, Matrix_Train=Matrix_Clean)
    Vector_KalmanEst, _ = Object_HansKalman.Execute_Kalman(Int_KalmanIter)
    print np.sum(np.abs(Vector_KalmanEst - VectorClean))

    plt.grid()
    plt.plot(VectorObs,'b')
    plt.plot(VectorClean,'g')
    plt.plot(Vector_KalmanEst,'r')
    plt.show()


