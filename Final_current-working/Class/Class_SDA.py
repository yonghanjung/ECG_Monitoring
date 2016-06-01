import numpy as np
from sklearn.linear_model import ElasticNet

class SDA:
    def __init__(self, dict_train, Flt_Lambda, Flt_L1):
        '''

        :param dict_train: dictionary of training data (key: 0 - normal / 1 - PVC)
        :param Flt_Lambda: L2 penalty value
        :param Flt_L1: L1
        :return:
        '''
        self.mat_wc_normal = dict_train[0] # N by 256 matrix
        self.mat_wc_PVC = dict_train[1] # V by 256 matrix
        self.dim = len(self.mat_wc_normal[0]) # 256

        self.X = np.concatenate((self.mat_wc_normal, self.mat_wc_PVC), axis=0) # N / V augmented matrix (transpose of [N|V])

        self.number_normal = len(self.mat_wc_normal) # N
        self.number_PVC = len(self.mat_wc_PVC) # V
        self.number_total = self.number_normal + self.number_PVC

        self.Y = self.Construct_Y()
        self.D = np.dot(np.transpose(self.Y), self.Y) / float(self.number_total) # P
        self.Q = np.ones((2,1))

        np.random.seed(123)
        InitialTheta = np.random.random(2)
        I = np.eye(2)
        Theta = np.dot(I - np.dot(np.dot(self.Q, np.transpose(self.Q)), self.D ), InitialTheta)
        Theta /= np.sqrt(np.dot(np.dot(np.transpose(Theta), self.D), Theta))

        MaxIter = 10000
        PrevTheta = InitialTheta
        PrevB = np.ones(self.dim)
        for idx in range(MaxIter):
            NewResp = np.dot(self.Y, Theta)
            elas = ElasticNet(alpha=Flt_Lambda, l1_ratio=Flt_L1)
            #
            # # Compute Coefficient
            # B = lasso.fit(X=self.X, y= NewResp).coef_
            B = elas.fit(X=self.X, y= NewResp).coef_
            # print B
            #
            # New OptScore
            Part1 = I - np.dot(np.dot(self.Q, np.transpose(self.Q)),self.D)
            Part2 = np.dot(Part1, np.linalg.inv(self.D))
            Part3 = np.dot(Part2, np.transpose(self.Y))
            WaveTheta = np.dot(np.dot(Part3, self.X), B)
            # print WaveTheta
            Theta = WaveTheta / np.sqrt(np.dot(np.dot(np.transpose(WaveTheta),self.D),WaveTheta))

            if np.sum(np.abs(B - PrevB)) < 1e-6:
                break
            else:
                PrevB = B

        # print B
        self.B = B 

    def Construct_Y(self):
        '''
        Construct Y matrix in Algorithm 1 in Sparse Discriminant Analysis (2012) by Clemmensen (Technometrics)
        :return: Y matrix
        '''

        Y = np.zeros((self.number_total, 2))
        for idx in range(len(Y)):
            if idx < self.number_normal:
                Y[idx][0] = 1
            else:
                Y[idx][1] = 1

        return Y


if __name__ == "__main__":
    print None