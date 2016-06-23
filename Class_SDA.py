import numpy as np
from sklearn.linear_model import ElasticNet

class SDA:
    def __init__(self, dict_train, Flt_Lambda, Flt_L1):
        '''
        Implementing Algorithm 1 in Sparse Discriminant Analysis (Clemenson), 2012, Technometrics
        :param dict_train: dictionary of training data (key: 0 - normal / 1 - PVC)
        :param Flt_Lambda: L2 penalty value
        :param Flt_L1: L1
        :return: sparse discriminant vector
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
        I = np.eye(2)
        for k in range(1):
            theta_initial = np.random.random(2)
            # 4-(a) in Algorithm 1 in Sparse Discriminant Analysis (2012) by Clemenson, Tehcnometrics
            theta = np.dot(I - np.dot(np.dot(self.Q, np.transpose(self.Q)), self.D ), theta_initial)
            theta /= np.sqrt(np.dot(np.dot(np.transpose(theta), self.D), theta)) # normalize

            iteration_num = 10000
            beta_prev = np.random.random(self.dim)
            # 4-(b)
            for idx in range(iteration_num):
                response = np.dot(self.Y, theta)
                elas = ElasticNet(alpha=Flt_Lambda, l1_ratio=Flt_L1) # alpha * l1_ration = lambda // 0.5 * alpha * (1 - l1_ratio) = gamma
                beta = elas.fit(X=self.X, y= response).coef_

                theta_factor_1 = I - np.dot(np.dot(self.Q, np.transpose(self.Q)),self.D)
                theta_factor_2 = np.dot(theta_factor_1, np.linalg.inv(self.D))
                theta_factor_3 = np.dot(theta_factor_2, np.transpose(self.Y))
                theta_factor_4 = np.dot(np.dot(theta_factor_3, self.X), beta)
                # print WaveTheta
                theta = theta_factor_4 / np.sqrt(np.dot(np.dot(np.transpose(theta_factor_4),self.D),theta_factor_4))

                if np.sum(np.abs(beta - beta_prev)) < 1e-6:
                    break
                else:
                    beta_prev = beta

            # print B
            self.sparse_discriminant_vector = beta

        ''' 160612 constructing sparse discriminant 'matrix' for two class case for applying T2.'''
        sparse_discriminant_matrix = np.zeros((self.dim, self.dim))
        non_zero_elem = list()

        for idx in range(self.dim):
            sparse_discriminant_matrix[idx][idx] = self.sparse_discriminant_vector[idx]
            if self.sparse_discriminant_vector[idx] != 0.0:
                non_zero_elem.append(idx)
        self.sparse_discriminant_matrix = sparse_discriminant_matrix
        self.non_zero_elem = non_zero_elem


    def Construct_Y(self):
        '''
        Construct Y matrix in Algorithm 1 in Sparse Discriminant Analysis (2012) by Clemmensen (Technometrics)
        :return: Y matrix s.t Y[ij] = 1 if i th vector is in j th class (j=0 if normal, j=1 if PVC)
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