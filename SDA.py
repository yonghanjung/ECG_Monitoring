__author__ = 'jeong-yonghan'

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Class_SDA import SDA


''' Gaussian mixture data genearation '''
np.random.seed(100)
dim = 3
class_num = 2
mean = [np.ones((dim,1)),np.ones((dim,1)),np.ones((dim,1))]
mean[0][2]=4
mean[1][0]=3

print mean[0]
print mean[1]

num_each_class = [50,50,0]
obs_num = 100
cov = np.eye(dim)
X = list()

for idx in range(class_num):
    X.append(np.random.multivariate_normal(np.ravel(mean[idx]),cov,num_each_class[idx]))

# X = np.concatenate((X[0],X[1],X[2]),axis=0)
X = np.concatenate((X[0],X[1]),axis=0)

''' Construct Y '''
Y = np.zeros((obs_num, class_num))
for idx in range(len(Y)):
    if idx < num_each_class[0]:
        Y[idx][0] = 1
    elif idx > num_each_class[0] and idx < num_each_class[0]+num_each_class[1]:
        Y[idx][1] = 1
    # elif idx > num_each_class[0] + num_each_class[1]:
    #     Y[idx][2] = 1

''' 1. PCA '''
pca = PCA()
pca.fit(X)
print pca.explained_variance_ratio_
X_transform = pca.fit_transform(X)

# ''' 1-1. Figure '''
plt.figure(1, figsize=(10,8))
ax_pca = plt.subplot(211)
for idx in range(len(X)):
    if idx < 50:
        ax_pca.plot(X_transform[idx,0],X_transform[idx,1],'bo')
    else:
        ax_pca.plot(X_transform[idx,0],X_transform[idx,1],'ro')

# plt.show()


''' 2. SDA '''
''' Dict train construction '''
dict_train = dict()
dict_train[0] = list()
dict_train[1] = list()
for idx in range(len(X)):
    if idx < num_each_class[0]:
        dict_train[0].append(X[idx])
    elif idx > num_each_class[0] and idx < num_each_class[0] + num_each_class[1]:
        dict_train[1].append(X[idx])

dict_train[0] = np.array(dict_train[0])
dict_train[1] = np.array(dict_train[1])

obj_SDA = SDA(dict_train=dict_train,Flt_L1=1.8,Flt_Lambda=0.1)
sparse_dicriminant_matrix =  obj_SDA.sparse_discriminant_matrix
sparse_dicriminant_vector = obj_SDA.sparse_discriminant_vector
# print sparse_dicriminant_vector, sparse_dicriminant_matrix
reduced = np.dot(X,sparse_dicriminant_matrix)
print sparse_dicriminant_matrix

ax_sda = plt.subplot(212)
for idx in range(len(X)):
    if idx < 50:
        ax_sda.plot(reduced[idx,0],reduced[idx,2],'bo')
    else:
        ax_sda.plot(reduced[idx,0],reduced[idx,2],'ro')
plt.show()