# Author: RenFun
# File: test01.py
# Time: 2021/06/05

# 导入linear_regression02中的数据集，利用linear_regression03中矩阵的方法，验证结果相同


import numpy as np
import matplotlib.pyplot as plt

points = np.genfromtxt('data.csv', delimiter=',')               # 导入data.vsc文件，里面存储着事先准备的数据集
x = points[:, 0]                # 第一列数据
y = points[:, 1]

Y = np.array(y)
print(Y)
print(Y.shape)
X = np.vstack((x, np.ones(len(x)))).T
print(X)
W = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))               # np.linalg.inv（）求矩阵的逆，np.dot（）求两个矩阵相乘
print(W)
print(W.shape)                      # W为2*1的向量，两行一列，第一行为w，第二行为b
Yi = np.dot(X, W)                   # x和Yi均解出，由此得到线性回归模型
print(Yi)
plt.plot(x, Yi, 'r')
plt.scatter(x, y)
plt.show()
