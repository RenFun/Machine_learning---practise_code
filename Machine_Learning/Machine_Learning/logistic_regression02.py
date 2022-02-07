# Author: RenFun
# File: logistic_regression02.py
# Time: 2021/06/08


# 利用梯度下降的方法求得对数几率回归中的参数w和b：
# 加载数据集
# 表示线性回归模型
# sigmoid函数
# 确定梯度，并更新梯度
# 损失函数图像


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


# 创建数据集:(x1,x2,y),每个x有两个特征，y为标记，取0或1，一共有200个样本
x, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1)
lr = 0.000001
num = 20000
loss_list = []
# print(x)
# print(y)
plt.scatter(x[:, 0], x[:, 1], c=y)           # 以数据集绘制图像
plt.show()


# 表示线性回归模型中的初始值
w = np.random.normal(0, 1, 3)
print(w)            # w内容为(w1,w2,b)
b = w[2]
print('b=', w[2])
B = np.array(w[2])          # 将数字b变成向量B
X = np.vstack((x.T, np.ones(len(x))))           # 将（x；1）写成向量模型
# # print('X=', X)
z = np.dot(w.T, X)              # 二元线性模型：w.T * x
# # print('z=', z)
#
#
# sigmoid 函数
Y = 1.0/(np.exp(-z)+1.0)
# print('sigmoid函数：', Y)
# # plt.plot(z, Y)
# # plt.show()


# 确定梯度，这里都是矩阵运算
w_gradient = np.dot(x.T, (Y-y))
b_gradient = Y-y
print('w的梯度', w_gradient)
print('b的梯度', b_gradient)
for i in range(200-1):              # 将B扩充，以便于后面进行计算
    B = np.append(B, b)
# print(B)


# 损失函数
def loss():
    loss = 0
    for i in range(200):
        X = np.vstack((x.T, np.ones(len(x))))
        z = np.dot(w.T, X)
        Y = 1.0 / (np.exp(-z) + 1.0)
        loss += (-y[i]) * np.log(Y[i] - (1-y[i]) * np.log(1-Y[i]))
    return loss


# 更新梯度
for j in range(num):
    w[0] = w[0] - lr * w_gradient[0]
    w[1] = w[1] - lr * w_gradient[1]
    # 更新b的梯度
    B = B - lr * b_gradient
    if j % 400 == 0:                           # 每400次记录一次loss值，将其放入loss_list中，loss值应该不断缩小
        loss_list.append(loss())
print(loss_list)
print('w0:', w[0], 'w1:', w[1])
print('b:', B)
plt.plot(loss_list)
plt.show()
