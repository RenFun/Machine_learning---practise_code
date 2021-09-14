# Author: RenFun
# File: SVM01.py
# Time: 2021/07/03


# 支持向量机SVM，算法的结果是找到一个超平面将不同类别的样本分开，需要确定两个参数：w和b
import numpy as np
import matplotlib.pyplot as plt


# make_classification()函数返回两个参数X和Y
from sklearn.datasets import make_classification
X, Y = make_classification(n_samples=20, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
# 主要参数如例，其余参数均为默认值。这里要注意n_redundant的默认值为2，若不明确写出，n_features的值必须大于2
# print(X.shape, Y.shape)     # X形式为（样本数量，每个样本的特征数）  Y形式为（样本数量，）
# print(X)                    # X的内容为（特征1， 特征2， ... ， 特征n_features）
# print(Y)                    # Y内容为：每个样本的类别（0或1）
# 作图:以返回的X为依据
plt.scatter(X[:, 0], X[:, 1], c=Y)          # scatter(x,Y,c),其中x和Y为点的位置; c为颜色,c=Y意味着用两种颜色表示两个类
plt.show()


# 使用拉格朗日乘子法得到对偶问题
w = []              # w为超平面的法向量，样本由两个属性描述，w就是二维列向量
b = 0               # b为偏移量，w.T * X + b = 0
alpha = []          # 拉格朗日乘子，维数等于样本数量



