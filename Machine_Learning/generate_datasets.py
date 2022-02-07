# Author: RenFun
# File: generate_datasets.py
# Time: 2021/06/06


# 在验证算法模型时需要合适的数据集，可以通过numupy库中的函数生成所需要数据集
# 可生成如下模型的数据集：回归，分类，聚类，多组多维正态分布


import numpy as np
import matplotlib.pyplot as plt

# 分类问题
from sklearn.datasets import make_classification
# make_classification()函数返回两个参数X和y
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
# 主要参数如例，其余参数均为默认值。这里要注意n_redundant的默认值为2，若不明确写出，n_features的值必须大于2
print(X.shape, y.shape)     # X形式为（样本数量，每个样本的特征数）  y形式为（样本数量，）
print(X)                    # X的内容为（特征1， 特征2， ... ， 特征n_features）
print(y)                    # y内容为：每个样本的类别（0或1）
# 作图:以返回的X为依据
plt.scatter(X[:, 0], X[:, 1], c=y)          # scatter(x,y,c),其中x和y为点的位置; c为颜色,c=y意味着用两种颜色表示两个类
plt.show()

# 回归问题

