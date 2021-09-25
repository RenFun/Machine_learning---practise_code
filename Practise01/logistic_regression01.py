# Author: RenFun
# File: logistic_regression01.pY
# Time: 2021/06/05


# 对数几率回归：用于处理二分类问题
# 利用sklearn库实现

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#
# # 加载数据鸾尾花数据集：一共150个样本，3种类别，每个类别50个样本，每个样本有4个属性描述，萼片和花瓣的长与宽，共4个属性
# # 样本0-49为类别0，样本50-99为类别1，样本100-149为类别2
# iris_x, iris_y = datasets.load_iris(return_X_y=True)
#

# 利用sklearn库创建数据集
from sklearn.datasets import make_classification
# make_classification()函数返回两个参数X和Y
X, Y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
# 主要参数如例，其余参数均为默认值。这里要注意n_redundant的默认值为2，若不明确写出，n_features的值必须大于2
print(X.shape, Y.shape)     # X形式为（样本数量，每个样本的特征数）  Y形式为（样本数量，）
print(X)                    # X的内容为（特征1， 特征2， ... ， 特征n_features）
print(Y)                    # Y内容为：每个样本的类别（0或1）
# 作图:以返回的X为依据
plt.scatter(X[:, 0], X[:, 1], c=Y)          # scatter(x,Y,c),其中x和Y为点的位置; c为颜色,c=Y意味着用两种颜色表示两个类
plt.show()

# 利用sklearn库完成对数几率回归

lr = LogisticRegression()
# 将数据集进行划分，分成训练集和测试集，其比例为4：1.
x_trian, x_test, y_trian, y_test = train_test_split(X, Y, test_size=0.25)
x_trian_x = x_trian[:, 0]
x_trian_y = x_trian[:, 1]
plt.scatter(x_trian_x, x_trian_y, c=y_trian)           # 展示训练数据集
plt.show()
lr.fit(x_trian, y_trian)                # 利用训练数据集进行拟合
f = lr.predict(x_test)                  # 利用测试集进行预测，返回的是每个样本的类别

# 显示测试数据集，同时得到性能指标（平均精度）
x_test_x = x_test[:, 0]
x_test_y = x_test[:, 1]
plt.scatter(x_test_x, x_test_y, c=y_test)
plt.show()
s = lr.score(x_test, y_test)
print(s)
# 展示参数w和b
print(lr.coef_)                     # 表示直线的斜率
print(lr.intercept_)                # 表示直线的截距
print(lr.classes_)                  # 表示类别，此时为一个二分类，（0， 1）

# 如何用一条直线将数据集分开？

# plt.plot(X,X*lr.coef_+lr.intercept_)
# plt.show()



