# Author: RenFun
# File: linear_regression04.py
# Time: 2021/06/05


# 导入sklearn库实现线性回归

import numpy as np
import matplotlib.pyplot as plt

points = np.genfromtxt('data.csv', delimiter=',')
x = points[:, 0]                # 第一列数据
y = points[:, 1]                # 第二列数据
num = len(points)               # len()函数求出列表points有多少个元素；count（）函数求出列表中指定元素出现的次数
print(num)                      # 共100个点
plt.scatter(x, y)

# 导入sklearn库
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# 数组新的shape属性应该要与原来的一致，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
x_new = x.reshape(-1, 1)            # 此时是将x变成列数为1的数组，若是x.reshape(1, -1)表示将x变成行数为1的数组
y_new = y.reshape(-1, 1)            # 变成一列
print(x_new, y_new)
lr.fit(x_new, y_new)                # 拟合线性模型，两个参数X，y。X是训练数据集，矩阵或者数组形式，y是标记
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

print(lr.coef_)                 # LinearRegression()函数的属性：线性模型的系数，多元线性回归模型时是（w1,w2,...wm）形式
print(lr.intercept_)            # LinearRegression()函数的属性：线性模型的截距

f = lr.predict(x_new)           # f为预测值，调用LinearRegression().predict(x)可以得到预测值
# f = x * lr.coef_[0] + lr.intercept_[0]
plt.plot(x, f, 'r')
plt.show()
