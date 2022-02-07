# Author: RenFun
# File: linear_regression03.py
# Time: 2021/06/04


# 最小二乘法中w和b的最优闭式可以利用矩阵的方法来求解：对于公式做向量化处理，再调用pycharm中专门处理矩阵的库进行运算求解
# 一元线性回归和多元线性回归其矩阵的表达公式是一致的，但是在多元线性回归中需要考虑公式中  逆矩阵  的不同情形
# 这里用到numpy库中有关矩阵的相关函数操作，需要额外再去学习

import numpy as np
import matplotlib.pyplot as plt

# 准备数据集
points = []             # 数据集（x，y）
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 11.5, 12, 13, 14.5, 15.5, 16.8, 17.3, 18, 18.7]
plt.scatter(x, y)                   # scatter（）依据点x和y，散列作图


# 将数据转化为向量形式：X，Y，W
# 其中W包含了w和b，X*W为Yi（预测值），Y包含了所有y值（标记，是列向量）
Y = np.array(y)                     # y是列表，这里把y变成一维数组Y
x = np.array(x)
print(Y.shape, x.shape)                      # Y.shape 是（10，）, x.shape是（10，）
x = np.mat(x).T                              # x变成矩阵，并转置
Y = np.mat(Y).T
one = np.mat(np.ones(len(x))).T              # 创建维数是len（x）的数组，元素全为1，然后变成矩阵并转置
X = np.hstack((x, one))          # np.vstack:按垂直方向（行数不断增加）堆叠数组构成一个新的数组
# ones()返回一个全1的n维数组，同样也有三个参数：
# shape（用来指定返回数组的大小）、dtype（数组元素的类型）、order（是否以内存中的C或Fortran连续（行或列）顺序存储多维数据）。
# 后两个参数都是可选的，一般只需设定第一个参数。和zeros一样
print(X)                                        # np.hstack:按水平方向（列数不断增加）堆叠数组构成一个新的数组
W = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))           # np.linalg.inv（）求矩阵的逆，np.dot（）求两个矩阵相乘
print(W)
Yi = np.dot(X, W)                   # x和Yi均解出，由此得到线性回归模型
print(Yi)
plt.plot(x, Yi, 'r')
plt.show()

# 补充知识点
# 1.数组，矩阵才能进行转置，求逆等操作，一维数组的转置需要注意
# 2.np.dot(a, b) ：矩阵相乘
# 3.np.multiply(a, b)：矩阵中对应元素相乘
# 4.a * b ：当为array的时候，默认a * b就是对应元素的乘积；当为mat的时候，默认a * b就是矩阵的乘积；混合的时候默认按照矩阵乘法
