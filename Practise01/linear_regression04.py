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


# 定义loss（w,b）函数
def loss():
    loss = 0
    for i in range(num):
        x = points[i, 0]
        y = points[i, 1]
        loss += pow((w * x + b - y), 2)/num
    return loss


# 导入sklearn库
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# 数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
x_new = x.reshape(-1, 1)            # 此时是将x变成列数为1的数组
y_new = y.reshape(-1, 1)
print(x_new, y_new)
lr.fit(x_new, y_new)
# y = w * x + b,其中w为coef（多元线性回归中为（w1,w2,...wm）），b为intercept（截距）
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

print(lr.coef_)                 # ？
print(lr.intercept_)

# 从训练好的模型中提取w和b
w = lr.coef_[0][0]
b = lr.intercept_[0]
print(w, b)

f = x * w + b
plt.plot(x, f, 'r')
plt.show()
