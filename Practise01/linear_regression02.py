# Author: RenFun
# File: linear_regression02.py
# Time: 2021/06/03


# 简单的一元线性回归模型实践，利用最小二乘法：
# 1.w和b的表达式用代码表示
# 2.拟合函数求最佳w和b
# 3.由确定的w和b求出loss函数

import numpy as np
import matplotlib.pyplot as plt


w = 0
b = 0
# 导入数据集
points = np.genfromtxt('data.csv', delimiter=',')               # 导入data.vsc文件，里面存储着事先准备的数据集
x = points[:, 0]                # 第一列数据
y = points[:, 1]                # 第二列数据


num = len(points)               # len()函数求出列表points有多少个元素；count（）函数求出列表中指定元素出现的次数
print(num)                      # 共100个点


# 定义loss（w,b）函数
def loss():
    loss = 0
    for i in range(num):
        x = points[i, 0]
        y = points[i, 1]
        loss += pow((w * x + b - y), 2)/num
    return loss


# 公式各个部分分别表示
sum_y = sum(points[:, 1])
sum_x = sum(points[:, 0])
x_bar = sum_x/num               # 所有x的取值求和除以num，求得x的平均值
# print(x_bar)
sum_yx = 0
sum_yx_bar = 0
sum_xx = 0
sum_wx = 0
for i in range(num):
    sum_yx += points[i, 0] * points[i, 1]
    sum_yx_bar += points[i, 1] * x_bar
    sum_xx += pow((points[i, 0]), 2)
# print(sum_yx, sum_yx_bar, sum_xx, pow(sum_x, 2)/num)
w = (sum_yx - sum_yx_bar) / (sum_xx - pow(sum_x, 2)/num)
for i in range(num):
    sum_wx += w * points[i, 0]
b = sum_y/num - sum_wx/num
print('w = ', w, 'b = ', b)
print(loss())

# 线性回归图像
f = w * x + b
plt.plot(x, f, 'r')
plt.scatter(x, y)               # 用 scatter 画出散点图
plt.show()
