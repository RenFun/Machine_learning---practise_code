# Author: RenFun
# File: linear_regression01.py
# Time: 2021/06/02


# 简单的一个一元线性回归的实践，利用梯度下降的方法，步骤如下：
# 1.加载数据
# 2.写出损失函数
# 3.计算梯度并不断更新
# 4.求出最佳的w和b

import numpy as np          # 导入numpy库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
import matplotlib.pyplot as plt     # Matplotlib是 Python 的绘图库，pyplot是其一个子模块


w = 0
b = 0
lr = 0.0001     # 学习率learning_rate
num = 20000     # 梯度下降的次数

# 加载数据,data 和loss_list均为全局变量
data = []       # 初始化data列表，用于存放数据（x，y）
loss_list = []  # 初始化loss_list列表，用于存放loss值

for i in range(100):                        # 循环一百次，取100个样本（x，y）
    x = np.random.uniform(-10., 10.)        # .uniform(low, high, size) : 从均匀分布[low，high]中随机采样，size为数量，缺省默认为1
    esp = np.random.normal(0, 1)         # esp为偏置量，.normal(loc, scale, size):产生正态分布的数组，，loc为均值(期望)，scale为标准差，size为数量，缺省默认为1
    y = 1.41*x + 0.89 + esp
    data.append((x, y))                     # 每一次的随机数x经如上操作变成y，组成（x，y）添加进data
data = np.array(data)                       # 创立一个n维数组，将data列表转换为计算机可识别的矩阵


# 损失函数loss（w，b）
def loss():
    loss = 0
    for i in range(100):
        x = data[i, 0]
        y = data[i, 1]
        loss += pow((w * x + b - y), 2)/100      # 100个点的坐标依次带入求和，当loss最小时，w和b为最佳值
    return loss


# 定义梯度：实际上就是loss函数分别对w和b求偏导
def gradient():
    w_gradient = 0
    b_gradient = 0
    for i in range(100):
        x = data[i, 0]
        y = data[i, 1]
        w_gradient += (2/100) * ((w * x + b) - y) * x
        b_gradient += (2/100) * ((w * x + b) - y)       # 求偏导的式子化简后
    return w_gradient, b_gradient


# 更新梯度，以及w和b，最终找到最佳的w和b
for j in range(num):
    w_gradient = 0
    b_gradient = 0
    w_gradient,b_gradient = gradient()          # 每次迭代都用gradient（）函数计算梯度
    # ？
    w = w - (lr * w_gradient)
    b = b - (lr * b_gradient)
    if j % 100 == 0:                            # 每100次输出一次当前的w，b以及loss值
        print('w_gradient', w_gradient, 'b_gradient',b_gradient, loss())
    if j % 4000 == 0:                           # 每4000次记录一次loss值，将其放入loss_list中
        loss_list.append(loss())
print('w = ', w, 'b = ', b)


m = data[:, 0]      # 第一列数据，实质为x横坐标
n = data[:, 1]      # 第二列数据，实质为y纵坐标
plt.scatter(m, n)           # 将data中的每一个（x，y）取出，画出散列点


m = np.arange(-10, 10, 0.2)         # m人为取值从-10至10，步长为0.2，若此条删除则为实际m的取值范围[-10,10],
f = w * m + b                       # f为预测值，w和b确定的线性回归模型
plt.plot(m, f, 'r')
plt.show()

plt.plot(loss_list)                 # loss曲线
plt.show()


