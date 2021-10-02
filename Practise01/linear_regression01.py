# Author: RenFun
# File: linear_regression01.py
# Time: 2021/06/02


# 简单的一个一元线性回归的实践，利用梯度下降的方法，步骤如下：
# 1.加载数据
# 2.写出损失函数
# 3.计算梯度并不断更新
# 4.求出最佳的w和b

import numpy as np          # 导入numpy库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


w = 0
b = 0
# 学习率learning_rate
lr = 0.0001
# 梯度下降的次数
num = 2000
# 算法结束还可以根据梯度向量的模是否收敛，即达到一个特定值（足够小）???

# 加载数据,data 和loss_list均为全局变量
data = []       # 初始化data列表，用于存放数据（x，y）
loss_list = []  # 初始化loss_list列表，用于存放loss值

for i in range(100):                        # 循环一百次，取100个样本（x，y）
    x = np.random.uniform(-10., 10.)
    # .uniform(low, high, size) : 从均匀分布[low，high]中随机采样，size为数量，缺省默认为1
    esp = np.random.normal(0, 1)
    # esp为偏置量，.normal(loc, scale, size):产生正态分布的数组，，loc为均值(期望)，scale为标准差，size为数量，缺省默认为1
    y = 1.41*x + 0.89 + esp
    data.append((x, y))                     # 每一次的随机数x经如上操作变成y，组成（x，y）添加进data
# 创立一个n维数组，将data列表转换为计算机可识别的矩阵
data = np.array(data)

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data[:, 0], data[:, 1], train_size=0.7)

# 损失函数loss（w，b）
def loss():
    loss = 0
    for i in range(70):
        x = train_x[i]
        y = train_y[i]
        loss += pow((w * x + b - y), 2)/100      # 100个点的坐标依次带入求和，当loss最小时，w和b为最佳值
    return loss


# 定义梯度：实际上w_gradient是loss（）对w求偏导，b_gradient是loss（）对b求偏导
def gradient():
    w_gradient = 0
    b_gradient = 0
    for i in range(70):                # 对全体数据集更新一次梯度，此时为批量梯度下降法
        x = train_x[i]
        y = train_y[i]
        w_gradient += (2/100) * ((w * x + b) - y) * x
        b_gradient += (2/100) * ((w * x + b) - y)       # 求偏导的式子化简后
    return w_gradient, b_gradient


# 更新梯度，以及w和b，最终找到最佳的w和b
# 梯度逐渐收敛-——越来越小？？？      while w_gradient < 0  ???
for j in range(num):
    w_gradient = 0
    b_gradient = 0
    w_gradient, b_gradient = gradient()          # 每次迭代都用gradient（）函数计算梯度
    # ？
    w = w - (lr * w_gradient)
    b = b - (lr * b_gradient)
    if j % 10 == 0:                            # 每10次输出一次当前的w，b以及loss值
        print('w_gradient', w_gradient, 'b_gradient',b_gradient, loss())
    if j % 40 == 0:                           # 每40次记录一次loss值，将其放入loss_list中
        loss_list.append(loss())
print('w = ', w, 'b = ', b)

# predict_train为训练集上的预测值
predict_train = w * train_x + b
# predict_test为测试集上的预测值
predict_test = w * test_x + b

# 绘制图像1
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("训练集散列图")
plt.xlabel("x坐标")
plt.ylabel("y坐标")
plt.scatter(train_x, train_y, label='训练集数据')
plt.plot(train_x, predict_train, 'r', label='拟合直线')
plt.legend(loc='upper left')
plt.show()

# 绘制图像2
plt.title("损失函数图")
plt.xlabel("梯度更新次数（*40）")
plt.ylabel("梯度值")
# 把x轴的刻度间隔设置为5，并存在变量里
x_major_locator = MultipleLocator(5)
# 把y轴的刻度间隔设置为5，并存在变量里
y_major_locator = MultipleLocator(5)
# ax为两条坐标轴的实例,plt.gca()是获取当前坐标
ax = plt.gca()
# 把x轴的主刻度设置为1的倍数
ax.xaxis.set_major_locator(x_major_locator)
# 把y轴的主刻度设置为5的倍数
ax.yaxis.set_major_locator(y_major_locator)
# 把x轴的刻度范围设置为-1到50，因为1不满一个刻度间隔(5)，所以数字不会显示出来，但是能看到一点空白
plt.xlim(-1, 50)
# 把y轴的刻度范围设置为-1到55，同理，-1不会标出来，但是能看到一点空白
plt.ylim(-1, 55)
plt.plot(loss_list)
plt.show()

# 模型的性能指标：均方误差，R指数
print('均方误差: %.2f' % mean_squared_error(test_y, predict_test))
print('决定系数：%.2f' % r2_score(test_y, predict_test))

