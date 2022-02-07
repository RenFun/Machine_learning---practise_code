# Author: RenFun
# File: linear_regression04.py
# Time: 2021/06/05


# 导入sklearn库实现一元线性回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 加载数据集
points = np.genfromtxt('data.csv', delimiter=',')
x = points[:, 0]                  # 第一列数据
y = points[:, 1]                  # 第二列数据
# 数组新的shape属性应该要与原来的一致，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
x = x.reshape(-1, 1)              # 此时是将x变成列数为1的数组，若是x.reshape(1, -1)表示将x变成行数为1的数组
y = y.reshape(-1, 1)              # 变成一列

# 将数据集划分为训练集和测试集，训练集比例为0.7
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# 定义线性回归模型
lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)

# 拟合线性模型，两个参数X，y，X是训练数据集，矩阵或者数组形式，y是标记
lr.fit(x_train, y_train)
# lr.coef_是系数，lr.intercept_是截距
print('系数：', lr.coef_)
print('截距：', lr.intercept_)
# y_predict为通过拟合模型进行预测得到的预测值
y_predict = lr.predict(x_test)

# 绘制图像1
# 用来正常显示中文，设置字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.title("训练集散列图和拟合的模型")
plt.xlabel("训练集的x坐标")
plt.ylabel("训练集的y坐标")
# 绘制训练集散列图
plt.scatter(x_train, y_train, label="训练集数据")
# plt.show()
# 绘制拟合的线性模型曲线
plt.plot(x_train, lr.coef_ * x_train + lr.intercept_, 'r', label="拟合曲线")
# plt.savefig("Linear_Reg_data.svg")
# plt.legend(loc='upper left')
plt.show()

# 绘制图像2
# plt.title("预测值与真实值的关系")
plt.xlabel("测试集样本")
plt.ylabel("预测值和真实值")
# 生成一个序列，1-30，步长是1
t = np.arange(1, len(x_test)+1, 1)
plt.plot(t, y_test, color='blue', label="真实值")
plt.plot(t, y_predict, color='red', label="预测值")
# plt.legend(loc='upper left')
plt.show()

# 模型的性能指标：均方误差，决定系数
print('均方误差: %.2f' % mean_squared_error(y_test, y_predict))
print('决定系数：%.2f' % r2_score(y_test, y_predict))
