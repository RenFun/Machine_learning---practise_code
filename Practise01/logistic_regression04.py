# Author: RenFun
# File: logistic_regression04.py
# Time: 2021/09/24


# 调用sklearn库实现逻辑回归，数据集为鸾尾花
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# 加载数据鸾尾花数据集：一共150个样本，3种类别，每个类别50个样本，每个样本有4个属性描述，萼片和花瓣的长与宽，共4个属性
# 样本0-49为类别0，样本50-99为类别1，样本100-149为类别2
iris_x, iris_y = datasets.load_iris(return_X_y=True)
# 选取属性0和属性1
# 选择不同的属性进行训练和测试，得到的模型也不同
iris_x = iris_x[:, 0:2]
# print(iris_x, iris_y)
# 对数据进行切片：选取样本的前两个属性——花萼长度和花萼宽度
x_coordinate = iris_x[:, 0]
y_coordinate = iris_x[:, 1]

# 将数据集划分为训练集和测试集
iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(iris_x, iris_y, test_size=0.3)
# 定义逻辑回归模型
lr = LogisticRegression(penalty='l2', max_iter=1000)
# 使用训练集去拟合模型
lr.fit(iris_x_train, iris_y_train)
# 使用predict()函数对测试集进行预测
iris_y_predict = lr.predict(iris_x_test)


# 绘制图像1
plt.rcParams['font.sans-serif'] = ['SimHei']        # 用来正常显示中文，设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
plt.title("鸾尾花样本散列图")
plt.xlabel("花萼长度")
plt.ylabel("花萼宽度")
plt.scatter(x_coordinate[:50], y_coordinate[:50], c='green', marker='s', label='类别1')
plt.scatter(x_coordinate[50:100], y_coordinate[50:100], c='red', marker='o', label='类别2')
plt.scatter(x_coordinate[100:150], y_coordinate[100:150], c='blue', marker='v', label='类别3')
plt.legend(loc='upper left')
plt.savefig('iris_scatter.svg', dpi=300)
plt.show()


# 绘制图像2
plt.title("测试集分类结果图")
plt.xlabel("花萼长度")
plt.ylabel("花萼宽度")
# 横纵坐标轴采样的数值，数值的大小会影响分类的边界状况
N, M = 500, 500
x_min, x_max = x_coordinate.min() - 0.5, x_coordinate.max() + 0.5   # 第0列的范围
y_min, y_max = y_coordinate.min() - 0.5, y_coordinate.max() + 0.5   # 第1列的范围
t1 = np.linspace(x_min, x_max, N)
t2 = np.linspace(y_min, y_max, M)
# 生成网格采样点，规格是500*500
x, y = np.meshgrid(t1, t2)
x_test = np.stack((x.flat, y.flat), axis=1)
# 颜色列表
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
# 所有采样点的预测值
y_hat = lr.predict(x_test)
y_hat = y_hat.reshape(x.shape)                 # 使之与输入的形状相同
# 传入颜色列表并展现分类的边界
plt.pcolormesh(x, y, y_hat, cmap=cm_light)
# plt.scatter(x_coordinate, y_coordinate, cmap=cm_dark)
iris_x_test_coordinate1 = iris_x_test[:, 0]
iris_x_test_coordinate2 = iris_x_test[:, 1]
# category = ['样本0', '样本1', '样本2']
# markers = ['s', 'o', 'v']
plt.scatter(iris_x_test_coordinate1, iris_x_test_coordinate2, c=iris_y_test.ravel(), cmap=cm_dark)
# plt.scatter(iris_x_test_coordinate1, iris_x_test_coordinate2, cmap=cm_light)
# plt.scatter(x_coordinate[:50], y_coordinate[:50], c='green', marker='s', label='样本0')
# plt.scatter(x_coordinate[50:100], y_coordinate[50:100], c='red', marker='o', label='样本1')
# plt.scatter(x_coordinate[100:150], y_coordinate[100:150], c='blue', marker='v', label='样本2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# plt.legend(loc='upper left')
plt.savefig('logistic_regression04.svg', dpi=300)
plt.show()



# 多分类模型的性能指标：1.准确率 2.混淆矩阵
# 二分类模型的性能指标：1.准确率 2.查准率（precision）、召回率（recall）、P-R曲线、F1 4.ROC曲线、AUC
print('真实类别：', iris_y_test)
print('预测类别：', iris_y_predict)
print('准确率：%.2f' % accuracy_score(iris_y_test, iris_y_predict))
# print('查准率：%.2f' % precision_score(iris_y_test, iris_y_predict, average='micro'))
# print('召回率：%.2f' % recall_score(iris_y_test, iris_y_predict, average='micro'))
# print('F1：%.2f' % f1_score(iris_y_test, iris_y_predict, average='micro'))
