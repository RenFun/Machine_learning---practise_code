# Author: RenFun
# File: Logistic_Reg_iris.py
# Time: 2021/09/24


# 调用sklearn库实现逻辑回归，数据集为鸾尾花
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


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
# plt.title("鸾尾花样本散列图")
plt.xlabel("花萼长度")
plt.ylabel("花萼宽度")
plt.scatter(x_coordinate[:50], y_coordinate[:50], c='green', marker='s', label='类别1')
plt.scatter(x_coordinate[50:100], y_coordinate[50:100], c='red', marker='o', label='类别2')
plt.scatter(x_coordinate[100:150], y_coordinate[100:150], c='blue', marker='v', label='类别3')
plt.legend(loc='upper left')
plt.savefig('Logistic_Reg_iris_scatter.svg')
plt.show()


# 绘制图像2
# plt.title("测试集分类结果图")
plt.xlabel("花萼长度")
plt.ylabel("花萼宽度")
# 横纵坐标轴采样的数值，数值的大小会影响分类的边界状况
N, M = 200, 200
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
plt.savefig('Logistic_Reg_iris_result.svg')
plt.show()


# 多分类模型的性能指标：1.准确率 2.混淆矩阵
# 二分类模型的性能指标：1.准确率 2.查准率（precision）、召回率（recall）、P-R曲线、F1 4.ROC曲线、AUC
print('真实类别：', iris_y_test)
print('预测类别：', iris_y_predict)
print('准确率：', accuracy_score(iris_y_test, iris_y_predict))
print('查准率：', precision_score(iris_y_test, iris_y_predict, average='macro'))
print('召回率：', recall_score(iris_y_test, iris_y_predict, average='macro'))
print('F1：', f1_score(iris_y_test, iris_y_predict, average='macro'))
print(classification_report(iris_y_test, iris_y_predict))


def plot_matrix(cm, classes, cmap=plt.cm.Blues):
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # str_cm = cm.astype(np.str).tolist()
    # for row in str_cm:
    #     print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,)

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 标注百分比信息
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if float(cm[i, j] * 100) > 0:
                ax.text(j, i, format(float(cm[i, j]), '.2f'),
                        ha="center", va="center",fontsize=15,
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xticks(rotation=30)
    plt.savefig('Logistic_Reg_iris_confusion_matrix.svg', bbox_inches='tight')
    plt.show()


confusion_mat = confusion_matrix(iris_y_test, iris_y_predict)
iris_target = ['Setosa', 'Versicolor', 'Virginica']
plot_matrix(confusion_mat, iris_target)
