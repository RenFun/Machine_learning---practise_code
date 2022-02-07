# Author: RenFun
# File: MLP_iris.py
# Time: 2021/10/15


# 利用sklearn实现多层感知机分类任务（MLPClassifier），数据集为鸢尾花

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=2)
# 创建多层感知类，设置单隐层神经元个数为50，使用梯度下降方法，激活函数为logistic()， 迭代200次，初始化学习率为0.01
# 不同的参数选择会影响模型的性能：隐层神经元个数hidden_layer_sizes，迭代次数max_iter，学习率初始值learning_rate_init
mlp = MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', activation='logistic', max_iter=200, learning_rate_init=0.01)
mlp.fit(x_train, y_train)
y_predict = mlp.predict(x_test)
iris_target = ['Setosa', 'Versicolor', 'Virginica']
print(classification_report(y_test, y_predict, target_names=iris_target))
print(confusion_matrix(y_test, y_predict))
print(precision_score(y_test, y_predict, average='macro'))
print(recall_score(y_test, y_predict, average='macro'))
print(f1_score(y_test, y_predict, average='weighted'))
# print(mlp.loss_)


# 绘制图像1：混淆矩阵
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
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xticks(rotation=30)
    plt.savefig('MLP_iris_confusion_matrix.svg', bbox_inches='tight')
    plt.show()


# 调用函数plot_Matrix（）用于可视化混淆矩阵
confusion_mat = confusion_matrix(y_test, y_predict)
iris_target = ['Setosa', 'Versicolor', 'Virginica']
plot_matrix(confusion_mat, iris_target)


# 绘制图像2：不同迭代次数下的模型性能
Score1 = []
Loss1 = []
temp = 100
for i in range(10):
    mlp1 = MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', activation='logistic', learning_rate_init=0.01, max_iter=temp+i*100)
    mlp1.fit(x_train, y_train)
    score1 = mlp1.score(x_test, y_test)
    loss1 = mlp1.loss_
    Score1.append(score1)
    Loss1.append(loss1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.title("不同max_iter下的学习曲线")
plt.xlabel("迭代次数")
plt.ylabel("精度")
# x轴坐标刻度变成100-800，数组名+1表示将其各个元素值+1
x_locator1 = np.arange(0, len(Score1)) + 1
x_locator1 = x_locator1 * 100
plt.plot(x_locator1, Score1, label='学习曲线')
plt.xticks(np.arange(100, 1001, 100))
# plt.legend(loc='upper left')
plt.grid(b=True, linestyle='--')
plt.savefig('MLP_iris_max_iter.svg')
plt.show()

# 绘制图像3：不同隐层神经元个数下的模型性能
Score2 = []
for j in range(10):
    mlp2 = MLPClassifier(hidden_layer_sizes=(10 + j * 10,), solver='sgd', activation='logistic', max_iter=200, learning_rate_init=0.01)
    mlp2.fit(x_train, y_train)
    score2 = mlp2.score(x_test, y_test)
    Score2.append(score2)
# plt.title("不同hidden_layer_sizes下的学习曲线")
plt.xlabel("单隐层神经元个数")
plt.ylabel("精度")
# x轴坐标刻度变成10-100，数组名+1表示将其各个元素值+1
x_locator2 = np.arange(0, len(Score2)) + 1
x_locator2 = x_locator2 * 10
plt.plot(x_locator2, Score2, label='学习曲线')
plt.xticks(np.arange(10, 101, 10))
# plt.legend(loc='upper left')
plt.grid(b=True, linestyle='--')
plt.savefig('MLP_iris_hidden_layer_sizes.svg')
plt.show()
