# Author: RenFun
# File: DT_iris.py
# Time: 2021/10/08


# 利用sklearn库实现决策树
# 安装graphviz用于决策树的可视化：需要安装grapgviz和python-graphviz 这两个包，安装完后需要重启pycharm
# 安装pydotplus包：提供了一个完整的界面，用于在图表语言中的计算机处理和过程图表
# 数据集使用sklearn库自带的鸾尾花数据集，可以和逻辑回归模型比较性能

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pydotplus


# 加载数据集
iris = datasets.load_iris()
# 划分数据集为训练集和数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.7, random_state=2)
# 定义决策树类：分裂节点时评价标准为基尼指数（CART算法），信息增益（ID3算法）,增益率（C4.5算法）；分裂节点时的策略选择最优分裂策略（在特征的所有划分点中找出最优的划分点）
# 关于剪枝的参数：1.max_depth 2.min_samples_split 3.min_samples_leaf 4.max_features
clf = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None)
# 在训练集上拟合
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
iris_target = ['Setosa', 'Versicolor', 'Virginica']
# 评估模型：P（查准率precision） R（召回率recall） F（F1指数） 混淆矩阵
# 微平均值：micro average，所有数据结果的平均值
# 宏平均值：macro average，所有标签结果的平均值
# 加权平均值：weighted average，所有标签结果的加权平均值
print(classification_report(y_test, y_predict, target_names=iris_target))
print(precision_score(y_test, y_predict, average='macro'))
print(recall_score(y_test, y_predict, average='macro'))
print(f1_score(y_test, y_predict, average='weighted'))
print(confusion_matrix(y_test, y_predict))


# 绘制图像：混淆矩阵
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
    plt.savefig('DT_iris_confusion_matrix.svg', bbox_inches='tight')
    plt.show()


confusion_mat = confusion_matrix(y_test, y_predict)
iris_target = ['Setosa', 'Versicolor', 'Virginica']
plot_matrix(confusion_mat, iris_target)


# 决策树的可视化
# 1.用pydotplus生成iris.pdf
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("DT_iris_tree.pdf")

# 2.生成iris.dot文件，再用graphziv打开
# with open("iris2.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)


# iris1.pdf：划分属性时使用的准则为gini（gini指数越小，当前数据集的纯度越高，故选择gini指数最小的属性进行划分），其余参数皆为默认值，score=0.9556
# iris2.pdf：划分属性时使用的准则为entropy（信息增益越大，则使用该属性进行划分所获得“纯度提升”越大），其余参数皆为默认值，score=0.9777
# iris3.pdf：gini, max_depth=none, min_samples_leaf=5, min_samples_split=10, max_features=3， score=0.9333


# 绘制图像：不同深度下的学习曲线
Score1 = []
for i in range(10):
    clf1 = DecisionTreeClassifier(max_depth=i+1)
    clf1.fit(x_train, y_train)
    score1 = clf1.score(x_test, y_test)
    Score1.append(score1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.title("不同max_depth下的学习曲线")
plt.xlabel("决策树的深度")
plt.ylabel("精度")
# x轴坐标刻度变成1-10，数组名+1表示将其各个元素值+1
x_locator1 = np.arange(0, len(Score1)) + 1
plt.plot(x_locator1, Score1)
plt.xticks(np.arange(1, 11))
# plt.legend(loc='upper left')
plt.grid(b=True, linestyle='--')
plt.savefig('DT_iris_max_depth.svg')
plt.show()


# 绘制图像：分类所需的最小数量的的节点数，当结点的样本数量小于该参数时，则不再产生分支，该分支的标签分类以该分支下标签最多的类别为准
Score2 = []
for i in range(1, 10):
    clf2 = DecisionTreeClassifier(min_samples_split=i+1)
    clf2.fit(x_train, y_train)
    score2 = clf2.score(x_test, y_test)
    Score2.append(score2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("结点分裂所需要的最小样本数量")
plt.ylabel("精度")
# x轴坐标刻度变成1-10，数组名+1表示将其各个元素值+1
x_locator2 = np.arange(1, len(Score2) + 1) + 1
plt.plot(x_locator2, Score2)
plt.xticks(np.arange(2, 11))
plt.grid(b=True, linestyle='--')
plt.savefig('DT_iris_min_samples_split.svg')
plt.show()

