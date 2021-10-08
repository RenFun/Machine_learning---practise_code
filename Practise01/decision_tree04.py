# Author: RenFun
# File: decision_tree04.py
# Time: 2021/10/08


# 利用sklearn库实现决策树
# 安装graphviz用于决策树的可视化：需要安装grapgviz和python-graphviz 这两个包，安装完后需要重启pycharm
# 安装pydotplus包：提供了一个完整的界面，用于在图表语言中的计算机处理和过程图表
# 数据集使用sklearn库自带的鸾尾花数据集，可以和逻辑回归模型比较性能

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import pydotplus
from IPython.core.display import display
from IPython.display import Image


# 加载数据集
iris = datasets.load_iris()
# 划分数据集为训练集和数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.7)
# 定义决策树类：分裂节点时评价标准为信息增益；分裂节点时的策略选择最优分裂策略（在特征的所有划分点中找出最优的划分点）
clf = DecisionTreeClassifier(criterion="entropy", splitter="best")
# 在训练集上拟合
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
# 评估模型
score = clf.score(x_test, y_test)
print(score)

# 决策树的可视化
# 这样可以直接把图产生在ipython的notebook
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)

# 用pydotplus生成iris.pdf
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")

# 直接把图产生在ipython的notebook
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
# display(Image(graph.create_png()))





