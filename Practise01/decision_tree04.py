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
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import graphviz
import pydotplus
import os
from IPython.core.display import display
from IPython.display import Image
os.environ["PATH"] += os.pathsep + 'D:\graphviz\bin'

# 加载数据集
iris = datasets.load_iris()
# 划分数据集为训练集和数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.7)
# 定义决策树类：分裂节点时评价标准为信息增益；分裂节点时的策略选择最优分裂策略（在特征的所有划分点中找出最优的划分点）
# 关于剪枝的参数：1.max_depth 2.min_samples_split 3.min_samples_leaf 4.max_features
clf = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=10, max_features=3)
# 在训练集上拟合
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
# 评估模型
score = clf.score(x_test, y_test)
print(score)

# 决策树的可视化
# 1.用pydotplus生成iris.pdf
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris3.pdf")

# # 2.生成iris.dot文件，再用graphziv打开
# with open("iris2.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)

# 3.直接把图产生在ipython的notebook
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
# display(Image(graph.create_png()))


# iris1.pdf：划分属性时使用的准则为gini（gini指数越小，当前数据集的纯度越高，故选择gini指数最小的属性进行划分），其余参数皆为默认值，score=0.9556
# iris2.pdf：划分属性时使用的准则为entropy（信息增益越大，则使用该属性进行划分所获得“纯度提升”越大），其余参数皆为默认值，score=0.9777
# iris3.pdf：gini, max_depth=none, min_samples_leaf=5, min_samples_split=10, max_features=3， score=0.9333

# 绘制图像：不同max_depth下的学习曲线
# Score[]存放不同max_depth下的score
Score = []
for i in range(10):
    clf1 = DecisionTreeClassifier(max_depth=i+1)
    clf1.fit(x_train, y_train)
    score1 = clf1.score(x_test, y_test)
    Score.append(score1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("不同max_depth下的学习曲线")
plt.xlabel("max_depth")
plt.ylabel("score")
x_locator = np.arange(0, len(Score)) + 1
plt.plot(x_locator, Score, label='学习曲线')
plt.xticks(np.arange(1, 11))
plt.legend(loc='upper left')
plt.show()



