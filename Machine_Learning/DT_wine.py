# Author: RenFun
# File: DT_wine.py
# Time: 2021/10/13

# 不同数据集（红酒数据集）下的决策树的P，R，F 值


from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


wine = datasets.load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, train_size=0.7, random_state=2)
clf = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(classification_report(y_test, y_predict))
print(precision_score(y_test, y_predict, average='macro'))
print(recall_score(y_test, y_predict, average='macro'))
print(f1_score(y_test, y_predict, average='weighted'))
print(confusion_matrix(y_test, y_predict))


Score = []
for i in range(10):
    clf1 = DecisionTreeClassifier(max_depth=i+1)
    clf1.fit(x_train, y_train)
    score1 = clf1.score(x_test, y_test)
    Score.append(score1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.title("不同max_depth下的学习曲线")
plt.xlabel("决策树的深度")
plt.ylabel("精度")
# x轴坐标刻度变成1-10，数组名+1表示将其各个元素值+1
x_locator = np.arange(0, len(Score)) + 1
plt.plot(x_locator, Score, color='r')
plt.xticks(np.arange(1, 11))
# plt.legend(loc='upper left')
plt.grid(b=True, linestyle='--')
plt.savefig('DT_wine_max_depth.svg')
plt.show()