# Author: RenFun
# File: KNN_wine.py
# Time: 2021/11/14


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# 使用红酒数据集进行实验
wine = load_wine()
x = wine.data
y = wine.target
target_name = wine.target_names
feature_name = wine.feature_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2, stratify=y)
# 对数据进行预处理，将所有数据缩放到[0,1]区间内
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x)
x = scaler.transform(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
# 留出法
print(precision_score(y_test, y_predict, average='macro'))
print(recall_score(y_test, y_predict, average='macro'))
print(f1_score(y_test, y_predict, average='macro'))
print(confusion_matrix(y_test, y_predict))
print('--------------------')
# 10折交叉验证法
# score_p = cross_val_score(knn, x, y, cv=10, scoring='precision')
# print(score_p.mean())
# score_r = cross_val_score(knn, x, y, cv=10, scoring='recall')
# print(score_r.mean())
# score_f = cross_val_score(knn, x, y, cv=10, scoring='f1')
# print(score_f.mean())
# score_acc = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
# print(score_acc.mean())


