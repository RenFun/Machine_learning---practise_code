# Author: RenFun
# File: SVM_iris.py
# Time: 2021/10/31


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


iris = load_iris()
iris_data = iris.data
iris_target = iris.target
target_name = iris.target_names
feature_name = iris.feature_names
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, train_size=0.7, random_state=2, stratify=iris_target)
svc = SVC(C=1, kernel='rbf', decision_function_shape='ovr')
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
print(precision_score(y_test, y_predict, average='macro'))
print(recall_score(y_test, y_predict, average='macro'))
print(f1_score(y_test, y_predict, average='macro'))
