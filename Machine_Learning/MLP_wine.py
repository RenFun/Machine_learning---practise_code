# Author: RenFun
# File: MLP_wine.py
# Time: 2021/10/17


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


wine = load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=2)
# 不同的参数选择会影响模型的性能：隐层神经元个数hidden_layer_sizes，迭代次数max_iter，学习率初始值learning_rate_init
mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 20), max_iter=800)
mlp.fit(x_train, y_train)
y_predict = mlp.predict(x_test)
wine_target = ['Gin', 'Shirley', 'Belmord']
print(classification_report(y_test, y_predict, target_names=wine_target))
print(confusion_matrix(y_test, y_predict))
print(precision_score(y_test, y_predict, average='macro'))
print(recall_score(y_test, y_predict, average='macro'))
print(f1_score(y_test, y_predict, average='weighted'))


# 绘制图像：增加隐层数量，对比模型精度
Score = []
MLP_ = []
mlp1 = MLPClassifier(hidden_layer_sizes=(100,),  max_iter=800)
mlp2 = MLPClassifier(hidden_layer_sizes=(100, 100,),  max_iter=800)
mlp3 = MLPClassifier(hidden_layer_sizes=(100, 100, 100,),  max_iter=800)
mlp4 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100,),  max_iter=800)
mlp5 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100,),  max_iter=800)
mlp6 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100,),  max_iter=800)
mlp7 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100,),  max_iter=800)
mlp8 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100,),  max_iter=800)
mlp9 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100,),  max_iter=800)
mlp10 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100), max_iter=800)
MLP_.append(mlp1)
MLP_.append(mlp2)
MLP_.append(mlp3)
MLP_.append(mlp4)
MLP_.append(mlp5)
MLP_.append(mlp6)
MLP_.append(mlp7)
MLP_.append(mlp8)
MLP_.append(mlp9)
MLP_.append(mlp10)
for i in MLP_:
    i.fit(x_train, y_train)
    s = i.score(x_test, y_test)
    Score.append(s)
print(Score)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("隐层个数")
plt.ylabel("精度")
x_locator = np.arange(0, len(Score)) + 1
plt.plot(x_locator, Score)
plt.xticks(np.arange(1, 11))
plt.grid(b=True, linestyle='--')
plt.savefig('MLP_wine_hidden_layer_number.svg')
plt.show()
