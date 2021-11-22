# Author: RenFun
# File: KNN_iris.py
# Time: 2021/10/27


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# 使用鸢尾花数据集进行实验
iris = load_iris()
x = iris.data
y = iris.target
target_name = iris.target_names
feature_name = iris.feature_names
# 用留出法对数据集进行分层采样，三个类别在数据集中比例为1：1：1。划分为训练集和测试集后，其类别比例仍是1：1：1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2, stratify=y)
# 对数据进行预处理，将所有数据缩放到[0,1]区间内
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
x = scaler.transform(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 实例化knn，k=5，平均加权，欧式距离
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
# 输出P-R-F值和混淆矩阵
print(precision_score(y_test, y_predict, average='macro'))
print(recall_score(y_test, y_predict, average='macro'))
print(f1_score(y_test, y_predict, average='macro'))
print(accuracy_score(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))


# 可视化混淆矩阵
def plot_matrix(cm, classes, cmap=plt.cm.Blues):
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    ax.imshow(cm, origin='upper', cmap=cmap)
    # 坐标轴设置
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,)
    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    # 标注信息
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if float(cm[i, j] * 100) > 0:
                ax.text(j, i, format(float(cm[i, j]), '.2f'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xticks(rotation=30)
    plt.savefig('KNN_iris_confusion_matrix.svg', bbox_inches='tight')
    plt.show()


confusion_mat = confusion_matrix(y_test, y_predict)
plot_matrix(confusion_mat, target_name)


# 绘制图像：使用交叉验证法在欧式距离下找到最佳的k取值
# k取值范围为[1,30]
k_range = np.arange(1, 31)
# 存放不同k取值下模型的精度
cv_score = []
for i in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=i, p=2, weights='uniform')
    score = cross_val_score(knn_k, x, y, cv=10, scoring='accuracy')
    # 每一个score中都有10个数据，需要求均值之后作为模型精度
    cv_score.append(score.mean())
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(k_range, cv_score)
plt.xlabel('k的取值')
plt.ylabel('精度')
plt.grid(b=True, linestyle='--')
plt.xticks(np.arange(0, 31, 1))
plt.savefig('KNN_iris_euclidean_k.svg', bbox_inches='tight')
plt.show()

# 绘制图像：使用交叉验证法在曼哈顿距离下找到最佳的k取值
# 存放不同k取值下模型的精度
cv_score1 = []
for i in k_range:
    knn_k1 = KNeighborsClassifier(n_neighbors=i, p=1, weights='uniform')
    score = cross_val_score(knn_k1, x, y, cv=10, scoring='accuracy')
    # 每一个score中都有10个数据，需要求均值之后作为模型精度
    cv_score1.append(score.mean())
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(k_range, cv_score1)
plt.xlabel('k的取值')
plt.ylabel('精度')
plt.grid(b=True, linestyle='--')
plt.xticks(np.arange(0, 31, 1))
plt.savefig('KNN_iris_manhattan_k.svg', bbox_inches='tight')
plt.show()

