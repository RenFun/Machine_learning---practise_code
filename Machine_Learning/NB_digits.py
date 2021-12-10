# Author: RenFun
# File: NB_digits.py
# Time: 2021/12/04


# 使用手写数字数据集进行实验
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 导入数据集
digits = load_digits()
digits_data = digits.data
digits_target = digits.target
feature_name = digits.feature_names
target_name = digits.target_names
# 数据集的可视化
# 设置colormap为gray
plt.gray()
# 展示一个矩阵或者向量在一个新的图像窗口
plt.matshow(digits.images[0])
plt.savefig('NB_digits_image0.svg')
plt.matshow(digits.images[1])
plt.savefig('NB_digits_image1.svg')
plt.show()
# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(digits_data, digits_target, train_size=0.7, random_state=2, stratify=digits_target)
# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(digits_data)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 实例化高斯朴素贝叶斯
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_predict = gnb.predict(x_test)
print(precision_score(y_test, y_predict, average='macro'))
print(recall_score(y_test, y_predict, average='macro'))
print(f1_score(y_test, y_predict, average='macro'))
print(confusion_matrix(y_test, y_predict))
# 混淆矩阵可视化
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
    plt.savefig('NB_digits_confusion_matrix.svg', bbox_inches='tight')
    plt.show()


confusion_mat = confusion_matrix(y_test, y_predict)
plot_matrix(confusion_mat, target_name)
# 不同样本数量的学习率
estimator = GaussianNB()
train_size, train_scores, test_scores = learning_curve(estimator, digits_data, digits_target, cv=10)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("样本数量")
plt.ylabel("精度")
plt.plot(train_size, np.mean(train_scores, axis=1), color="r", label="训练集精度")
plt.plot(train_size, np.mean(test_scores, axis=1), color="b", label="测试集精度")
plt.legend(loc='upper right')
plt.grid(b=True, linestyle='--')
plt.savefig('NB_digits_learning_curve.svg')
plt.show()
