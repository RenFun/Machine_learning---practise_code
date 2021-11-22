# Author: RenFun
# File: SVM_image_segmentation.py
# Time: 2021/10/26


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# 数据集采用图像分割数据集（Image Segmentation），每个样本有19个属性，共7个类别，训练集样本数量为210，测试集样本数为2100。
# 加载本地数据集：np.loadtxt()
Data_train_target = np.loadtxt('segmentation.data', dtype='str', delimiter=',', skiprows=2, usecols=(0, ))
Data_train_data = np.loadtxt('segmentation.data', dtype='float', delimiter=',', skiprows=2, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
Data_test_target = np.loadtxt('segmentation.test', dtype='str', delimiter=',', skiprows=5, usecols=(0,))
Data_test_data = np.loadtxt('segmentation.test', dtype='float', delimiter=',', skiprows=5, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
scaler = StandardScaler()
# 标准化训练集
scaler.fit(Data_train_data)
Data_train_data = scaler.transform(Data_train_data)
# 标准化测试集
Data_test_data = scaler.transform(Data_test_data)
# 实例化多分类SVM，注意参数选择
svc = SVC(C=1, kernel='rbf', decision_function_shape='ovr')
svc.fit(Data_train_data, Data_train_target)
Data_test_target_pre = svc.predict(Data_test_data)
# 模型的性能指标
print(precision_score(Data_test_target, Data_test_target_pre, average='macro'))
print(recall_score(Data_test_target, Data_test_target_pre, average='macro'))
print(f1_score(Data_test_target, Data_test_target_pre, average='macro'))
print(classification_report(Data_test_target, Data_test_target_pre))
print(svc.score(Data_test_data, Data_test_target))
print(confusion_matrix(Data_test_target, Data_test_target_pre))


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
    plt.savefig('SVM_image_segmentation_confusion_matrix.svg', bbox_inches='tight')
    plt.show()


confusion_mat = confusion_matrix(Data_test_target, Data_test_target_pre)
segmentation_target = ['brickface', 'sky', 'foliage', 'cement', 'window', 'path', 'grass']
plot_matrix(confusion_mat, segmentation_target)


# 参数C的取值对模型的影响，取值范围[0.01, 1000]
# 将四个子图放在一张画布上，第一个位置显示C的取值范围[0.01,0.1]
Score4 = []
for i in range(1, 11):
    svc_C = SVC(C=i * 0.01, kernel='rbf', decision_function_shape='ovr')
    svc_C.fit(Data_train_data, Data_train_target)
    score4 = svc_C.score(Data_test_data, Data_test_target)
    Score4.append(score4)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x_locator4 = np.arange(1, len(Score4) + 1) * 0.01
Score3 = []
for i in range(1, 11):
    svc_C = SVC(C=i * 0.1, kernel='rbf', decision_function_shape='ovr')
    svc_C.fit(Data_train_data, Data_train_target)
    score3 = svc_C.score(Data_test_data, Data_test_target)
    Score3.append(score3)
x_locator3 = np.arange(1, len(Score3) + 1) * 0.1
Score2 = []
for i in range(1, 11):
    svc_C = SVC(C=i, kernel='rbf', decision_function_shape='ovr')
    svc_C.fit(Data_train_data, Data_train_target)
    score2 = svc_C.score(Data_test_data, Data_test_target)
    Score2.append(score2)
x_locator2 = np.arange(1, len(Score2) + 1)
Score1 = []
for i in range(1, 11):
    svc_C = SVC(C=i*10, kernel='rbf', decision_function_shape='ovr')
    svc_C.fit(Data_train_data, Data_train_target)
    score1 = svc_C.score(Data_test_data, Data_test_target)
    Score1.append(score1)
x_locator1 = np.arange(1, len(Score1) + 1) * 10
Score5 = []
for i in range(1, 11):
    svc_C = SVC(C=i*100, kernel='rbf', decision_function_shape='ovr')
    svc_C.fit(Data_train_data, Data_train_target)
    score5 = svc_C.score(Data_test_data, Data_test_target)
    Score5.append(score5)
x_locator5 = np.arange(1, len(Score5) + 1) * 100
# 合并精度列表
Score4.extend(Score3)
Score4.extend(Score2)
Score4.extend(Score1)
Score4.extend(Score5)
plt.xlabel("规范化的惩罚参数")
plt.ylabel("精度")
x_locator = np.arange(1, 51)
plt.plot(x_locator, Score4)
plt.grid(b=True, linestyle='--')
# a中存放真实的惩罚参数
a = []
a.extend(x_locator4)
a.extend(x_locator3)
a.extend(x_locator2)
a.extend(x_locator1)
a.extend(x_locator5)
plt.savefig('SVM_image_segmentation_C.svg')
plt.show()
# 利用grid search寻找C的最佳参数
parameters = {'C': np.arange(0.01, 1000)}
grid = GridSearchCV(svc, parameters, scoring='accuracy')
grid.fit(Data_train_data, Data_train_target)
print(grid.best_params_)
