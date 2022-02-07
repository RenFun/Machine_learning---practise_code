# Author: RenFun
# File: Kmeans_iris2.py
# Time: 2021/12/23

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 使用鸢尾花数据集进行参数对比试验
iris = load_iris()
iris_data = iris.data
iris_target = iris.target


# 绘制图像1：选择不同k值，比较模型性能
# 设置一个列表，存放多个模型
estimator_k = []
FMI = []
for i in range(1, 11):
    # 设置不同的簇数量
    kmeans_k = KMeans(n_clusters=i, random_state=3)
    estimator_k.append(kmeans_k)
    kmeans_k.fit(iris_data)
    pred_label = kmeans_k.labels_
    fmi = fowlkes_mallows_score(iris_target, pred_label)
    FMI.append(fmi)
plt.rcParams['font.sans-serif'] = ['SimHei']          # 用来正常显示中文，设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
plt.plot(np.arange(1, 11), FMI)
plt.xlabel('簇的数量')
plt.ylabel('FMI')
plt.xticks(np.arange(1, 11))
plt.grid(b=True, linestyle='--')
plt.savefig('Kmeans_iris_k.svg', bbox_inches='tight')
plt.show()


# 选择不同的初始化均值向量的方法，比较模型的性能
kmean_init1 = KMeans(n_clusters=3, random_state=3, init='random')
kmean_init1.fit(iris_data)
pred_label_init1 = kmean_init1.labels_
jc_init1 = jaccard_score(iris_target, pred_label_init1, average='weighted')
fmi_init1 = fowlkes_mallows_score(iris_target, pred_label_init1)
ari_init1 = adjusted_rand_score(iris_target, pred_label_init1)
dbi_init1 = davies_bouldin_score(iris_data, pred_label_init1)
silhouette_init1 = silhouette_score(iris_data, pred_label_init1)
ami_init1 = adjusted_mutual_info_score(iris_target, pred_label_init1)
vmeas_init1 = v_measure_score(iris_target, pred_label_init1)
#
kmean_init2 = KMeans(n_clusters=3, random_state=3, init='k-means++')
kmean_init2.fit(iris_data)
pred_label_init2 = kmean_init2.labels_
jc_init2 = jaccard_score(iris_target, pred_label_init2, average='weighted')
fmi_init2 = fowlkes_mallows_score(iris_target, pred_label_init2)
ari_init2 = adjusted_rand_score(iris_target, pred_label_init2)
dbi_init2 = davies_bouldin_score(iris_data, pred_label_init2)
silhouette_init2 = silhouette_score(iris_data, pred_label_init2)
ami_init2 = adjusted_mutual_info_score(iris_target, pred_label_init2)
vmeas_init2 = v_measure_score(iris_target, pred_label_init2)
print('随机选取初始均值向量')
print('JC：', jc_init1)
print('FMI：', fmi_init1)
print('ARI：', ari_init1)
print('DBI：', dbi_init1)
print('Silhouette：', silhouette_init1)
print('AMI：', ami_init1)
print('V-meas：', vmeas_init1)
print('使用优化过的k-means++')
print('JC：', jc_init2)
print('FMI：', fmi_init2)
print('ARI：', ari_init2)
print('DBI：', dbi_init2)
print('Silhouette：', silhouette_init2)
print('AMI：', ami_init2)
print('V-meas：', vmeas_init2)
