# Author: RenFun
# File: Kmeans_iris1.py
# Time: 2021/12/17


# 使用鸢尾花数据集进行Kmeans
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


# 导入数据
iris = load_iris()
iris_data = iris.data
iris_target = iris.target
# 实例化Kmeans，n_clusters表示簇的数量（可以比较不同值下模型的性能）；max_iter表示迭代次数
kmeans = KMeans(n_clusters=3, max_iter=300, init='k-means++', random_state=3)
# 训练数据
kmeans.fit(iris_data)
# 聚类后的标签
predict_label = kmeans.labels_
# 聚类后得到各个簇类的质心，即均值
centers = kmeans.cluster_centers_
# 外部性能度量：JC，FMI和RI
# 内部性能度量：DBI和DI
jc = jaccard_score(iris_target, predict_label, average='weighted')
fmi = fowlkes_mallows_score(iris_target, predict_label)
ari = adjusted_rand_score(iris_target, predict_label)
dbi = davies_bouldin_score(iris_data, predict_label)
silhouette = silhouette_score(iris_data, predict_label)
ami = adjusted_mutual_info_score(iris_target, predict_label)
vmeas = v_measure_score(iris_target, predict_label)
# di =
print('JC(Kmeans)：', jc)
print('FMI（Kmeans）：', fmi)
print('ARI(Kmeans)：', ari)
print('##################################')
print('DBI(Kmeans)：', dbi)
# print('DI(Kmeans)：', di)
print('##################################')
print('轮廓系数：', silhouette)
print('调整互信息：', ami)
print('V-measure：', vmeas)


# 绘制图像1：训练集样本的可视化，选取两个属性
iris_feature1 = iris_data[:, 0]
iris_feature2 = iris_data[:, 1]
iris_feature3 = iris_data[:, 2]
iris_feature4 = iris_data[:, 3]
plt.rcParams['font.sans-serif'] = ['SimHei']          # 用来正常显示中文，设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
plt.xlabel("花萼长度")
plt.ylabel("花萼宽度")
# 选取前两个属性作为x，y轴，对样本进行可视化
plt.scatter(iris_feature1, iris_feature2, c='royalblue')
plt.savefig('Kmeans_iris_scatter.svg')
plt.show()


# 绘制图像2：训练样本的可视化，即高维数据的可视化
tsne = TSNE(n_components=2).fit(iris_data)
# 将原始数据embedding_转化为DataFrame形式，DataFrame相当于二维数组
df = pd.DataFrame(tsne.embedding_)
# 将聚类结果存储到df数据表中
df['labels'] = kmeans.labels_
# 提取不同标签的数据
df1 = df[df['labels'] == 0]
df2 = df[df['labels'] == 1]
df3 = df[df['labels'] == 2]
plt.scatter(df1[0], df1[1], c='lightgreen', label='类别1', marker='s')
plt.scatter(df2[0], df2[1], c='red', label='类别2', marker='o')
plt.scatter(df3[0], df3[1], c='royalblue', label='类别3', marker='v')
plt.legend(loc='upper left')
plt.show()

plt.scatter(df[0], df[1], c='royalblue')
plt.savefig('Kmeans_iris_tsne.svg')
plt.show()

# 绘制图像3：聚类结果的可视化，选取两个属性作为x，y轴坐标
x0 = iris_data[predict_label == 0]
x1 = iris_data[predict_label == 1]
x2 = iris_data[predict_label == 2]
plt.scatter(x0[:, 0], x0[:, 1], c='lightgreen', label='类别1', marker='s')
plt.scatter(x1[:, 0], x1[:, 1], c='red', label='类别2', marker='o')
plt.scatter(x2[:, 0], x2[:, 1], c='orange', label='类别3', marker='v')
plt.scatter(centers[:, 0], centers[:, 1], c='black', label='质心', marker='x', s=50)
plt.legend(loc='upper left')
plt.xlabel("花萼长度")
plt.ylabel("花萼宽度")
plt.savefig('Kmeans_iris_result.svg')
plt.show()
