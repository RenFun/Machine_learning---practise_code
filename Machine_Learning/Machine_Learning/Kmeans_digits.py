# Author: RenFun
# File: Kmeans_digits.py
# Time: 2021/12/18


# tsne高维表示
# 超参数：簇的数目，
# 不同聚类方法的比较
# 性能度量：FMI，DBI...
# 查看kmengs的属性
# 是否要进行数据预处理（PCA）
# 导入数据集

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 加载数据集
digits = load_digits()
digits_data = digits.data
digits_target = digits.target
feature_name = digits.feature_names
target_name = digits.target_names

# 实例化Kmeans，n_clusters表示簇的数量（可以比较不同值下模型的性能）；max_iter表示迭代次数
kmeans = KMeans(n_clusters=10, max_iter=800, init='k-means++', random_state=3)
# 训练数据
kmeans.fit(digits_data)
# 聚类后的标签
predict_label = kmeans.labels_
# 聚类后得到各个簇类的质心，即均值
centers = kmeans.cluster_centers_
# 外部性能度量：JC，FMI和RI
# 内部性能度量：DBI和DI
jc = jaccard_score(digits_target, predict_label, average='weighted')
fmi = fowlkes_mallows_score(digits_target, predict_label)
ari = adjusted_rand_score(digits_target, predict_label)
dbi = davies_bouldin_score(digits_data, predict_label)
silhouette = silhouette_score(digits_data, predict_label)
ami = adjusted_mutual_info_score(digits_target, predict_label)
vmeas = v_measure_score(digits_target, predict_label)
print('JC(Kmeans)：', jc)
print('FMI（Kmeans）：', fmi)
print('ARI(Kmeans)：', ari)
print('##################################')
print('DBI(Kmeans)：', dbi)
print('##################################')
print('轮廓系数：', silhouette)
print('调整互信息：', ami)
print('V-measure：', vmeas)


# 绘制图像1：使用TSNE使得高维数据可视化
plt.rcParams['font.sans-serif'] = ['SimHei']          # 用来正常显示中文，设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
tsne = TSNE(n_components=2, perplexity=50,).fit(digits_data)
# 将原始数据embedding_转化为DataFrame形式，DataFrame相当于二维数组
df = pd.DataFrame(tsne.embedding_)
# 将聚类结果存储到df数据表中
df['labels'] = kmeans.labels_
plt.scatter(df[0], df[1], c='royalblue')
plt.savefig('Kmeans_digits_tsne.svg')
plt.show()

# 绘制图像2：聚类结果可视化
df1 = df[df['labels'] == 0]
df2 = df[df['labels'] == 1]
df3 = df[df['labels'] == 2]
df4 = df[df['labels'] == 3]
df5 = df[df['labels'] == 4]
df6 = df[df['labels'] == 5]
df7 = df[df['labels'] == 6]
df8 = df[df['labels'] == 7]
df9 = df[df['labels'] == 8]
df10 = df[df['labels'] == 9]
plt.scatter(df1[0], df1[1], c='lightgreen', label='类别1')
plt.scatter(df2[0], df2[1], c='red', label='类别2')
plt.scatter(df3[0], df3[1], c='royalblue', label='类别3')
plt.scatter(df4[0], df4[1], c='black', label='类别4')
plt.scatter(df5[0], df5[1], c='brown', label='类别5')
plt.scatter(df6[0], df6[1], c='gold', label='类别6')
plt.scatter(df7[0], df7[1], c='violet', label='类别7')
plt.scatter(df8[0], df8[1], c='cadetblue', label='类别8')
plt.scatter(df9[0], df9[1], c='aquamarine', label='类别9')
plt.scatter(df10[0], df10[1], c='lightgray', label='类别10')
plt.legend(loc='upper left')
plt.savefig('Kmeans_digits_result.svg')
plt.show()