# Author: RenFun
# File: NB_20newsgroups.py
# Time: 2021/12/05


# 使用多项式朴素贝叶斯分类器处理离散数据集（20个主题的18000个新闻组的帖子）
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 导入数据集
news = fetch_20newsgroups(subset='all')
news_data = news.data
news_target = news.target
target_name = news.target_names
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(news_data, news_target, train_size=0.7, random_state=2, stratify=news_target)
print(x_test)
# 导出数据集到本地
outputfile_data = "D:/PycharmProjects/Machine_Learning/news_data.txt"
outputfile_target = "D:/PycharmProjects/Machine_Learning/news_target.txt"
df = pd.DataFrame(news_data, index=range(18846), )
pf = pd.DataFrame(news_target, )
# jj = df.join(pf,how='outer')
df.to_json(outputfile_data)
pf.to_json(outputfile_target)
# 特征提取
tf = TfidfVectorizer()
#
x_train = tf.fit_transform(x_train)
x_test = tf.transform(x_test)
print(tf.get_feature_names())
# 实例化多项式朴素贝叶斯分类器
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_predict = mnb.predict(x_test)
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
                        ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xticks(rotation=30)
    plt.savefig('NB_news_confusion_matrix.svg', bbox_inches='tight', )
    plt.show()


target = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
confusion_mat = confusion_matrix(y_test, y_predict)
plot_matrix(confusion_mat, target)


# 超参数：平滑参数alpha
cv_score = []
for i in range(11):
    mnb_alpha = MultinomialNB(alpha=i*0.1)
    mnb_alpha.fit(x_train, y_train)
    score = mnb_alpha.score(x_test, y_test)
    # score = cross_val_score(mnb_alpha, news_data, news_target, cv=10, scoring='accuracy')
    # 每一个score中都有10个数据，需要求均值之后作为模型精度
    cv_score.append(score)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(np.arange(0, 1.1, 0.1), cv_score)
plt.xlabel('平滑参数的取值')
plt.ylabel('精度')
plt.grid(b=True, linestyle='--')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.savefig('NB_news_alpha.svg', bbox_inches='tight')
plt.show()
