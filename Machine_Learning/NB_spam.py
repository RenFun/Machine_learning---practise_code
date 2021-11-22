# Author: RenFun
# File: NB_spam.py
# Time: 2021/11/18


# 使用朴素贝叶斯识别中文垃圾邮件
from re import sub
import os
from os import listdir
from collections import Counter
from itertools import chain
from numpy import array
from jieba import cut
from sklearn.naive_bayes import MultinomialNB

allwords = []
def get_words_from_file(txtfile):
    words = []
    with open(txtfile, encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            line = sub(r'[.【】0-9、一。，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word : len(word) > 1, line)
            words.append(line)
    return words

def get_top_words(N):
    txtfiles = [str(i) + '.txt' for i in range(21)]
    for txtfile in txtfiles:
        allwords.append(get_words_from_file(txtfile))
    freq = Counter(chain(*allwords))
    return [w[0] for w in freq.most_common(N)]

top_words = get_top_words(600)
# 获取特征向量
vector = []
for words in allwords:
    temp = list(map(lambda x: words.count(x), top_words))
    vector.append(vector)
vector = array(vector)
labels = array([1]*127 + [0]*24)
# 实例化多项式朴素贝叶斯
MNB = MultinomialNB()
MNB.fit(vector, labels)
def predict(txtfile):
    words = get_words_from_file()
    current_vector = array(tuple(map(lambda x: words.count(x), top_words)))
    result = MNB.predict(current_vector.resharp(1, -1))
    return '垃圾邮件' if result==1 else '正常邮件'


print(predict('51.txt'))


