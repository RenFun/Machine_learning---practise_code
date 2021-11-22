# Author: RenFun
# File: NB_spambase.py
# Time: 2021/11/17


import re
import random
import numpy as np


# 将切分的样本词条整理成不重复的词条列表，即词汇表
def createVocabList(dataset):
    # 初始化词汇表
    vocab_set = set([])
    # 在样本数据集中遍历
    for document in dataset:
        # 取并集
        vocab_set = vocab_set | set(document)
    # 返回列表形式的词汇表
    return list(vocab_set)


# 根据词汇表将 切分的词条列表 向量化，向量的每个元素为1或0，并返回
def setofWords2Vec(vocab_set, inputset):
    # 初始化向量
    return_vec = [0] * len(vocab_set)
    for word in inputset:
        if word in vocab_set:
            # 如果词条列表中的元素存在于词汇表中，则置为1
            return_vec[vocab_set.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    # 返回向量文档
    return return_vec


# 根据词汇表，构建词袋模型
def bagofWoeds2VecMN(vocab_set, inputset):
    return_vec = [0] * len(vocab_set)
    for word in inputset:
        if word in vocab_set:
            # 如果词条列表中的元素存在于词汇表中，则置为1
            return_vec[vocab_set.index(word)] += 1
    # 返回向量文档
    return return_vec


# 训练朴素贝叶斯分类器