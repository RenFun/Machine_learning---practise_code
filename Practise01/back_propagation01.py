# Author: RenFun
# File: back_propagation01.py
# Time: 2021/06/25


# 标准BP算法，规定迭代次数
import numpy as np
import matplotlib.pyplot as plt


# make_classification()函数返回两个参数X和Y
from sklearn.datasets import make_classification
X, Y = make_classification(n_samples=10, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
# 主要参数如例，其余参数均为默认值。这里要注意n_redundant的默认值为2，若不明确写出，n_features的值必须大于2
# print(X.shape, Y.shape)     # X形式为（样本数量，每个样本的特征数）  Y形式为（样本数量，）
# print(X)                    # X的内容为（特征1， 特征2， ... ， 特征n_features）
# print(Y)                    # Y内容为：每个样本的类别（0或1）
# 作图:以返回的X为依据
plt.scatter(X[:, 0], X[:, 1], c=Y)          # scatter(x,Y,c),其中x和Y为点的位置; c为颜色,c=Y意味着用两种颜色表示两个类
plt.show()


# 激活函数sigmoid（）
def sigmoid(x):
    y = 1.0/(np.exp(-x)+1.0)
    return y


# bp函数
def bp(x, y):              # x = (x1,x2,...,xd),一共d个输入神经元，y是真实值
    x = x.T
    y = y.T
    # 根据输入x确定隐层神经元的阈值value1，输出层神经元的阈值value2，
    # 输入层到隐层的权值v和隐层到输出层的权值w的维数，同时进行初始化
    (n, m) = x.shape
    q = n + 1              # 隐层神经元个数 = 输入层神经元个数 + 1
    l = 1                  # 输出层神经元个数 = 1
    # 这里隐层神经元个数和输出神经元个数的选择根据实际情况可以调整
    # 对阈值和权值进行随机初始化
    value1 = np.random.rand(q)          # q个隐层神经元，value1为1*q的矩阵
    value2 = np.random.rand(l)          # l个输出神经元，value2为1*l的矩阵
    v = np.random.rand(n, q)            # n个输入神经元，q个隐层神经元，v为n*q的矩阵
    w = np.random.rand(q, l)            # q个隐层神经元，l个输出神经元，w为q*l的矩阵
    # 初始化隐层神经元输出，输出层神经元输出,隐层神经元梯度，输出层神经元梯度
    b = []
    predict = []
    learningrate = 0.01             # 学习率
    num = 100                       # 迭代次数
    EE = []
    error_w = np.zeros((q, l))
    error_value2 = np.zeros(l)
    error_v = np.zeros((n, q))
    error_value1 = np.zeros(q)
    # 向前传播 + 反向传播误差
    while (num > 0):                            # 迭代结束条件是迭代num次，也可以有其他结束条件，例如训练误差达到一个很小的值
        for i in range(10):                     # n个样本，每进行一个样本就更新一次参数，这是标准BP算法
            eh = []                             # 每次循环前都要初始化eh和gj
            gj = []
            out = []                            # 用于存储每个样本的输出值
            alpha = np.dot(x[:, i].T, v)        # alpha为q个隐层神经元的输入，1*q
            b = alpha - value1                  # b为q个隐层神经元的输出，1*q
            # print(b.shape, alpha.shape, v.shape)
            beta = np.dot(b, w)                 # beta为l个输出层神经元的输入，1*l
            out.append(sigmoid(beta[0] - value2[0]))
            predict = sigmoid(beta-value2)      # predict为预测的输出值
            for j in range(l):
                temp1 = predict[j] * (1 - predict[j]) * (y[j] - predict[j])
                gj.append(temp1)                # append是列表的方法，数组不适用
            gj = np.array(gj)
            for k in range(q):
                # print(w[k][0], gj)
                temp2 = b[k] * (1 - b[k]) * w[k][0] * gj[0]
                eh.append(temp2)
            eh = np.array(eh)
            # print(gj.shape, b.shape)
            error_w = learningrate * gj * b                 # 1*1的矩阵乘上1*3的矩阵，在这种情况下点乘和np.dot结果一致
            error_value2 = - (learningrate * gj)
            # print(x[:, i].shape, eh.shape)
            # 两个都是一维数组，无法使用np.dot，此时利用np.reshape或者np.matrix进行转化
            error_v = learningrate * x[:, i].reshape((2, 1)) * eh.reshape((1, 3))
            error_value1 = - (learningrate * eh)
            w = w + error_w.reshape(3, 1)
            v = v + error_v
            value1 = value1 + error_value1
            value2 = value2 + error_value2
        out = np.array(out)
        Ek = []
        for t in range(len(out)):
            Ek.append((out[t] - y[t])**2)
        sum = 0
        lenth = len(Ek)
        for s in Ek:
            sum = sum + s
        E = sum/lenth               # E为训练集上的累计误差
        print(E)
        num = num - 1

        EE.append(E)
    print('输出结果：', out)
    print('参数v：', v)
    print('参数value1：', value1)
    print('参数w：', w)
    print('参数value2：', value2)
    print('累计误差E：', E)
    xx = np.arange(100)                 # x轴默认从0到num，步长为1
    yy = np.array(EE)                   # y轴为每次的E，共num个
    plt.scatter(xx, yy)
    plt.xlabel('numbei of iteractions')
    plt.ylabel('error')
    plt.show()
    return num


print(bp(X, Y))






