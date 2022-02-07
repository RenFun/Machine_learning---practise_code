# Author: RenFun
# File: decision_tree03.py
# Time: 2021/06/24


# 计算连续属性值的信息增益
import numpy as np
watermelon = np.genfromtxt('watermelon3.csv', delimiter=',')
print(watermelon)
l = len(watermelon)
a = sorted(watermelon[:, 1])
b = sorted(watermelon[:, 2])                # 列表中的值类型为float
# print(type(a[1]))
ta = []                                     # 候选划分点集合
tb = []
for i in range(l-1):
    x = (a[i] + a[i + 1]) / 2
    y = (b[i] + b[i + 1]) / 2
    ta = np.append(ta, x)
    tb = np.append(tb, y)
Ent = 0.998                                 # 根节点的信息熵


# 计算信息熵函数
def information_entopy(P1, P2):
    if P1 == 0:
        return - (P2 * np.log2(P2))
    if P2 == 0:
        return - (P1 * np.log2(P1))
    return - (P1 * np.log2(P1) + P2 * np.log2(P2))


sum1 = 0
sum2 = 0
gain = 0
Gain = {}
for j in range(l-1):                # l-1个候选划分点，每一个划分点将原数据集分成两份，D1和D2
    temp = ta[j]
    D1 = []
    D2 = []
    turecount1 = 0
    falsecount1 = 0
    turecount2 = 0
    falsecount2 = 0
    for k in range(l):              # 遍历原数据集
        if watermelon[k, 1] <= temp:                # 将原数据集中密度比划分值小的分为D1
            D1 = np.append(D1, k)
        else:                                       # 剩余的化为D2
            D2 = np.append(D2, k)
    # 原数据集二分结束后，计算该划分点的信息增益
    for m in range(len(D1)):
        if watermelon[int(D1[m]), 3] == 1:
            turecount1 = turecount1 + 1
        else:
            falsecount1 = falsecount1 +1
    p1 = turecount1/len(D1)
    p2 = falsecount1/len(D1)
    x1 = information_entopy(p1, p2)
    sum1 = len(D1)/l * x1

    for n in range(len(D2)):
        if watermelon[int(D2[n]), 3] == 1:
            turecount2 = turecount2 + 1
        else:
            falsecount2 = falsecount2 +1
    p3 = turecount2/len(D2)
    p4 = falsecount2/len(D2)
    x2 = information_entopy(p3, p4)
    sum2 = len(D2)/l * x2
    gain = Ent - (sum1 + sum2)
    xx = {temp: gain}
    Gain.update(xx)                 # 添加字典中的元素

print(ta)
for key, value in Gain.items():
    if(value == max(Gain.values())):            # 找到value最大值，以及对应的key。此时value为最佳信息增益，key为最佳划分点
        print(key, value)
