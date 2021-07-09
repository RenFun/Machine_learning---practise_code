# Author: RenFun
# File: decision_tree02.py
# Time: 2021/06/21


# 记载创建好的西瓜数据集，计算信息增益，增益率，基尼指数，为创建决策树做准备
import numpy as np

watermelon = np.genfromtxt('watermelon.txt', delimiter=' ', dtype=str)  # 导入数据时若出现nan，即将dtype设为str，其默认值为float
attribute = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
print(attribute)
print(watermelon)
l = len(watermelon)


# 计算信息熵函数
def information_entopy(P1, P2):
    if P1 == 0:
        return - (P2 * np.log2(P2))
    if P2 == 0:
        return - (P1 * np.log2(P1))
    return - (P1 * np.log2(P1) + P2 * np.log2(P2))


# 信息增益：对每个属性计算信息增益，其值越大，表示当前属性值作为划分属性所得到的结点纯度越高
# 西瓜数据集中有6个离散的属性：色泽， 根蒂， 敲声， 纹理， 脐部， 触感
# 通过遍历属性列，返回属性值数量，以及对应的正反例个数
def find(att):                          # 以列表为参数（属性列为一个列表）
    C = []                              # 每个属性对应的样本数量，形式为1*n
    TN = []                             # 每个属性对应的正例的数量
    FN = []                             # 每个属性对应的负例的数量
    list1 = att
    attlist = set(list1)
    attlist = list(attlist)             # 数组无法进行遍历，这里需要把attlist这个无需不重复数组变成列表
    # print(type(attlist))
    attcount = len(attlist)
    print(attlist, attcount)            # 每个属性的取值量，例如色泽这个属性有三个取值：青绿，浅白，乌黑，attcount = 3
    for j in range(attcount):
        count = 0
        turenum = 0
        falsenum = 0
        for k in range(len(list1)):
            if list1[k] == attlist[j]:
                count = count + 1
                if watermelon[k, 8] == '好瓜':
                    turenum = turenum + 1
                else:
                    falsenum = falsenum + 1

        C = np.append(C, count)
        TN = np.append(TN, turenum)
        FN = np.append(FN, falsenum)
    return C, TN, FN


# 根节点的信息熵
count1 = 0
count2 = 0
for i in range(len(watermelon)):
    if watermelon[i, 8] == '好瓜':
        count1 = count1 + 1
p1 = count1/len(watermelon)
print('第一类样本所占比例：', p1)
for i in range(len(watermelon)):
    if watermelon[i, 8] == '坏瓜':
        count2 = count2 + 1
p2 = count2/len(watermelon)
print('第二类样本所占比例：', p2)
Ent = - (p1 * np.log2(p1) + p2 * np.log2(p2))                  # 以2为底的log函数
print('根结点信息熵：', format(Ent, '.3f'))                    # 信息熵越小，当前样本集合的纯度越高


# 计算离散属性值的信息增益
COUNT, TURENUM, FALSEUNM = find(watermelon[:, 0])              # 带入不同的参数，得到不同的结果
print(COUNT, TURENUM, FALSEUNM)
ent = []
num = len(COUNT)
for n in range(num):
    e = information_entopy(TURENUM[n]/COUNT[n], FALSEUNM[n]/COUNT[n])
    ent = np.append(ent, round(e, 3))
print('分支节点的信息熵：', ent)
sum1 = 0
for p in range(len(ent)):
    sum1 = sum1 + COUNT[p]/l * ent[p]
Gain2 = Ent - sum1
print('信息增益：', format(Gain2, '.3f'))


# 计算各个属性的固有值
sum2 = 0
for q in range(num):
    sum2 = sum2 + COUNT[q]/l * np.log2(COUNT[q]/l)
IV = -sum2
print('增益率：', format(IV, '.3f'))


# 计算基尼指数


