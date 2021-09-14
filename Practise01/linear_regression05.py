# Author: RenFun
# File: linear_regression05.py
# Time: 2021/09/14


# 调用sklearn库中的函数
# 使用UCI官方数据集或者sklearn库中的数据集（糖尿病病人）


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# 加载糖尿病数据集：总共有442个样本，每个样本有10个属性描述
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)                # 返回（data，target）形式数据
print(diabetes_X)
# 将442个样本按属性分组，变成442*1*10，即442个样本按照一个属性分成一组，一共10组，使用10个属性中的第3个属性（编号为2）
diabetes_X = diabetes_X[:, np.newaxis, 2]
print(diabetes_X)
# 利用列表的切片操作将数据集划分为训练集和数据集：编号0-421为训练集，422-441为测试集
diabetes_X_train = diabetes_X[:-20]                     # 从列表起始到-20切片，从左至右为正方向，编号为0到441
diabetes_X_test = diabetes_X[-20:]                      # 从-20到列表末尾切片，从右至左为负方向，编号为-1到-442
diabetes_y_train = diabetes_y[:-20]                     # 切片操作是左闭右开，右值在正方向时-1，负方向时+1，可确定为后一个值
diabetes_y_test = diabetes_y[-20:]

lr = linear_model.LinearRegression()
lr.fit(diabetes_X_train, diabetes_y_train)              # 使用训练集去拟合线性模型
diabetes_y_pred = lr.predict(diabetes_X_test)           # diabets_y_pred 是预测值
print('系数: ', lr.coef_)
print('截距: ', lr.intercept_)

# mean_squared_error()表示测试集的实际值与预测值的均方方差，即损失函数，越接近0说明模型越好
print('均方误差: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
# plt.xticks(())                    # Passing an empty list removes all xticks 传递一个空列表将会删除所有的标记
# plt.yticks(())
plt.show()


# 补充知识点
# 1.列表的切片：格式：  list[start : end : step]
# 2.格式化输出：在 print() 函数中，由引号包围的是格式化字符串，它相当于一个字符串模板，可以放置一些转换说明符（占位符）
# 格式化字符串中包含一个%d说明符，它会被后面的 age 变量的值所替代。%是一个分隔符，它前面是格式化字符串，后面是要输出的表达式
# 格式化字符串中也可以包含多个转换说明符，这个时候也得提供多个表达式，用以替换对应的转换说明符；多个表达式必须使用小括号( )

