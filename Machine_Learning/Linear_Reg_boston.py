# Author: RenFun
# File: Linear_Reg_boston.py
# Time: 2021/10/27


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


diabetes_X, diabetes_y = datasets.load_boston(return_X_y=True)
# 将数据集划分为训练集和测试集，训练集占0.7，随机数种子 = 0 或 none，表示重复实验时获得的随机数将不同
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_y, train_size=0.7, random_state=2)
lr = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
# 使用训练集去拟合线性模型
lr.fit(diabetes_X_train, diabetes_y_train)

# diabetes_y_pred 是测试集的预测值
diabetes_y_pred = lr.predict(diabetes_X_test)
# diabetes_y_train_pred 时训练集的预测值
diabetes_y_train_pred = lr.predict(diabetes_X_train)

print('均方误差: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('均方根误差：%.2f' % np.sqrt(mean_squared_error(diabetes_y_test, diabetes_y_pred)))
print('平均绝对误差：%.2f' % mean_absolute_error(diabetes_y_test, diabetes_y_pred))
print('决定系数：%.2f' % r2_score(diabetes_y_test, diabetes_y_pred))


plt.rcParams['font.sans-serif'] = ['SimHei']        # 用来正常显示中文，设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
# plt.title("预测值与真实值之间的关系", fontsize=20)
plt.xlabel("测试集样本")
plt.ylabel("预测值和真实值")
t = np.arange(1, len(diabetes_X_test)+1, 1)
plt.plot(t, diabetes_y_test, color='red', label='真实值')
plt.plot(t, diabetes_y_pred, color='blue', label='预测值')
# 显示图例，设置图例的位置
plt.legend(loc='upper left')
plt.grid(b=True, linestyle='--')
# plt.savefig("Linear_Reg_diabetes_true&pre.svg")
plt.show()