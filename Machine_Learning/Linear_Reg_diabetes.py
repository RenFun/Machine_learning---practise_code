# Author: RenFun
# File: Linear_Reg_diabetes.py
# Time: 2021/09/14


# 调用sklearn库实现多元线性回归
# 使用UCI官方数据集或者sklearn库中的数据集（糖尿病病人）
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# 加载糖尿病数据集：总共有442个样本，每个样本diabetes_X有10个属性描述，分别为：年龄、性别、体质指数、血压、s1~s6（6种血清的化验数据）
# 但需要注意的，以上的数据是经过预处理， 10个特征都做了归一化处理。
# diabetes_y是目标值，一年疾后的病情定量测量，它是一个连续的实数值，符合线性回归模型评估的范畴。
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# 将数据集划分为训练集和测试集，训练集占0.7，随机数种子 = 0 或 none，表示重复实验时获得的随机数将不同
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_y, train_size=0.7, random_state=2)
lr = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
scaler = StandardScaler()
scaler.fit(diabetes_X_train)
diabetes_X_train = scaler.transform(diabetes_X_train)
diabetes_X_test = scaler.transform(diabetes_X_test)
# 使用训练集去拟合线性模型
lr.fit(diabetes_X_train, diabetes_y_train)

# diabetes_y_pred 是测试集的预测值
diabetes_y_pred = lr.predict(diabetes_X_test)
# diabetes_y_train_pred 时训练集的预测值
diabetes_y_train_pred = lr.predict(diabetes_X_train)
print('系数: ', lr.coef_)
print('截距: ', lr.intercept_)

# 绘制图像1
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
plt.savefig("Linear_Reg_diabetes_true&pre.svg")
plt.show()

# 绘制图像2
# 残差值图：通过将预测结果减去对应的目标变量的真实值，便可获得残差值。X轴表示预测结果，Y轴表示残差值。其中一条直线 Y=0，表示残差为0的位置
# plt.title("残差值图")
plt.xlabel('预测值')
plt.ylabel('残差值')
# 训练集数据以圆形，蓝色形式散列；测试集数据以正方形，绿色形式散列
plt.scatter(diabetes_y_train_pred, diabetes_y_train_pred - diabetes_y_train, c='blue', marker='o', label='训练集数据')
plt.scatter(diabetes_y_pred, diabetes_y_pred - diabetes_y_test, c='green', marker='s', label='测试集数据')
plt.legend(loc='upper left')
# y = 0 ，表示残差值为0的位置
plt.hlines(y=0, xmin=0, xmax=300, linestyles='solid', colors='red')
plt.savefig("Linear_Reg_diabetes_residual.svg")
plt.show()

# 模型的性能指标：均方误差，均方根误差，平均绝对误差，决定系数
# mean_squared_error()表示测试集的实际值与预测值的均方方差，即损失函数，越接近0说明模型越好
# r2_score()表示相关指数，也叫决定系数，其值越接近1说明模型越好
print('均方误差: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('均方根误差：%.2f' % np.sqrt(mean_squared_error(diabetes_y_test, diabetes_y_pred)))
print('平均绝对误差：%.2f' % mean_absolute_error(diabetes_y_test, diabetes_y_pred))
print('决定系数：%.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# 补充知识点
# 1.列表的切片：格式：  list[start : end : step]
# 2.格式化输出：在 print() 函数中，由引号包围的是格式化字符串，它相当于一个字符串模板，可以放置一些转换说明符（占位符）
# 格式化字符串中包含一个%d说明符，它会被后面的 age 变量的值所替代。%是一个分隔符，它前面是格式化字符串，后面是要输出的表达式
# 格式化字符串中也可以包含多个转换说明符，这个时候也得提供多个表达式，用以替换对应的转换说明符；多个表达式必须使用小括号( )

