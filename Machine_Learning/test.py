# Author: RenFun
# File: test.py
# Time: 2021/06/10


# 线性判别分析LDA：给定训练数据集，将其投影到一条直线上，使得同类样本的投影点尽可能近，异类样本的投影点尽可能远
# 可适用于多分类问题

import numpy as np
import matplotlib.pyplot as plt
x = np.arange(100)
y = np.square(x)
plt.plot(x, y, label='test')
plt.legend(loc='best')
plt.savefig('test.svg')
