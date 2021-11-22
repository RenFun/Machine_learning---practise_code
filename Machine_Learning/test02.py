# Author: RenFun
# File: test02.py
# Time: 2021/06/10


# 测试sigmoid函数图像绘制,二维平面
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


nums = np.arange(-10, 10, 1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


# 测试sigmoid函数图像，三维立体图像


