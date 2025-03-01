from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

# 载入 MNIST 手写数字图片数据集
dataset = fetch_openml("mnist_784", data_home="./dataset", parser='auto')

for img_num in range(50):
    # 图片处理
    img = np.array(dataset.data)[img_num]
    img = img.reshape(28, 28)

    # 绘图
    plt.subplot(5, 10, img_num + 1)
    plt.imshow(img, cmap='gray')

    # 隐藏坐标轴
    plt.axis("off")

    # 显示图片类别
    plt.title(dataset.target[img_num])

plt.show()