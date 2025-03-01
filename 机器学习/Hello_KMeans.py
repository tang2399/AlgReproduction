import os

os.environ["OMP_NUM_THREADS"] = "1"  # 禁用多线程

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#
# # 定义几个点
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [4, 2], [4, 4], [4, 0],])
X = np.random.randint(0, 100, size=(100, 2))
print(X)

# 初始化KMeans聚类模型
model = KMeans(n_clusters=5, random_state=0, n_init='auto')
# 模型训练
model.fit(X)

# 模型输出
unique_labels = np.unique(model.labels_)
colors = plt.cm.get_cmap('Paired', len(unique_labels))
for i in range(len(X)):
    print('对于点：', X[i], '，它的类别为：', model.labels_[i])
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=colors, s=23, edgecolor='black', linewidths=0.5)

# 输出模型聚类中心点
print('模型的聚类中心为：', model.cluster_centers_)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='x', s=71, color='red')

plt.show()
