from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 载入 MNIST 手写数字图片数据集
dataset = fetch_openml("mnist_784", data_home="./dataset", parser='auto')

# # 绘制示例图片及其类别
# img_num = 0
# img = np.array(dataset.data)[img_num]
# print(dataset.target[img_num])
# img = img.reshape(28, 28)
# plt.imshow(img, cmap='gray')
# plt.show()

# 利用标准归一化对数据进行预处理
scaler = StandardScaler()
X = scaler.fit_transform(dataset.data)
y = dataset.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# 模型预测
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# 均方差
print("训练集准确率：%.2f" % accuracy_score(y_train, y_train_pred))
print("测试集准确率：%.2f" % accuracy_score(y_test, y_pred))

# 预测图片类别
img_num = 0
img = np.array(dataset.data)[img_num]
print(model.predict([img]))
img = img.reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()

