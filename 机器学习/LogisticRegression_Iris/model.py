from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
# feature_names = dataset.feature_names  # 特征名称列表
# DESCR = dataset.DESCR  # 数据集详细描述
# print(X, y, feature_names, DESCR)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建逻辑回归模型，模型会根据y自动决定是几分类问题，max_iter为迭代次数
model = LogisticRegression(max_iter=1000)
# model = LogisticRegression(multi_class='ovr')  # 将多分类任务转换成多个二分类任务
# model = LogisticRegression(multi_class='multinomial')  # 利用Softmax直接做多分类任务

# 训练模型
model.fit(X_train, y_train)

# 模型预测
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# 均方差
print("训练集准确率：%.2f" % accuracy_score(y_train, y_train_pred))
print("测试集准确率：%.2f" % accuracy_score(y_test, y_pred))