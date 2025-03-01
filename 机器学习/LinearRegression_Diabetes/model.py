from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载糖尿病数据集
dataset = datasets.load_diabetes()
X = dataset.data
y = dataset.target
# feature_names = dataset.feature_names  # 特征名称列表
# DESCR = dataset.DESCR  # 数据集详细描述
# print(X, y, feature_names, DESCR)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 调用基于普通最小二乘法（OLS）的线性回归模型，适用于预测连续型目标变量
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 模型预测
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# 均方差
print("训练集均方误差：%.2f" % mean_squared_error(y_train, y_train_pred))
print("测试集均方误差：%.2f" % mean_squared_error(y_test, y_pred))
