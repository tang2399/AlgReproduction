from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载新闻数据集
dataset_train = fetch_20newsgroups(data_home="./dataset", subset="train")
dataset_test = fetch_20newsgroups(data_home="./dataset", subset="test")

X_train, y_train = dataset_train.data, dataset_train.target
X_test, y_test = dataset_test.data, dataset_test.target
# print(X_train[0], y_train[0])

# 创建pipeline，用于新闻文本特征提取，提取后再使用逻辑回归
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# 训练模型
pipeline.fit(X_train, y_train)

# 模型预测
y_pred = pipeline.predict(X_test)

print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

