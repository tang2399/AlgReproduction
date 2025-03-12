import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 加载数据
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 转换为XGBoost的数据格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 参数设置
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'max_depth': 5,                  # 树的最大深度
    'eta': 0.1,                      # 学习率
    'subsample': 0.8,                # 每棵树采样80%数据
    'lambda': 1.0                    # L2正则
}

# 训练模型
model = xgb.train(params, dtrain, num_boost_round=100)

# 预测
predictions = model.predict(dtest)