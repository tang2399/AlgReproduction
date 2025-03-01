import pickle

import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
from sklearn.mixture import GaussianMixture

# 语音数据集目录
source = './dataset/speaker-identification/development_set/'
# 训练集
train_file = './dataset/speaker-identification/development_set_enroll.txt'
# 模型存放目录
modelPath = './speaker_models'

# 特征向量
features = np.asarray(())

# 每几个声音文件属于一个人
num = 5
count = 1

file_path = open(train_file, 'r')
for path in file_path:
    path = path.strip()
    print(path)

    # 依次读取.wav文件的采样率及其本身
    sr, audio = read(source + path)

    # 提取40个维度的特征(MFCC + Δ MFCC)
    vector = extract_features(audio, sr)

    # 将数据装入向量
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    if count == 5:
        # 一个人的5条语音，对应使用一个GMM模型建模
        # 传入需要拟合的高斯分布数，越高越精细；对角矩阵
        model = GaussianMixture(n_components=17, covariance_type='diag', n_init='auto',max_iter=200)
        model.fit(features)
        # 保存每个人对应的GMM模型，根据人名分割
        modelFile = path.split('-')[0] + '.gmm'
        pickle.dump(model, open(modelPath+modelFile, 'wb'))

        features = np.asarray(())
        count = 0
    count += 1



