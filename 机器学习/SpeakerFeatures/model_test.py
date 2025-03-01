# 语音数据集目录
import os
import pickle

import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features


source = './dataset/speaker-identification/development_set/'
# 测试集
test_file = './dataset/speaker-identification/development_set_test.txt'
file_path = open(test_file, 'r')
# 模型存放目录
modelPath = './speaker_models'
# 得到每个模型文件路径
modelFiles = [os.path.join(modelPath, fname) for fname in
              os.listdir(modelPath) if fname.endswith('.gmm')]
# 加载模型
models = [pickle.load(open(fname, 'rb')) for fname in modelFiles]
# 获取人名
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
            in modelFiles]
# 遍历测试集
for path in file_path:
    # 依次读取.wav文件的采样率及其本身
    sr, audio = read(source + path)

    # 提取40个维度的特征(MFCC + Δ MFCC)
    vector = extract_features(audio, sr)
    log_likelihood = np.zeros(len(models))

    # 使用模型
    for i in range(len(models)):
        model = models[i]
        # 评价
        scores = np.array(model.score(vector))
        # 加和后即为对数似然
        log_likelihood[i] = scores.sum()

    # 最终概率最大的人的索引
    winner = np.argmax(log_likelihood)
    print("\tdetected as -", speakers[winner])



