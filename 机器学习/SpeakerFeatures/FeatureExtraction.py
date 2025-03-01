import numpy as np
import python_speech_features as mfcc
from sklearn import preprocessing


# 计算delta
def calculate_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, cols))
    N = 2
    denominator = 2 * sum([n ** 2 for n in range(1, N + 1)])  # 归一化因子

    for i in range(rows):
        numerator = np.zeros(cols)
        for j in range(1, N + 1):  # j从1到N
            upper = min(i + j, rows - 1)  # 处理上边界
            lower = max(i - j, 0)  # 处理下边界
            numerator += j * (array[upper] - array[lower])
        deltas[i] = numerator / denominator  # 核心公式

    return deltas


# 用来提取语音特征
def extract_features(audio, rate):
    # 生成20个特征维度
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True)
    # 归一化处理
    mfcc_feat = preprocessing.scale(mfcc_feat)

    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat, delta))
    return combined
