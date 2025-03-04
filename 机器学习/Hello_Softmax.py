import math


def Softmax(Input):
    """
    Softmax的计算方式：对每个元素计算指数，再计算指数和，最后分别处以指数和得到概率分布
    :param Input:输入向量
    :return:概率
    """
    e_Input, Output = [], []
    for x in Input:
        e_Input.append(math.exp(x))
    for x in e_Input:
        Output.append(x / sum(e_Input))
    for i in range(len(Input)):
        print(Input[i], ": %.2f" % (Output[i]*100) + '%')
    return Output


X = [2, 3, 5, 6]
Softmax(X)
