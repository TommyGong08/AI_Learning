import numpy as np

def sigmoid(x):
    # 激活函数是 f(x) = 1 / ( 1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # 计算激活函数的倒数
    fx = sigmoid(x)
    return fx * (1 - fx)

class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias

    # 计算前馈神经元的值
    def feedforward(self,inputs):
        # 输入值和权值点成
        total = np.dot(self.weights, inputs) + self.bias
        # 最后用激活函数激活一下
        return sigmoid(total)

weights = np.array([0,1])
bias = 4
n = Neuron(weights,bias)

# 两个神经元输入值分别为2， 3
x = np.array([2,3])
print(n.feedforward(x))
# x = 0.9990889488055994