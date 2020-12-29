'''
手写神经网络实现根据人的身高、性别判断性别
'''

from my_Neuron import *
import numpy as np

def mse_loss(y_true,y_pred):
    # 用最小平方差作为损失函数
    return ((y_true - y_pred) ** 2).mean()


class NeuralNetwork:
    '''
    此神经元含有
      -两个输入神经元
      -两个输入值
      -一个输出神经元
    每个神经元的参数
        权重 w = [0,1]
        偏置 b = 0
    '''
    def __init__(self):

        # 随机初始化权重
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # 随机初始化偏置
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self,x):
        out_h1 = sigmoid(self.w1*x[0] + self.w2*x[1]+self.b1)
        out_h2 = sigmoid(self.w3*x[0] + self.w4*x[1]+self.b2)
        out_o1 = sigmoid(self.w5*out_h1 + self.w6*out_h2+self.b3)

        return out_o1
    def train(self,data,all_y_trues):

        learning_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x,y_true in zip(data,all_y_trues):
                temp_h1 = self.w1*x[0] + self.w2*x[1] + self.b1
                h1 = sigmoid(temp_h1)

                temp_h2 = self.w3*x[0] + self.w3*x[1] + self.b2
                h2 = sigmoid(temp_h2)

                temp_o1 = self.w4*h1 + self.w6*h2 +self.b3
                o1 = sigmoid(temp_o1)
                y_pred = o1

                # 计算偏导数，一堆微积分公式
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # 输出神经元o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(temp_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(temp_o1)
                d_ypred_d_b3 = deriv_sigmoid(temp_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(temp_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(temp_o1)

                # 输入神经元h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(temp_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(temp_h1)
                d_h1_d_b1 = deriv_sigmoid(temp_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(temp_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(temp_h2)
                d_h2_d_b2 = deriv_sigmoid(temp_h2)

                # 更新权重和偏置
                self.w1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                self.w5 -= learning_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learning_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learning_rate * d_L_d_ypred * d_ypred_d_b3

                # 每10个epochs输出一次损失函数
                if epoch % 10 == 0 :
                    # 返回的是一个根据func()函数以及维度axis运算后得到的的数组
                    y_preds = np.apply_along_axis(self.feedforward, 1 , data)
                    loss = mse_loss(all_y_trues,y_preds)
                    print("Epoch %d loss %.3f" % (epoch,loss))


# 数据集
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# 开始训练
network = NeuralNetwork()
network.train(data, all_y_trues)

# 测试集
Tommy = np.array([-7, -3]) # 128 pounds, 63 inches
Mary = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(Tommy)) # 0.949 - F
print("Frank: %.3f" % network.feedforward(Mary)) # 0.000 - M
