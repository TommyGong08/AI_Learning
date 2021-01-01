'''
设计一个BP神经网络用于大作业第三问 训练棋局的评估函数
BP神经网络结构：
    输入层：input_dim = 361(棋盘19*19）
    第一层：64个神经元
    第二层：32个神经元（还可以测一下64的效果）
    输出层：一个神经元
'''

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
import numpy as np

# 加载数据集，尚未制作。先做一个虚拟数据集
# 生成数据集
x_train = np.random.randint(0,10001,(1000,361)) # 1000行，361列 ;1000个数据集 [0,10000)
y_train = np.random.randint(0,10001,(1000,1))  # 产生1000个 0-10000的随机数
x_test = np.random.randint(0,10001,(100,361)) # 100个测试集
y_test = np.random.randint(0,10001,(100,1)) # 产生1000个 0-10000的随机数

# 构建模型
model = keras.Sequential()
model.add(Dense(64,activation='relu',input_dim=361))
# 为了防止过拟合，训练时丢弃某些神经元
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='softmax'))
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])

# 训练模型
model.fit(x_train,y_train,epochs=2000,batch_size=100)

# evaluate返回损失值和你选定的指标值
score = model.evaluate(x_test,y_test,batch_size=100)
print(score)