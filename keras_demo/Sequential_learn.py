# 使用Keras Sequential顺序模型

# 对于具有2个类的单输入模型（二分类）
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# 初始化模型，为了代码简明，采用一层一层添加网络的方式
model = Sequential()
model.add(Dense(32, activation='relu',input_dim=100))
model.add(Dense(1, activation= 'sigmoid'))
'''
在模型编译compile接受三个参数：
    -优化器optimizer : 'rmsprop' , 'adagrad'
    -损失函数loss ： 'mse', 'categorical' ,也可以是个目标函数
    -评估标准metrics : 'accuracy'
'''
model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])


# 生成虚拟数据集
data = np.random.random((1000,100)) # 1000行 100列的随机浮点数，浮点数范围 : (0,1)
labels = np.random.randint(2,size=(1000,1)) #[0,2)的随机整型

# 训练模型,以32个样本作为一个batch进行训练
model.fit(data,labels,epochs=10,batch_size=32)



