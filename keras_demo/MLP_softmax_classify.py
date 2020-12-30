'''基于多层感知机(MLP)的多分类'''

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
import numpy as np

# 生成数据集
x_train = np.random.random((1000,20)) # 1000行，20列 ;1000个数据集
y_train = keras.utils.to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)
x_test = np.random.random((100,20)) # 100个测试集
y_test = keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)

# 构建模型
model = keras.Sequential()
model.add(Dense(64,activation='relu',input_dim=20))
# 为了防止过拟合，训练时丢弃某些神经元
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# 训练模型
model.fit(x_train,y_train,epochs=20,batch_size=128)

# evaluate返回损失值和你选定的指标值
score = model.evaluate(x_test,y_test,batch_size=128)
print(score)
