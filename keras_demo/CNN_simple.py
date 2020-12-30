'''一个简单的卷积神经网络'''
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD
import numpy as np

# 生成虚拟的数据集
x_train = np.random.random((100,100,100,3)) # 100个数据集，100*100*3图像
y_train = keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)
x_test = np.random.random((20,100,100,3)) # 20个测试集
y_test = keras.utils.to_categorical(np.random.randint(10,size=(20,1)),num_classes=10)

# 模型构建
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
# 十个类别
model.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# 训练模型
model.fit(x_train,y_train,batch_size=32,epochs=10)

# evaluate返回损失值和你选定的指标值
score = model.evaluate(x_test,y_test,batch_size=32)
print(score)