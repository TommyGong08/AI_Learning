import tensorflow as tf
import numpy as np

# 数据集
data = np.array([
    [-2,-1], # Alice
    [2,6], # Bob
    [17,4], # Charlie
    [-1,-6],
]
)
y_all_trues = np.array([
    1, #Alice
    0,#Bob
    0,#Charlie
    1,# Diana
])

# 构建神经网络
inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(2,use_bias=True)(inputs)
outputs = tf.keras.layers.Dense(1,use_bias=True,activation='sigmoid')(x)
m = tf.keras.Model(inputs,outputs)

# 编译模型 lr是learning_rate
m.compile(tf.keras.optimizers.SGD(lr=0.1), 'mse' )

#训练模型，以一个样本为一个batch进行迭代，verbose输出日志
m.fit(data,y_all_trues,epochs=1000,batch_size=1,verbose=1)

emily = np.array([[-1,-3]])
frank = np.array([[20,2]])
print(m.predict(emily)) # 0.9918876
print(m.predict(frank)) # 1.5566727e-15


