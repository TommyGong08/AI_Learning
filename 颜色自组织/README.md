
基于SOFM的颜色自组织(Python)
====

这是人工智能基础课上老师布置的一道思考题 

如何将无序的颜色图组织成有序的颜色图？

![quiz](/som_cm/pic/test.png/) 

#### 思路  
用RGB三原色值代表各种颜色，作为自组织映射网的输入，即输入3维的数据。
以平面上二维的点作为自组织映射网的输出。 
当输入某颜色值之后，各神经元相互竞争，竞争获胜的神经元上显示相应颜色。 

#### 代码解析  
![deteil_1](/som_cm/pic/code1.png)   
随机从输入样本集合中采样，获得当前输入数据
初始化：设置32*32的map

![deteil_2](/som_cm/pic/code2.png)  
训练过程：
计算当前输入数据下竞争获胜的神经元，更新权重
 
![deteil_3](/som_cm/pic/code3.png)  
更新学习率和协同区域大小 

#### 依赖  

* **python3.6**
* **NumPy**
* **SciPy**
* **matplotlib**
* **OpenCV**


#### 运行程序

把需要颜色组织的图片命名为 ```test.png```
``` bash
  > python main.py
```

#### 结果

![result](/som_cm/pic/result.png) 

##### 代码魔改自 https://github.com/tody411/SOM-ColorManifolds  
听说nice的人都会顺手star以下哟~
