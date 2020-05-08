#### 一、实验目的
&ensp;&ensp;理解逻辑回归模型，掌握逻辑回归模型的参数估计算法。

#### 二、实验要求及实验环境

##### 实验要求
1. 实现两种损失函数的参数估计（1，无惩罚项；2. 加入对参数的惩罚）
2.  采用梯度下降、共轭梯度或者牛顿法

##### 实验环境
1. 硬件环境：X64CPU; 4.0Ghz; 16G RAM
2. 软件环境：Windows 64位
3. 开发环境：Python 3.6.8(64位); Anaconda3; Visual Stuio Code

#### 三、设计思想（本程序中的用到的主要算法及数据结构）

问题描述：有 $m$ 组包含 $n$ 个特征的列向量 $\boldsymbol{x}$ 分别对应两个类别 $y \in \{0,1\}$ ,找到一个列向量 $\boldsymbol{w}$ 使得  $\prod{P(y|\boldsymbol{x}, \boldsymbol{w})}$ 取到最大值。所以问题可以形式化表示为：
$$
\underset{\boldsymbol{w}}{\mathrm{argmax}} \prod_{i=1}^n{P(y|\boldsymbol{x}, \boldsymbol{w})} = \underset{\boldsymbol{w}}{\mathrm{argmax}} \sum_{i=1}^n {\ln P(y|\boldsymbol{x},\boldsymbol{w})}\tag{1}
$$
其中 $P(y|\boldsymbol{x}, \boldsymbol{w})$ 可以表示为如下：
$$
P(y=1|\boldsymbol{x},\boldsymbol{w})=\frac{1}{1+ e^{-\boldsymbol{x^Tw}}}
$$
我们定义 $sigmoid$ 函数为 $$\sigma(z) = \frac{1}{1+e^{-z}}$$
所以
$$
\begin{aligned}
    P(y=1|\boldsymbol{x},\boldsymbol{w})&=\sigma(\boldsymbol{x^Tw}) \\ P(y=0|\boldsymbol{x},\boldsymbol{w})&=1-\sigma(\boldsymbol{x^Tw})
\end{aligned}
$$
所以根据 $(1)$ 式可以定义出如下的 $loss$ 函数（标量形式）
$$
\begin{aligned}
    loss &=\sum_{i=1}^n y_i \ln(P(y_i = 1|\boldsymbol{x},\boldsymbol{w})) + (1-y_i) \ln(P(y_i = 0)|\boldsymbol{x},\boldsymbol{w})\\
    &= \sum_{i=1}^ny_i\ln(\sigma(\boldsymbol{x_i^Tw}))+(1-y_i)\ln(1-\sigma(\boldsymbol{x_i^Tw}))
\end{aligned}
$$
根据上述的标量形式，可以转换成如下的向量形式。其中
$$
Y=
 \left[
 \begin{matrix}
   y_1 \\
   y_2 \\
   y_3 \\
   \vdots   \\
   y_m \\
  \end{matrix}
  \right]
X=
 \left[
 \begin{matrix}
   1 & x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(n)} \\
   1 & x_2^{(1)} & x_2^{(2)} & \cdots & x_2^{(n)} \\
   1 & x_3^{(1)} & x_3^{(2)} & \cdots & x_3^{(n)} \\
   \vdots & \vdots & \vdots & \ddots & \vdots &\\
   1 & x_m^{(1)} & x_n^{(2)} & \cdots &x_m^{(n)}
  \end{matrix}
  \right]
W =  \left[
 \begin{matrix}
   w_0 \\
   w_1 \\
   w_2 \\
   \vdots   \\
   w_n \\
  \end{matrix}
  \right]
$$
由于 $(1)$ 式中所表示的正确的概率之和，所以取值越大证明结果越优，但是$loss$ 的优化过程是尽量减小 $loss$ 的值。所以对之前所求的等式取反即可。将上述的 $loss$ 函数转换为向量形式后如下：
$$
loss = -Y^T\ln(\sigma(X\boldsymbol{w}))-(1-Y^T)\ln(1-\sigma(X\boldsymbol{w}))
$$
上式对于 $\boldsymbol{w}$ 的导数如下：
$$
\frac{\partial loss}{\partial \boldsymbol{w}}=X^T(\sigma(X\boldsymbol{w} - Y))
$$
所以只需对 $\boldsymbol{w}$ 进行迭代优化即可。此次实验使用梯度下降法来进行参数优化。
#### 四、实验结果与分析
 - 数据生成
  利用python生成数据：两堆点，分别根据两个中心点 $(0.25, 0.25),(0.75, 0.75)$并且在 $x,y$ 值上添加高斯噪声。这是符合朴素贝叶斯假设的数据，对于不符合朴素贝叶斯假设的数据，可以令生成的点的横纵坐标相等，在这基础之上添加噪声。噪声是可以通过调整$\sigma^2$控制的。
  真实的数据使用了UCI网站上的水仙花的数据，来进行对真实数据的监测分类。
 - 变量及解释
   $num_x$ : 生成的点的个数
   $epoch$ : 训练的轮数
   $\sigma^2$ : 噪声的方差
   $\lambda$ : 正则项的系数
 - 符合高斯分布的数据
  （测试数据均为200个点）

  | 样本数量 | 是否含有惩罚项 | 测试正确率 |
  | :-: | :-: | :-: |
  |200|无|1.00
  |200|有|0.995
  |10|无|0.940
  |10|有|0.970
  样本数量：200，有正则项
  <img src="./Figure_1.png" width = 50%><img src="./Figure_2.png" width = 50%>
  样本数量：200，无正则项
  <img src="./Figure_3.png" width = 50%><img src="./Figure_4.png" width = 50%>
  样本数量：10，无正则项
  <img src="./Figure_5.png" width = 50%><img src="./Figure_6.png" width = 50%>
  样本数量：10，有正则项
  <img src="./Figure_7.png" width = 50%><img src="./Figure_8.png" width = 50%>
  <br></br>
 - 不符合高斯分布的数据
  
  | 样本数量 | 是否含有惩罚项 | 测试正确率 |
  | :-: | :-: | :-: |
  |200|无|0.945
  |200|有|0.975
  |10|无|0.795
  |10|有|0.945

样本数量：200，有正则项
  <img src="./Figure_9.png" width = 50%><img src="./Figure_10.png" width = 50%>
样本数量：200，无正则项
  <img src="./Figure_11.png" width = 50%><img src="./Figure_12.png" width = 50%>
样本数量：10，有正则项
  <img src="./Figure_13.png" width = 50%><img src="./Figure_14.png" width = 50%>
样本数量：10，无正则项
  <img src="./Figure_15.png" width = 50%><img src="./Figure_16.png" width = 50%>
  <br></br><br></br>
 - 真实数据
  在此实验中，使用了UCI的水仙花数据，该数据是一个四维的二分类数据，具体测试结果如下

  |样本数量|是否有惩罚项|训练集正确率|测试集正确率
  |:-:|:-:|:-:|:-:|
  |80 | 无|1.00|1.00
 - 分析
  对于两堆的点（二维数据）进行简单的二分类，或者较低维度（四维数据）的数据，逻辑回归是可以基本处理这类问题的。但是数据是否是满足朴素贝叶斯分类，$loss$ 函数是否有正则项以及训练集点的数目是会影响分类结果的。用点的二分类举例，可以看到无论点是否符合朴素贝叶斯假设，在数量较大的情况下，均可以比较准确的完成分类任务。但是在点的数量很少时，会发生过拟合现象，具体表现在10个点时，无正则项的训练集的准确率较低。加了正则项以后，过拟合现象比较好的受到了抑制。

#### 五、附录：源代码

```python
#生成数据部分( makedata.py )
import numpy as np
import matplotlib.pyplot as plt
import random
import re

def get_data(num_data, sigma, data):
    if data == 'satisfy':#生成符合贝叶斯分布的数据
        num_data = int(num_data / 2)
        X = np.ones((num_data * 2, 2))
        Y = np.ones(num_data * 2)
        for i in range(num_data):
            X[i][0] = 1 + np.random.normal(0, sigma)#横坐标和
            X[i][1] = 1 + np.random.normal(0, sigma)#纵坐标是分别独立生成的
            Y[i] = 1
            X[i + num_data][0] = 2 + np.random.normal(0, sigma)
            X[i + num_data][1] = 2 + np.random.normal(0, sigma)
            Y[i + num_data] = 0
        Y = Y[:, np.newaxis]
        return X, Y
    if data == 'dissatisfy':#生成不符合贝叶斯分布的数据
        num_data = int(num_data / 2)
        X = np.ones((num_data * 2, 2))
        Y = np.ones(num_data * 2)
        for i in range(num_data):
            X[i][0] = np.random.normal(0.25, sigma)#横坐标随机生成
            X[i][1] = X[i][0] + np.random.normal(0, sigma)#纵坐标在横坐标的数值附近生成
            Y[i] = 1
            X[i + num_data][0] = np.random.normal(0.75, sigma)
            X[i + num_data][1] = X[i + num_data][0] + np.random.normal(0, sigma)
            Y[i + num_data] = 0
        Y = Y[:, np.newaxis]
        return X, Y
    if data == 'iris':#载入水仙花数据
        X_train = np.zeros((80, 4))#取80个数据作为训练集
        Y_train = np.ones((40, 1), dtype = float)
        Y_train = np.concatenate((Y_train, np.zeros((40, 1))), axis = 0)
        X_test  = np.zeros((20, 4))#取20个点作为测试集
        Y_test  = np.ones((10, 1), dtype = float)
        Y_test  = np.concatenate((Y_test, np.zeros((10, 1))), axis = 0)
        filepath = "\data\iris.txt"
        f = open(filepath)
        lines = f.readlines()
        p = re.compile(',')
        for i in range(100):
            l = p.split(lines[i])
            if(i <= 39):#第一类的训练集数据
                X_train[i, np.array([0, 1, 2, 3], dtype = 'i')] = [l[0], l[1], l[2], l[3]]
            if(i <= 49 and i > 39):#第一类的测试集数据
                X_test[i - 40, np.array([0, 1, 2, 3], dtype = 'i')] = [l[0], l[1], l[2], l[3]]
            if(i <= 89 and i > 49):#第二类的训练集数据
                X_train[i - 10, np.array([0, 1, 2, 3], dtype = 'i')] = [l[0], l[1], l[2], l[3]]
            if(i > 89):#第二类的测试集数据
                X_test[i - 80, np.array([0, 1, 2, 3], dtype = 'i')] = [l[0], l[1], l[2], l[3]]
        return X_train, Y_train, X_test, Y_test

```

```python
#网络模型、优化器、loss函数定义部分( model.py )
import numpy as np


class network(object):
    def __init__(self, args):#网络数据初始化
        self.n = args["num_x"]
        if(args['data'] == 'iris'):#水仙花数据有四维，再加上一列1共五维
            self.m = 5
        else:
            self.m = 3#点数据有两维，再加上一列1共三维
        self.w = np.mat(np.random.rand(self.m, 1))#系数矩阵初始化为随机

    def paramaters(self):#获得参数列表
        return self.w

    def backward(self, gradient):#梯度反向传播
        self.w = self.w + gradient

    def forward(self, x):#矩阵计算直接得到预测值
        return np.dot(x, self.w)

def sigmoid(z):#sigmoid函数
    return 1 / (1 + np.exp(-z))

class binary_crossentropy_Loss(object):#loss函数
    def __init__(self, y_pre, y_true, w, lamda):
        self.y_true = np.mat(y_true)#y的真实值
        self.y_pre  = np.mat(y_pre)#y的预测值
        self.w      = np.mat(w)#参数
        self.lamda  = lamda#正则项的lamda
        self.loss   = - self.y_true.T.dot(np.log(sigmoid(self.y_pre))) -\
                        (1 - self.y_true.T).dot(np.log(1 - sigmoid(self.y_pre))) +\
                        self.lamda * self.w.T.dot(self.w)#loss计算
                        

    def backward(self, optimizer, net):#调用网络的函数和优化器的函数进行梯度的反向传播
        net.backward(optimizer.grad(self.y_true))


class GDoptim(object):#梯度下降法，计算梯度的方法报告已给出
    def __init__(self, w, x, LR, lamda):
        self.w = np.mat(w)
        self.x = np.mat(x)
        self.lr = LR
        self.lamda = lamda
    
    def grad(self, y_true):#计算梯度
        return self.lr * (self.x.T.dot(y_true - sigmoid(self.x.dot(self.w))) - self.lamda * self.w)

```

```python
#主函数部分( main.py )
import numpy as np
import random
import matplotlib.pyplot as plt
import makedata
import model as nn

args = dict(
    epoch = 100,
    num_x = 10,
    sigma = 0.15,
    lamda = 0.,
    LR    = 0.1,
    data  = 'satisfy' # iris, dissatisfy, satisfy
)

num_x = args['num_x']
sigma = args['sigma']
epoch = args['epoch']
lr    = args['LR']
lamda = args['lamda']
data  = args['data']

if(data != 'iris'):
    train_X, train_Y = makedata.get_data(num_x, sigma, data)
else:
    num_x = 80
    train_X, train_Y, test_X, test_Y = makedata.get_data(0, 0, data)
    test_inputs = np.insert(test_X, 2, values = np.ones((1, 20)), axis = 1)
x0 = np.ones((1, num_x))
inputs = np.insert(train_X, 2, values = x0, axis = 1)
net = nn.network(args)
if(data != 'iris'):#如果不是水仙花数据，确定绘图板的大小
    x_min = min(train_X[...,0]) - 0.1
    x_max = max(train_X[...,0]) + 0.1
    y_min = min(train_X[...,1]) - 0.1
    y_max = max(train_X[...,1]) + 0.1
    plt.ion()
for i in range(epoch):#迭代训练
    output = net.forward(inputs)#计算预测值
    optimizer = nn.GDoptim(net.paramaters(), inputs, lr, lamda)#初始化优化器
    loss_function = nn.binary_crossentropy_Loss(output, train_Y, net.paramaters(), lamda) #初始化损失函数
    loss_function.backward(optimizer, net)#利用损失函数，优化器，网络进行梯度反向传播

    loss = loss_function.loss
    print(loss.A[0]/num_x)#输出loss的值
    #如果不是水仙花数据进行可视化
    if data != "iris":
        plt.cla()
        w = net.paramaters()
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        for i in range(num_x):
            if(output[i] > 0):
                plt.plot(train_X[i][0], train_X[i][1], 'ro')
            else:
                plt.plot(train_X[i][0], train_X[i][1], 'bo')
        X = np.linspace(min(train_X[...,0]), max(train_X[...,0]), 5)
        Y = - w[0].A[0] / w[1].A[0] * X - w[2].A[0] / w[1].A[0]
        plt.plot(X, Y)
        plt.pause(0.1)
#如果不是水仙花数据进行可视化
if data != "iris":
    plt.figure()
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    for i in range(num_x):
        if(train_Y[i] > 0):
            plt.plot(train_X[i][0], train_X[i][1], 'ro')
        else:
            plt.plot(train_X[i][0], train_X[i][1], 'bo')
    plt.ioff()
    plt.show()
else:
    w = net.paramaters()
    output = test_inputs.dot(w)
    count = 0
    for i in range(len(output)):
        if(output[i] > 0 and test_Y[i] == 1):
            count = count + 1
        if(output[i] < 0 and test_Y[i] == 0):
            count = count + 1
    print(count / 20)
```