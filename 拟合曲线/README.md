#### 一、实验目的
&ensp;&ensp;掌握最小二乘法（无惩罚项损失函数）、掌握加惩罚项（2范数）的损失函数优化、梯度下降法、共轭梯度法。理解过拟合、掌握克服过拟合的方法（如加惩罚项、增加样本）

#### 二、实验要求及实验环境

##### 实验要求
1. 生成数据，加入噪声
2. 用高阶多项式函数拟合曲线
3. 用解析解求解两种loss的最优解（无正则项和有正则项）
4. 优化方法求解最优解（梯度下降，共轭梯度）
5. 用得到的实验数据，解释过拟合
6. 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果。
7. 语言不限，可以用matlab，python。求解解析解时可以利用现成的矩阵求逆。梯度下降，共轭梯度要求自己求梯度，迭代优化自己写。不许用现成的平台，例如pytorch，tensorflow的自动微分工具。

##### 实验环境
1. 硬件环境：X64CPU; 4.0Ghz; 16G RAM
2. 软件环境：Windows 64位
3. 开发环境：Python 3.6.8(64位); Anaconda3; Visual Stuio Code

#### 三、设计思想（本程序中的用到的主要算法及数据结构）

首先需要提出误差方程 (${\rm loss}$ ${\rm function}$)，即：预测数据和真实数据之间的差距。方程如下：
$$loss(\hat y,y)={\frac{1}{n}}{\sum_{i=1}^n {(y_i-\hat y_i)^2}} \tag{1} $$
其中$\hat y$是求到的多项式拟合出的结果，同时作为预测数据。多项式是如下定义的：
$$\hat y=w_0+w_1x+w_2x^2+......+w_mx^m \tag{2}$$
这样每有一个输入值 $x_i$ 均会得到一个预测值 $\hat y_i$ ，为了加速运算并且减少数据中噪声的影响，我们将 $x$ 作为一个矩阵作为输入，得到一个列向量输出 $\hat y_i$ 。具体计算式如下：
$$Xw=\hat y \tag{3}$$
其中
$$
X=
 \left[
 \begin{matrix}
   1 & x_1 & x_1^2 & \cdots & x_1^m \\
   1 & x_2 & x_2^2 & \cdots & x_2^m \\
   1 & x_3 & x_3^2 & \cdots & x_3^m \\
   \vdots & \vdots & \vdots & \ddots & \vdots &\\
   1 & x_n & x_n^2 & \cdots &x_n^m
  \end{matrix}
  \right]
$$
$$
w=
 \left[
 \begin{matrix}
   w_0 \\
   w_1 \\
   w_2 \\
   \vdots   \\
   w_m \\
  \end{matrix}
  \right]
$$
$$
\hat y=
 \left[
 \begin{matrix}
   y_1 \\
   y_2 \\
   y_3 \\
   \vdots   \\
   y_n \\
  \end{matrix}
  \right]
$$
这样直接利用矩阵的计算就可以一次性完成 $n$ 个 $x$ 对应的输出。所以$(1)$式中的损失函数可以改成如下的矩阵计算形式：
$$ 
loss=\frac {1}{n} (\hat y - y)^T(\hat y - y) \tag{4} $$
将$(3)$式带入其中，可以得到：
$$
\begin{aligned}
loss & =\frac {1}{n} (Xw - y)^T(Xw - y)\\
     & =\frac {1}{n} (w^TX^TXw - w^TX^Ty - y^TXw+y^Ty)\\ \tag{5}
\end{aligned}
$$

  - 梯度下降法
     这个方法的思想是，根据 $(5)$ 式，可以得到 $loss$ 是一个关于 $w$ 的函数，所以 $loss$ 的大小是关于 $w$ 的变化而变化的。利用线性代数的知识，可以对 $(5)$ 式求关于 $w$ 的偏导数。如下：<br></br>
$$
\begin{aligned}
\frac {\partial loss}{\partial w} & = \frac {1}{n}      \frac {\partial (w^TX^TXw - w^TX^Ty - y^TXw+y^Ty)} {\partial w}\\
& = \frac {1}{n} \left (\frac {\partial w^TX^TXw}{\partial w} - \frac {\partial w^TX^Ty}{\partial w} - \frac {\partial y^TXw}{\partial w} + \frac {\partial y^Ty}{\partial w} \right) 
\end{aligned} \tag{6}
$$
对上式中的每一项进行分析$\frac {\partial y^Ty}{\partial w}$中不含 $w$ 项，所以该项对 $w$ 求偏导为 $0$ 
$$\frac {\partial y^Ty}{\partial w} = 0 $$
其余三项利用线性代数和矩阵求导知识可以分别得到导函数的表达式：
$$
\begin{aligned}
\frac {\partial w^TX^TXw}{\partial w} &= 2X^TXw \\
\frac {\partial w^TX^Ty}{\partial w} &= X^Ty \\
\frac {\partial y^TXw}{\partial w} &= X^Ty
\end{aligned}
$$
综上
$$
\begin{aligned}
\frac {\partial loss}{\partial w} &= 2X^TXw - 2X^Ty\\
&=\frac {2}{n}X^T(Xw-y)
\end{aligned}\tag{7}
$$
$(7)$ 式便是 $loss$ 对 $w$ 的偏导数，可以证明 $loss$ 函数是一个下凸函数，所以在对于任意一组 $w$ 我们可以计算出在该位置时导数。也就是梯度。而沿着梯度的反方向对 $w$ 进行修正便可以使 $loss$ 降低。
  - 共轭梯度法
该方法同样为迭代算法， 在解 $Xw-y=0$方程时，由于求矩阵的逆是十分复杂的，所以可以使用共轭梯度法来迭代计算。共轭梯度法的大致思路是将所求的 $w$ 拆解为 $m+1$ 共轭向量，每一次迭代可以得到一维的方向和步长。所以理论上迭代 $m+1$ 即可得到准确解。推理步骤不再列出了，下面给出计算步骤：
$$
\forall w \in \mathbb{R}^m, r^{(0)}=y-Xw^{(0)},p^{(0)}=r^{(0)}
$$
$ k = 0,1,2 \cdots$
$$
\begin{aligned}
\alpha_k &= \frac {(r^{(k)}, r^{(k)})}{(p^{(k)}, Ap^{(k)})}\\
w^{(k+1)} &= w^{(k)} + \alpha_kp{(k)}\\
r^{(k+1)} &= r^{(k)} - \alpha_kAp^{(k)}, \beta_k=\frac {(r^{(k+1)}, r^{(k+1)})}{(r^{(k)}, r^{(k)})}\\
p^{(k+1)} &= r^{(k+1)} + \beta_kp^{(k)}
\end{aligned}
$$
这样迭代了 $m+1$ 次后即可得到所需要的 $w$.
  - 无正则项的解析解
根据 $(7)$ 式结合 $loss$ 函数是一个下凸函数可以得到，在 $\frac {\partial loss}{\partial w}=0$ 时 $loss$ 可以取到最小值。所以直接解矩阵方程即可得到解析解。 $X$ 必定不全为 $0$ ,所以 $Xw-y=0$ 。由于 $X$ 不一定存在逆矩阵，所以在等式两侧同时乘以 $X^T$ ，可以确定 $X^TX$ 是存在逆矩阵的。推导过程如下：

$$
\begin{aligned}
\frac {\partial loss}{\partial w} &= 0\\
\frac {2}{n}X^T(Xw-y) &= 0\\
Xw-y &= 0\\
Xw &= y \\
X^TXw &= X^Ty\\
w &= (X^TX)^{-1}X^Ty
\end{aligned}
$$
所以不带有正则项的解析解为 $w = (X^TX)^{-1}X^Ty$ 
  - 有正则项的解析解
在没有正则项时解析解会发生比较严重的过拟合，同时观察生成的 $w$ 可以发现在阶数较高时 $w$ 会十分巨大，这也是过拟合导致的。所以我们可以在 $loss$ 中加入 $w$ 的二范数的平方来限制过拟合：
$$
\left\|w \right\|_2^2 = \lambda {\sum_{i=1}^nw_i^2}
$$
矩阵形式：
$$
\left\|w \right\|_2^2 = \lambda w^Tw \tag{8}
$$
由于 $w$ 的二范数和 $loss$ 的量纲不同，所以需要在 $(9)$ 式之前乘系数 $\lambda$ 来控制其对 $loss$ 的影响。 $loss'$ 的导数如下：
$$
\frac {\partial loss}{\partial w} = 2X^TXw - 2X^Ty+2\lambda Iw
$$
令 $\frac {\partial loss}{\partial w} = 0$ 可得以下推导
$$
\begin{aligned}
2X^TXw - 2X^Ty+2\lambda Iw &= 0\\
(X^TX-\lambda I)w &= X^Ty\\
设A = X^TX-\lambda I\\
Aw &= X^Ty\\
A^TAw &= A^TX^Ty\\
w &= (A^TA)^{-1}A^TX^Ty
\end{aligned}
$$
所以 $w = (A^TA)^{-1}A^TX^Ty$ 为带有正则项的解析解
#### 四、实验结果与分析
 - 数据生成
  利用python生成数据 $y=sin2\pi x$ ，并且在 $y$ 值上添加高斯噪声。其噪声是可以通过调整$\sigma^2$控制的。
 - 变量及解释
   $n$ : 生成的数据的组数
   $m$ : 拟合的多项式的阶数
   $epoch$ : 训练的轮数
   $\sigma^2$ : 噪声的方差
   $\lambda$ : 正则项的系数
   接下来的图片中，绿色线为原函数绘制的曲线，红色线为拟合的多项式绘制的曲线。
 - 梯度下降法实验
  $n=10,m=1,\sigma^2=0.2$
  <img src="./result/10,1.png" width = 30%>
  $n=10,m=4,\sigma^2=0.2$
  <img src="./result/10,4.png" width = 40%>
  $n=10,m=10,\sigma^2=0.2$
  <img src="./result/10,10.png" width = 40%>
  $n=10,m=15,\sigma^2=0.2$
  <img src="./result/10,15.png" width = 40%>
  $n=100,m=1,\sigma^2=0.2$
  <img src="./result/100,1.png" width = 40%>
  $n=100,m=4,\sigma^2=0.2$
  <img src="./result/100,4.png" width = 40%>
  $n=100,m=10,\sigma^2=0.2$
  <img src="./result/100,10.png" width = 40%>
  $n=100,m=100,\sigma^2=0.2$
  <img src="./result/100,15.png" width = 40%>
  可以看到对于原函数的拟合一阶的多项式都是无法拟合的。点数较少比如10个点的情况下，10阶和15阶都产生了较严重的过拟合。点数较多时，10阶过拟合现象并不明显，15阶依然发生了过拟合。但是无论点的数量是多少，4阶均可基本完美拟合原曲线。
 - 共轭梯度法实验
  $n=10,m=1,\sigma^2=0.2$
  <img src="./result/10,11.png" width = 40%>
  $n=10,m=4,\sigma^2=0.2$
  <img src="./result/10,41.png" width = 40%>
  $n=10,m=10,\sigma^2=0.2$
  <img src="./result/10,101.png" width = 40%>
  $n=10,m=15,\sigma^2=0.2$
  <img src="./result/10,151.png" width = 40%>
  $n=100,m=1,\sigma^2=0.2$
  <img src="./result/100,11.png" width = 40%>
  $n=100,m=4,\sigma^2=0.2$
  <img src="./result/100,41.png" width = 40%>
  $n=100,m=10,\sigma^2=0.2$
  <img src="./result/100,101.png" width = 40%>
  $n=100,m=15,\sigma^2=0.2$
  <img src="./result/100,151.png" width = 40%>
  由于共轭梯度法与梯度向量的思想基本相同，是优化迭代的算法，只是在计算过程中的思路产生了改变，所以对曲线拟合的情况基本相同。在点数较少时较高阶的多项式发生了严重的过拟合。点数多的时候，过拟合现象并不明显。
 - 无正则项的解析解实验
  $n=10,m=4,\sigma^2=0.2$
  <img src="./result/10,42.png" width = 40%>
  $n=10,m=10,\sigma^2=0.2$
  <img src="./result/10,92.png" width = 40%>
  $n=10,m=15,\sigma^2=0.2$
  <img src="./result/10,152.png" width = 40%>
  $n=100,m=4,\sigma^2=0.2$
  <img src="./result/100,42.png" width = 40%>
  $n=100,m=15,\sigma^2=0.2$
  <img src="./result/100,152.png" width = 40%>
  可以看到在点数较少时，较低阶的多项式可以比较接近曲线。但是阶数大于等于点数时，由于算法本身的特性，拟合的曲线将会过所有的点，产生严重的过拟合。点数较多时低阶多项式依然可以较好拟合曲线，阶数高时仍然会产生过拟合。
 - 有正则项的解析解实验
  $n=10,m=4,\sigma^2=0.2$
  <img src="./result/10,43.png" width = 40%>
  $n=10,m=10,\sigma^2=0.2$
  <img src="./result/10,93.png" width = 40%>
  $n=10,m=15,\sigma^2=0.2$
  <img src="./result/10,153.png" width = 40%>
  $n=100,m=4,\sigma^2=0.2$
  <img src="./result/100,43.png" width = 40%>
  $n=100,m=15,\sigma^2=0.2$
  <img src="./result/100,153.png" width = 40%>
  观察所有图像可以发现，加了正则项以后，由于正则项对系数矩阵 $W$ 起到了约束的作用，所有即使点数较少阶数较高的情况下依然能保证不出现过拟合的情况。点数多的情况则达到了更好的效果。

#### 五、结论
 - 迭代优化算法
  梯度下降法和共轭梯度法的思想是接近的，都是对 $loss$ 函数进行优化使得函数值尽可能的小。在该问题中，由于 $loss$ 函数在接近最优解时的梯度已经变得十分小，所以梯度下降法的迭代次数达到了数十万次才能接近最优解。并且在点数较少阶数较高时还容易产生过拟合的现象。共轭梯度法利用了共轭向量的思想，将理论迭代次数降到了多项式的阶数。但是由于计算的精度问题，导致了在计算了理论次数后并不能达到理论上的最优解。但是效果依然十分优秀。同时其也拥有梯度下降法的问题。在数据量较小并且阶数较高时容易产生过拟合现象。
 - 解析法
  两种解析法最直接的优势在于他们均可以经过一次计算得到问题的最优解。不需要迭代多次。但是在矩阵较大时，计算过程中的矩阵求逆的精度和计算速度是较大的问题。这也是解析法劣于迭代优化算法的部分。不含正则项的解析法极易产生过拟合现象。添加了正则项以后，不论是数据量的大小，还是阶数的大小，表现出的效果都十分优秀。利用在这个问题中，对 $loss$ 函数添加正则项削弱过拟合现象的方法。在处理回归问题时，可以通过观察系数矩阵等方式，添加正则项，削弱过拟合，达到在更多数据集上获得同样好的效果的目的。

#### 六、附录：源代码
生成数据部分( $makedata.py$ )
```python
import numpy as np
import random

def getSource(num_x, sigma):
    x = np.linspace(0, 1, num_x)
    y = np.sin(2*np.pi*x) + np.random.normal(0, sigma, num_x)#添加噪声
    return x, y

```
网络模型、优化器、$loss$ 函数定义部分( $model.py$ )
```python
import numpy as np

class network(object):
    def __init__(self, args):#初始化
        self.n = args["num_group"]
        self.m = args["order"] + 1
        self.w = np.mat(np.random.rand(self.m, 1))#系数矩阵初始化为随机

    def parameters(self):
        return self.w

    def sync(self, w):
        self.w = w

    def backward(self, gradient):#梯度反向传播
        self.w = self.w + gradient

    def forward(self, x):#矩阵计算直接得到预测值
        return np.dot(x, self.w)


class MSELoss(object):#loss函数
    def __init__(self, y_pre, y_true):
        self.y_pre = np.mat(y_pre)
        self.y_true = np.mat(y_true)
        self.loss = np.dot((self.y_pre-self.y_true).T,
                           (self.y_pre-self.y_true))

    def backward(self, optimizer, net):#调用网络的函数和优化器的函数进行梯度的反向传播
        net.backward(optimizer.grad(self.y_true))


class optim(object):#定义优化器的超类
    def __init__(self, w, x, LR):
        self.w = w
        self.x = x
        self.lr = LR


class GDoptim(optim):#梯度下降法的子类，计算梯度的方法报告已给出
    def __init__(self, w, x, LR):
        super(GDoptim, self).__init__(w, x, LR)

    def grad(self, y_true):
        return - np.dot(self.lr * self.x.T, np.dot(self.x, self.w) - y_true) / (self.x.size/self.w.size)


class CGoptim(optim):#共轭梯度法的子类，确定下一步方向和步长的方法报告中已给出
    def __init__(self, w, x, y_true):
        super(CGoptim, self).__init__(w, x, 1)
        self.y_true = np.dot(x.T, y_true)
        self.x = np.dot(x.T, x)
        self.r = self.y_true - np.dot(self.x, self.w)
        self.p = self.r

    def nextStep(self, net):
        self.a = np.dot(self.r.T, self.r) / \
            np.dot(self.p.T, np.dot(self.x, self.p))
        self.a = np.asarray(self.a)[0][0]
        self.w = self.w + self.a * self.p
        self.rr = self.r
        self.r = self.r - self.a * np.dot(self.x, self.p)
        self.b = np.dot(self.r.T, self.r) / np.dot(self.rr.T, self.rr)
        self.b = np.asarray(self.b)[0][0]
        self.p = self.r + self.b * self.p
        net.sync(self.w)

```
解析解函数定义部分( $functional.py$ )
```python
import numpy as np


class MSE(object):#不带正则项的解析解求解
    def cla_W(x, y_true):
        #计算过程（报告中已给出）
        A = np.dot(x.T, x)
        A = np.linalg.pinv(A)
        return np.dot(A, np.dot(x.T, y_true))


class MSElam(object):#带正则项的解析解求解
    def cla_W(x, y_true, lamda, n, m):
        #计算过程（报告中已给出）
        lamda = lamda * n / 2
        A = np.dot(x.T, x)
        A = lamda * np.identity(m) + A
        B = np.linalg.pinv(np.dot(A.T, A))
        return np.dot(np.dot(B, A.T), np.dot(x.T, y_true))

```
主函数部分( $main.py$ )
```python
import makedata as data
import model as nn
import matplotlib.pyplot as plt
import numpy as np
import functional as F

plt.ion()
plt.show()

#绘图的函数
def view(w, x_rate, y_rate):
    print_x = np.vander(np.linspace(0, 1, 100), order + 1, True)
    print_y = np.dot(print_x, w) * y_rate

    sinx = np.linspace(0, 1, 100)
    sinx = np.sin(2 * np.pi * sinx)
    plt.cla()
    plt.plot(np.linspace(0, 1, 100), print_y.T.A[0], c="r")
    plt.plot(np.linspace(0, 1, 100), sinx, c="g")
    plt.scatter(source_x, source_y, s=10, c="b")
    plt.pause(0.1)


args = dict(
    epoch=1000000,
    num_x=100,
    sigma=0.2,
    num_group=1,
    order=15,
    lamda=0.00001,
    optim="MSElamda",
    LR=0.1
)
epoch = args["epoch"]#训练轮数
order = args["order"]#多项式阶数
LR = args["LR"]#学习率
num_x = args["num_x"]#数据的组数
sigma = args["sigma"]#噪声的方差
group = num_x#一次喂到网络中的数据的组数
lamda = args["lamda"]#正则项的系数

source_x, source_y = data.getSource(num_x, sigma)
#归一化
input = source_x / max(source_x)
input = np.vander(input, order + 1, True)
input = np.mat(input)
y_true = source_y / max(max(source_y), -min(source_y))
y_true = np.mat(y_true).T

net = nn.network(args)

#梯度下降
if args["optim"] == "GD":
    for i in range(epoch):
        output = net.forward(input)#获得预测值
        optimizer = nn.GDoptim(net.parameters(), input, LR)#定义优化器为梯度下降法
        loss_founction = nn.MSELoss(output, y_true)#定义loss函数
        loss_founction.backward(optimizer, net)#loss反向传播

        if i % 2000 == 0:#每隔2000轮画一次图
            view(net.parameters(), max(source_x),
                 max(max(source_y), -min(source_y)))
            loss = loss_founction.loss
            print(loss.A[0]/group)
#共轭梯度法
if args["optim"] == "CG":
    optimizer = nn.CGoptim(net.parameters(), input, y_true)#定义优化器为共轭梯度法
    for i in range(order):
        output = net.forward(input)#获得预测值
        loss_founction = nn.MSELoss(output, y_true)#定义loss函数
        optimizer.nextStep(net)#优化器确定共轭梯度的方向并且向前走一步
        #每走一步画一次图
        view(net.parameters(), max(source_x),
             max(max(source_y), -min(source_y)))
        loss = loss_founction.loss
        print(loss.A[0]/group)
    plt.pause(2)
#不带正则项的解析解
if args["optim"] == "MSE":
    #直接调用计算解析解的函数得到系数矩阵进行画图
    view(F.MSE.cla_W(input, y_true), max(source_x),
         max(max(source_y), -min(source_y)))
    plt.pause(5)
#带正则项的解析解
if args["optim"] == "MSElamda":
    #直接调用计算解析解的函数得到系数矩阵进行画图
    view(F.MSElam.cla_W(input, y_true, lamda, group, order + 1), max(source_x),
         max(max(source_y), -min(source_y)))
    plt.pause(5)

```