# 优化算法

这些东西对于深度学习的基础和优化器调参是有一定帮助的

## 一些概念

- 优化目标
    一般形式为:
    $$
    minimize f(x)  \ subject\ to \ x\in C
    \\ f: \mathbb{R}^n \to \mathbb{R}
    $$
    如果$C  = \mathbb{R}^n$那就是不受限
- 全局最小和局部最小

    全局最小:
    $$x^*： f(x^*) \leq f(x) \ \forall x\in C$$

    局部最小:
    $$x^*: \exists \epsilon, \text{使得}f(x^*) \leq f(x), \forall x: ||x-x^*|| \leq \epsilon$$

    一般来说迭代算法只能找到局部最小（掉到坑里面梯度就变成0了，就没有办法进行gradient descent
- 凸集和凸函数

    这个比较简单，稍微查一下就有了
- 凸函数优化

    如果代价函数$f$是凸的，且限制集合$C$是凸的，那么就是凸优化问题，凸优化问题的局部最小一定是全剧最小

    严格凸优化问题有唯一的全局最小

    很遗憾的是，只有线性回归和Softmax回归是凸优化问题，MLP、CNN、RNN、Attention都是非凸的，因为里头有非线性激活函数。因为凸函数的表达能力是十分有限的。
- 梯度下降

    梯度下降算法
    $$for \ t = 1,\dots, T\\
        x_t = x_{t-1} - \eta \nabla f(x_{t-1})
    $$
    其中$\eta$为学习率 learning rate

- 随机梯度下降

    Stochastic gradient descent：当有$N$个样本的时候，我们计算损失：
    $$
    f(x) = \frac {1}{n} \sum_{i=0}^{n}\mathcal{l}(x)
    $$
    计算这个损失的导数需要太大的代价，我们需要对整个数据集进行遍历，在$N$比较大的时候，是不划算的。

    $$
    x_t = x_{t-1} - \eta_t \nabla l_{t_i}(x_{t-1})\\
    \text{其中} \mathbb{E} \left[\nabla  l_{t_i}(x_{t-1}) = \mathbb{E} \nabla f(x) \right]
    $$

    随机梯度下降肯定是`没有梯度下降稳定`，但是他的速度快，整体来说是划得来的。

    随机梯度下降抽取单个样本来进行梯度计算，但是并不能完全利用硬件的并行性能。

    所以我们抽取部分的样本（一个batch），以此来计算损失，这些样本是对于原来数据集的一个无偏估计（unbiased estimation），这就是所谓的`小批量随机梯度下降(mini-batch Stochastic gradient descent)`,这种方法能够有效地降低方差。

- 动量法 Momentum

    冲量法或者也叫动量法，使用`平滑过后的梯度`对权重进行更新。

    $$
    \mathbf{g}_t = \frac {1}{b} \sum_{i \in I_t} \nabla \mathcal{l}_i(x_{t-1})\\
    \mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t\\
    \mathbf{w}_t = \mathbf{w}_{t-1} - \eta \mathbf{v}_t
    $$

    可以看到$v_t$就是平滑之后的梯度，展开$v_t$我们有:
    $$
    \mathbf{v}_t = \mathbf{g}_t + \beta\mathbf{g}_{t-1} + \beta^2 \mathbf{g}_{t-2} + \beta^3 \mathbf{g}_{t-3} + \dots
    $$

    要看一下过去时间的梯度，进行平滑，使得变动不那么剧烈。

    $\beta$一般取[0.5,0.9,0.95,0.99]，样本比较大的时候也可以取0.999

    在SGD和Adam里面都实现了冲量，可以通过设置Momentum来实现冲量。

- Adam

    优化效果与SGD相比并不是说谁好谁坏，Adam的最大的优点就是它对于学习率并不是特别地敏感，其实它跟SGD效果差不多。

    稍微跟Momentum不同的是，它做了非常多的平滑，因此它对`学习率不敏感`

    $$
    v_t = \beta_1 v_{t-1} + (1-\beta_1)g_t\\
    \text{展开得到:}v_t = (1-\beta_1)(g_t + \beta_1 g_{t-1} + \beta_1^2 g_{t-2}+\beta_1^3 g_{t-3}+ \dots)\\
    $$

    由于:
    $$
    \sum_{i=0}^{\infty} \beta_1^i = \frac {1}{1-\beta_1}
    $$
    所以权重的和为1

    由于$v_0 = 0$而且有$\sum_{i=0}^t \beta_1^t = \frac{1-\beta_1^t}{1-\beta_1}$，所以要对$ v_t $进行修正，修正之后的$v_t$有
    $$
    \hat v_t = \frac{v_t}{1-\beta_1^t}
    $$
    当t比较小的时候，需要修正一下，t比较大的时就没啥卵用

    还要记录一个$s_t$：
    $$
    s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2\\
    \text{修正st：} \hat s_t = \frac{s_t}{1-\beta_2^t}
    $$
    对于梯度的平方也做了一个平滑，存储在$s_t$里头

    然后重新调整梯度为：
    $$
    g_t^{\prime} = \frac {\hat v_t}{ \sqrt{\hat s_t^{\prime}}+\epsilon}
    $$
    这是从adagrad里面引用过来的。

    最后更新$w_t$：
    $$
    w_t = w_{t-1} + \eta g_t^{\prime}
    $$

    通常 $\beta_1 = 0.9, \beta_2=0.999$

- 关于Momentum、AdaGrad、RMSProp、Adadelta、Adam等算法，在本文件夹都有实现。

## 总结

### 梯度下降

- 梯度下降需要计算一整个数据集的损失，然后再进行BP，这显然是不划算的
- 随机梯度下降从数据集中取一个样本，计算损失，进行BP(反向传播),这个样本是整个数据集的无偏估计，所以这个办法是可行的，速度快，但是没有办法利用到目前计算机硬件的并行性能。
- 小批量随机梯度下降就是前面两者的折中，相比于第二点，它降低了采样数据的方差，更加稳定，而且能够利用现在计算机硬件的并行性能，目前基本上都是`小批量随机梯度下降`。
- 我们将batch_size调成1就是随机梯度下降了，调成数据集大小n就是梯度下降了，处于这两个数中间就是小批量，所以目前大多数我们都是使用小批量随机梯度下降。

### 改进方法

我们还可以从更新方式上面进行改进，而不是简单的 $x = x - \eta * \frac {\partial f}{\partial x}$

- Momentum:使用动量法汇总过去的梯度，加速收敛，加强稳定性
- AdaGrad:使用坐标缩放来处理数据分布不均匀的问题（实现高效计算的预处理器）
- RMSProp:通过调整学习率来分离每个坐标的缩放
- Adadelta:使用变化量本身作为未来变化的校准，它没有学习率这个超参数
- Adam:整合上面所有的长处，但是有些缺点，2018年提出的Yogi热补丁进行了一些改进
