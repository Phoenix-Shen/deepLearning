# Dive Into Deep Learning

## 介绍

   [原书地址在此](https://zh-v2.d2l.ai/)

   *Dive Into Deep Learning* 的学习代码以及笔记，原本是在我的强化学习仓库里头的，结果发现太多了，便独立出来。

- 之前更新的内容：

  - 各种normlization
  - class activation mapping

- 2023-04-15更新：
  VIT及其在tiny-imagenet上面的微调

- 2022-12-22更新：

   发现有些东西这书里面没有，我自己又加了一些内容，比如Difffusion model和Transformer的其它完整代码版本。

   以后还有一些新的东西的话，都会加到这里面来。

  - 加入Difffusion model

- 2022-12-25更新：

  - 联邦学习（FL）

## 包含内容

1. 数学基础知识
   - 梯度计算
   - 概率和分布
2. LNN
   - 线性回归
   - Softmax 回归
3. MLP
   - 暂退法
   - 模型创建
   - 多层感知机
   - 数值稳定性
   - 过拟合和欠拟合
   - 权重衰减和 L2 惩罚
4. CNN
   - CNN 的架构
   - 多通道的 CNN
   - LeNet
   - 填充和步幅
   - 池化
5. 现代 CNN
   - AlexNet
   - 批量归一化
   - 层归一化(自己加的)
   - 实例归一化(自己加的)
   - VGG
   - NIN
   - GoogLeNet
   - DesneNet
   - ResNet
   - Class Activation Mapping(卷积网络中的注意力)
6. 优化算法
   - 优化和深度学习
   - 梯度下降
   - 随机梯度下降
   - 动量法
   - AdaGrad
   - RMSProp
   - Adam
7. RNN
   - 序列模型
   - 文本预处理
   - 语言模型和数据集构建
   - 从零开始实现RNN
   - 使用Pytorch构建RNN
   - RNN的梯度计算
8. 现代RNN
   - GRU
   - LSTM
   - 深度RNN
   - 双向RNN
   - 机器翻译和数据集
   - 编码器-解码器架构
   - 序列到序列的翻译
9. Attention 机制(NLP的Attention)
   - 注意力提示
   - Nadaraya-Watson核回归
   - 注意力分数
   - Bahadanau Attention
   - 多头注意力
   - 自注意力和位置编码
   - Transformer(实现了两个版本)
   - VisionTransformer（VIT）
10. 计算机视觉ComputerVision

    - 图像增广
    - 微调finetuning
    - 目标检测和边界框
    - 锚框
    - 多尺度目标检测
    - 目标检测数据集
    - 单发多框检测（SSD）
    - R-CNN(区域卷积神经网络)
    - 语义分割
    - 转置卷积
    - 全卷积网络
    - 风格迁移和图像翻译

## 杂谈&经验

### 问题

- tensorboard+vscode有个bug，关掉vscode tensorboard不会退出，解决办法参考[这个网站](https://blog.csdn.net/Yonggie/article/details/119922972)

### 关于Pytorch，摘自[这里](https://www.bilibili.com/video/BV1xW4y1M7JH/?spm_id_from=333.880.my_history.page.click&vd_source=8a3baf666bc9210627c288b6ec6d567a)

- [fastai](https://www.fast.ai/)
- 学习率可以采用周期性学习率例如one cycle实现巨大的加速

   ```python
   scheduler=torch.optim.lr_scheduler.OneCycleLR()
   ```

- 使用AdamW优化器
- 通常情况下使用硬件允许的最大的batch_size是有助于加速的，但是修改了batch_size我们对应的lr也需要增加，batch_size加倍的时候lr也需要加倍
- Dataloader的num_worker设置为GPU数量的4倍，如果硬件性能很好，pin_memory设置为True会增加数据加载至GPU的速度
- 使用AMP(automatic mixed precision)自动混合精度进行训练，能够训练更快
- 如果模型输入大小不变而且结构固定，可以使用torch.backends.cudnn.benchmark=True,可以自动选择最优的卷积算法，更快
- 分布式训练可以使用DataParallel和DistributedDataParallel，pytorch更推荐后者
- 使用梯度累加，这也是增加batch的一种方法
- 验证阶段使用torch.no_grad()
- 可以使用torch.nn.utils.clip_grad_norm_进行梯度裁剪，对Transformer和resnet很有用
- BatchNorm2d之前的卷积层不需要增加bias，因为BatchNorm2d将数据归一化至标准化分布，使用bias会改变数据分布，降低效率
- 不要频繁使用tensor.cpu(),tensor.cuda()，在转换np.ndarray的时候，我们尽量使用torch.as_tensor(np.ndarray)或者torch.from_numpy(np.ndarray)
- 传输列表类型的参数的时候，为了避免参数被改变，要使用copy或者deepcopy
- 使用默认参数的时候要注意参数在编译的时候就进行初始化了,需要将默认参数设置为None从而避免错误结果。

  ```python
   # 正确的
   def display_time(data=None):
      if data is None:
         data = datetime.now()
      print(data)
   print(display_time.__defaults__)#(是一个定值,)
   # 错误的
   def display_time(data=datetime.now()):
      print(data)
      
   print(display_time.__defaults__)#(None,)
  ```

- 使用logging会帮助debug

### 训练模型时候的一些常见手段

#### Data Augmentation

- 基本上都能在`torchvision.transforms`下面找到，下面介绍几种，核心思想就是防止模型“记图”，防止过拟合，并提升模型泛化性
- Normalize
  我们看到的`(0.485, 0.456, 0.406)`和`(0.229, 0.224, 0.225)`便是imageNet数据集三通道每个通道的mean和std，在这里，要将图像每个通道减去mean除以std来实现归一化
- ToTensor
  十分常见，将[H,W,C]维度转化成[C,H,W]维度
- Resize
  重新缩放大小
- RandomCrop
  随机裁剪
- RandomHorizontalFlip&RandomVerticalFlip
  随机水平、垂直翻转
- ColorJitter
  随机改变图像的亮度、对比度，饱和度和色调
- RandomErasing
  随机擦除图像的内容
- [MixUp&CutMix](https://timm.fast.ai/mixup_cutmix)
  将两张照片融合成一张，具体点进去链接就可以看到例子

#### Prevent Overfitting & stablize the training process

- Weight Decay
  将权重的模长作为损失的一部分，有效防止过拟合，可以在我的[这个IPYNB](./MLP/weightDecay.ipynb)中看到
- Model EMA (exponential moving average)
  是timm库中的，目前的VisionTransformer都集成了这个东西，保持一个running mean，对于$W_t$模型参数来说，running mean为$\bar{W}_t = \gamma W_t + (1-\gamma)\bar{W}_{t-1}$，$\gamma$一般取0.01,0.0001等
- [Learning Rate Scheduler](https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863)
  根据训练的步长（step）自动调整学习率，能让模型收敛的更快，有线性的，余弦的，指数的等等，每种都有自己的好处和坏处
- Loss Function
  - LabelSmoothingCrossEntropy
  - SoftTargetCrossEntropy
  在数据标签有错误的时候，可以减少模型的错误率
- AdamW optimizer
  相比于Adam优化器，他俩的Weight Decay方法不一样

  ```python
  # Ist: Adam weight decay implementation (L2 regularization)
   final_loss = loss + wd *all_weights.pow(2).sum() / 2
  # IInd: AdamW
   w = w - lr* w.grad - lr *wd* w
  ```

- Gradient Clipping
   防止梯度爆炸，在Transformer和一些RNN中很常见

#### Training Acceleration

- Auto Mixed Precision
  `torch.autocast`，我们有时候会看到这种代码，这就是自动转换精度，将某些tensor转换成低精度来实现加速
- Model Distillation
  模型蒸馏，在DeiT(data efficient ViT)中使用到了
- Pin Memory
- CuDNN Benchmark
  `torch.backend.cudnn.benchmark=True`，我们有时候会看到，这个是使用cuDNN库去加速卷积操作
- Deepspeed
  快速分布式训练，还没有富到能分布式训练，暂时放这里

#### Logging

- wandb.ai (wandb means `weight and bias`)
  weight and bias，可以传到网页中，随时查看，炸了就可以发邮件提醒你
- Tensorboard (now integrated in `torch.utils`)
  这个也不错
- Visdom
  这个也还行

## 参考资料

[Dive into deep learning](https://zh-v2.d2l.ai/)

[Pytorch 教程Youtube](https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3)

[Pytorch 官网](https://pytorch.org/)

[Transformer](https://wmathor.com/index.php/archives/1455/)

[papers with code, 这个还是不错的，主流算法都有代码](https://paperswithcode.com/)

[李沐的视频](https://space.bilibili.com/1567748478)

[优化算法-知乎](https://zhuanlan.zhihu.com/p/201139622)

[Pytorch Image Models(timm) 库](https://timm.fast.ai/)
