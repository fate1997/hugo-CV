---
title: G-SphereNet笔记
date: 2022-04-06 14:10:15
categories: 分子生成 (Molecule Generation)
mathjax: true
---

### An Autoregressive Flow Model for 3D Molecular Geometry Generation from Scratch

[[Paper]](https://openreview.net/forum?id=C03Ajc-NS5W)  [[Github]](https://github.com/divelab/DIG)

[](1.png)

<!-- more -->

- **目标**：从头生成3D分子结构

- **现有模型**

  1. 生成2D分子
  2. 通过已有的2D分子生成3D结构

- **模型概要**

  1. 模型一部分来自于SphereNet
  2. 序列地放置原子
  3. 分子的3D坐标隐式地通过距离、角度和二面角体现
  4. 通过SphereNet和注意力机制提取条件信息

- **相关工作**

  1. 通过序列模型生成SMILES
  2. 通过图生成模型去得到原子类型、邻接矩阵
  3. 3D分子从头生成，这个任务可以被分为两个任务，一个是从3D分子结构中学习到一个随机生成模型使得其能够生成有效的3D分子结构；另一个是学习到一个目标分子发现模型使得量子性质分数最大化
  4. G-SchNet使用自回归模型序列地生成新的原子，并放置到网格点上
  5. EDMNet和3DMolNet分别通过GAN和VAE生成两个原子间的距离
  6. E-NFs将flow模型和E(n)不变图神经网络结合，生成所有原子的one-hot坐标并且定义了在隐空间的子空间的先验概率去保证平移不变性

- [**流模型**](https://lilianweng.github.io/posts/2018-10-13-flow-models/)

  1. 直接学习p(x)
  2. 给定一个随机变量$z$满足$z \sim \pi(z)$，建立一个新的随机变量$x$，满足$x=f(z)$其中$f$是可逆的有，因此$z=f^{-1}(x)$，那么目前的问题就变成了如何推断这个未知概率密度的$x$ 可以看到，因为两个随机变量都满足归一化，对$x$求导数后把$z=f^{-1}(x)$代入，就可以得到$p(x)$的表达式了$$\begin{aligned}
     &\int p(x) d x=\int \pi(z) d z=1 ; \text { Definition of probability distribution. } \\
     &p(x)=\pi(z) \frac{d z}{d x}=\pi\left(f^{-1}(x)\right) \frac{d f^{-1}}{d x}=\pi\left(f^{-1}(x)\right)\left|\left(f^{-1}\right)^{\prime}(x)\right|
     \end{aligned}$$
  3. Normalizing Flows：用可逆函数把已知分布的$z_{i-1}$映射到$z_i$，也就是上面那个式子，为了简单起见，对等式两边取$\log$函数$$\log p_{i}\left(\mathbf{z}_{i}\right)=\log p_{i-1}\left(\mathbf{z}_{i-1}\right)-\log \operatorname{det} \frac{d f_{i}}{d \mathbf{z}_{i-1}}$$因此第$K$个变量的概率密度可写为：$$\begin{aligned}
     \mathbf{x}=\mathbf{z}_{K} &=f_{K} \circ f_{K-1} \circ \cdots \circ f_{1}\left(\mathbf{z}_{0}\right) \\
     \log p(\mathbf{x})=\log \pi_{K}\left(\mathbf{z}_{K}\right) &=\log \pi_{K-1}\left(\mathbf{z}_{K-1}\right)-\log \operatorname{det} \frac{d f_{K}}{d \mathbf{z}_{K-1}} \\
     &=\log \pi_{K-2}\left(\mathbf{z}_{K-2}\right)-\log \operatorname{det} \frac{d f_{K-1}}{d \mathbf{z}_{K-2}}-\log \operatorname{det} \frac{d f_{K}}{d \mathbf{z}_{K-1}} \\
     &=\ldots \\
     &=\log \pi_{0}\left(\mathbf{z}_{0}\right)-\sum_{i=1}^{K} \log \operatorname{det} \frac{d f_{i}}{d \mathbf{z}_{i-1}}
     \end{aligned}$$这样就可以得到一个新的概率分布，但是需要注意的是转换函数$f_i$应满足两个性质，一是它的可逆比较好算，二是它的雅克比行列式比较好算。Normalizing Flows的损失函数可以通过下式进行计算：$$\mathcal{L}(\mathcal{D})=-\frac{1}{|\mathcal{D}|} \sum_{\mathbf{x} \in \mathcal{D}} \log p(\mathbf{x})$$
  4. 根据这个$f$的选择，目前有几个模型可供选择：

  1) RealNVP：使用bijection(或者叫affine coupling layer)，这种映射关系是一对一的，也就是输入与输出是成对存在的。这个模型采用的bijection是：$$\begin{aligned}
     \mathbf{y}_{1: d} &=\mathbf{x}_{1: d} \\
     \mathbf{y}_{d+1: D} &=\mathbf{x}_{d+1: D} \odot \exp \left(s\left(\mathbf{x}_{1: d}\right)\right)+t\left(\mathbf{x}_{1: d}\right)
     \end{aligned}$$这里$s(.)$和$t(.)$分别是放大和平移函数。这个映射的雅克比行列式是一个下三角矩阵$$\operatorname{det}(\mathbf{J})=\prod_{j=1}^{D-d} \exp \left(s\left(\mathbf{x}_{1: d}\right)\right)_{j}=\exp \left(\sum_{j=1}^{D-d} s\left(\mathbf{x}_{1: d}\right)_{j}\right)$$因为不用计算$s(.)$和$t(.)$的雅克比行列式，因此他们都可以是神经网络。由于在一个仿射层中一些维度是保持不变的，因此为了让所有的输入都有机会改变，我们可以在每次应用放射层时翻转输入。对于大规模的输入，可以在仿射层中采用采样的方法，具体可见[paper](https://arxiv.org/abs/1605.08803)
  2) [Glow](https://arxiv.org/abs/1807.03039)：拓展了ReakNVP模型，总共有三个部分，第一个部分是Activation normalization，和BN很像，但是只在1个batch下进行计算，将每个channel的输出归一化后使用新的放大和偏置参数去训练；第二个部分是可逆的1x1卷积，它的雅克比行列式可写为$$\log \operatorname{det} \frac{\partial \operatorname{conv} 2 \mathrm{~d}(\mathbf{h} ; \mathbf{W})}{\partial \mathbf{h}}=\log \left(|\operatorname{det} \mathbf{W}|^{h \cdot w} \mid\right)=h \cdot w \cdot \log |\operatorname{det} \mathbf{W}|$$其中$h$和$w$分别是张量的高和宽，$\mathbf{h};\mathbf{W}$分别是输入张量和参数矩阵；第三个部分就是Bijection层

  5. 自回归流模型：自回归是为了处理序列数据，就是在给定之前的状态预测下一时刻的状态，即$p(x_i|x_{1:i-1})$。如果使用一个自回归模型作为flow的变换时这个模型就叫做自回归流模型，下面将先介绍几个自回归模型MADE,PixelRNN，WaveNet，然后再讲几个自回归流模型MAF和IAF

  1) [MADE](https://arxiv.org/abs/1502.03509)：把Autoencoder里面的权重矩阵乘以一个binary-masked的矩阵，从而使得输出只考虑之前的信息
  2) [PixalRNN](https://arxiv.org/abs/1601.06759)：针对图像的深度生成模型，一次产生一个像素，在生成当前像素点时，	模型会看到之前生成的像素点。使用对角的BiLSTM去获取之前的Context，但是没办法并行了
  3) [MAF](https://arxiv.org/abs/1705.07057)：生成数据的概率是已知概率分布的仿射变换
  4) [IAF](https://arxiv.org/abs/1606.04934)：与MAF相反，使用相逆的仿射函数

- **方法**

  1. 使用点云作为输入，也就是$G=(A,R)$其中$A$是原子类型矩阵，$R$是原子坐标矩阵
  2. 将分子生成任务作为一个序列决策任务，在第$i$个步骤，使用之前的点云信息得到隐变量，根据这个隐变量生成分子$$a_{i}=g^{a}\left(z_{i}^{a} ; A_{i}, R_{i}\right), r_{i}=g^{r}\left(z_{i}^{r} ; A_{i}, R_{i}\right), \quad i \geq 1$$
  3. 通过自回归流进行分子生成，首先需要把原子类型转为连续变量$$\tilde{a}_{i}=a_{i}+u, u \sim U(0,1)^{k}, \quad i \geq 1$$为了生成$\tilde{a}_{i}$，首先从标准正态分布中取值得到一个隐变量$z_i$，然后通过映射函数$$\tilde{a}_{i}=s_{i}^{a} \odot z_{i}^{a}+t_{i}^{a}, \quad i \geq 1$$其中缩放因子和偏置都是根据之前生成的原子得到的条件信息
  4. 与分子类型同理，可以生成距离，角度和二面角
  5. 首先通过G-SphereNet得到每个原子更新后的表达，然后经过一个MLP分类器，如果分类的分数比0.5大，那就放在待选列表里，如果待选列表为空，表明不需要生成分子了，否则应随机地从待选列表里选择一个原子
  6. 作者表示如果直接进行atom-wise FFNN的话生成的新原子的位置不靠谱，因此使用了多头自注意力机制使focal atom可以得到其他原子的信息
  7. 特别地，对于空间的上下文信息，作者将其与原子类型的embedding相乘

- **训练**

  1. 使用Prim算法使得采样的focal atom总是离最新的原子最近
  2. 损失函数如下：$$\begin{aligned}
     \log p(G) &=\sum_{i=1}^{n-1}\left[\log p_{Z_{a}}\left(z_{i}^{a}\right)+\log \left|\frac{\partial \tilde{a}_{i}}{\partial z_{i}^{a}}\right|\right]+\sum_{i=1}^{n-1}\left[\log p_{Z_{d}}\left(z_{i}^{d}\right)+\log \left|\frac{\partial d_{i}}{\partial z_{i}^{d}}\right|\right] \\
     &+\sum_{i=2}^{n-1}\left[\log p_{Z_{\theta}}\left(z_{i}^{\theta}\right)+\log \left|\frac{\partial \theta_{i}}{\partial z_{i}^{\theta}}\right|\right]+\sum_{i=3}^{n-1}\left[\log p_{Z_{\varphi}}\left(z_{i}^{\varphi}\right)+\log \left|\frac{\partial \varphi_{i}}{\partial z_{i}^{\varphi}}\right|\right]
     \end{aligned}$$其中$z_i$可以通过映射的反函数求得

- **实验**

  1. 随机3D分子几何生成，使用QM9数据集。3D分子几何通过[Kim](https://onlinelibrary.wiley.com/doi/abs/10.1002/bkcs.10334)提出的方法变为3D分子，使用有效性作为评价指标，有效性是指所有不违反化学价规则的分子占所有生成分子的比例。通过键长分布的MMD距离，把距离与常见的化学键距离做比较，从而确定生成的到底是什么键(好麻烦，不如直接用化学键生成)，有效性才88%
  2. 目标分子发现，用了两个任务，一个是最小化HOMO-LUMO gap，另一个是最大化各向异构极化性。将QM9里面小的HOMO-LUMO和大的各向异构极化性的分子拿出来，然后去做分子生成。评价的指标是看生成的分子中比QM9里最小的HOMO-LUMO以及最大的各向异构极化性分子还大的分子的比例
  3. 消融实验：local feature < global feature；without 3D information < with 3D information；focal atom从50%里面选择 > 直接采用softmax后最大比例的分子

- **Limitations**

  1. 原子类型有没有都考虑
   2. 化学键的选择对于生成的位置坐标的精度要求有点高
   3. 由点云->分子图的方法是否合理
   4. 反向设计是否不太现实

- **Development**

  1. 不要用点云，保留键的信息，从而避免从点云->分子图这个过程
  2. 反向设计可以使用动态规划、强化学习等序贯决策过程进行设计
  3. 需要考虑多个原子类型
  4. 综上所述，context encoder应满足几个条件

  1) 能够处理不同原子类型
  2) 使用角度、距离、二面角生成分子
  3) 不使用点云而使用分子结构
  4) 为了生成合适的键，需要考虑原子的杂化类型
