## Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

[TOC]

### Abstract

- what is internal covariate shift? and how to address it?
  - the change in the distributions of network activations due to the change in network parameters during training. 
  - normalize layer input
- Batch Normalization: perform normalization for each training mini-batch
- Advantages
  - make training faster 
  - no sensitive to initialization

### Introduction

- Advantage of mini-batch
  - the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases.
  - more efficient in computation than one sample

- the small change to the network parameters amplify as the network becomes deeper
- the distribution of x remain fixed overtime make training more efficient

- the distribution of x change will arise gradient vanish and slow down the convergence.
- the ways to address saturated problem are:
  - ReLU
  - careful initialization
  - small learning rate
- Batch Normalization
  - fix the means and variances of layer input
  - reduce the need for Dropout

### Towards Reducing Internal Covariate Shift

- the network training converges faster if its inputs are whitened

- consider whitening activation at every training step or at some interval

  - may reduce the effect of the gradient step

    

- the previous approaches for normalization is expensive, which motivate then to seek an alternative that performs input normalization in a way that is differentiable and does not require the analysis of the entire training set after every parameter update

### Normalization via Mini-Batch Statistics

- They make two necessary simplification to make whiten efficient

  1. make the feature have the mean of zero and the variance of 1 instead of whitening the features in layer inputs and outputs jointly.

  $$
  \hat{x}^{(k)}=\frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}
  $$

  ​			E and Var is from training data set.

  - to address the represent problem, they introduce $\gamma^{(k)}$ and $\beta^{(k)}$ which can be learned along with the model
    $$
    y^{(k)}=\gamma^{(k)}\hat{x}^{(k)}+\beta^{(k)}
    $$
    

  2. use mini-batch to estimate the whole training set

     ![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211103100427878-1784659313.png)

- BN transform is a differentiable transformation 

#### Training and Inference with Batch-Normalized Networks

![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211103102402959-2061063484.png)

During inference, the normalization is simply as a linear transform 

#### Batch-Normalized Convolutional Networks

- add the BN transform before the acitivate function. such as

  - $$
    z = g(Wu+b)
    $$

  - normalizing x = Wu+b because it is likely to have stable distribution 

  - so the normalized result is $z=g(BN(Wu))$ 

    > BN transform is applied independently to each dimension of x , with a separate pair of learned parameters $\gamma^{(k)}、\beta^{(k)}$ per dimension

- For convolutional layers learn a pair of parameters $\gamma^{(k)}、 \beta^{(k)}$per feature map rather than per feature map. During inference the BN transform applied the same linear transformation to each activation in a given feature map.

#### Batch Normalization enables higher learning rates

- with BN , back-propagation through a layer is unaffected by the scale of its parameters. and BN will stablilize the parameter growth.

#### Batch Normalization regularizes the model

- mini-batch is advantageous to the generalizaiton of the network
- Dropout in a batch-normalized network can be removed or reduced in strength.

### Experiments

#### Activation over time

![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211103112621512-723054763.png)

BN make the distribution of activation more stable and train faster

#### Image-Net classification

![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211103171116062-217533371.png)

Applied Batch Normalization to a new variant of GoogLeNet

- $5\times5$ convolutional replaced by two consecutive layers of $3\times3$convolutions with up to 128 filters.
- $13.6\times10^{6}$ parameters
- no fully-connected layers
- SGD with momentum
- mini-batch size :32
- a single crop per image
- 

#### Accelerating BN Networks

- Increase learning rate
- Remove Dropout
- Reduce the L2 weight regularization (reduced by a factor of 5)
- Accelerate the learning rate decay (deacy exponentially)
- Remove LRN
- Shuffle training examples more thoroughly(enable within-shard)
- Reduce the photometric distortion
- learning rate:0.0015

#### Single-Network Classification

![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211103145704584-1236873034.png)



![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211103145833462-874512264.png)

After BN, the model can train faster and can use sigmoid and big learning rate which is unsuited in original inception model without BN. 

#### Ensemble Classification

- result: 4.82 test error exceeds the estimated accuracy of human raters 
- six model with dropout 5% or 10%
- per-activation BN in last hidden layers of the model
- prediction based on the arithmetic average of class probabilities predicted by the constituent networks
- ![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211103152033584-1266169761.png)

### Conclusion

- their motivation is to solve covariate shit which is the hindrance of traing model
- they perform the normalization for each mini-batch which make the model can be trained with saturating nonlinearities and more tolerant to increased training rates , and ofen don't require Dropout.
- they modify the model and achieve state-of-the-art in classification
- compare to standardization which perform standardization to the output of the nonlinearity

- they will investigate the RNN with BN

- they will investigate whether BN can help with domain adaptation



### Confuse

- [x] within-shard shuffling

> 更彻底的打扰数据集

- [x] network's capacity

> 有关capacity的解释：实际上BN可以看作是在原模型上加入的“新操作”，这个新操作很大可能会改变某层原来的输入。当然也可能不改变，不改变的时候就是“还原原来输入”。如此一来，既可以改变同时也可以保持原输入，那么模型的容纳能力（capacity）就提升了。

- [x] scale and shift ($\gamma 、 \beta$)

> 操作则是为了让因训练所需而“刻意”加入的BN能够有可能还原最初的输入

- [x] Internal Covariate Shift

> 统计机器学习中的一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如，transfer learning/domain adaptation等。而covariate shift就是分布不一致假设之下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，对所有![[公式]](https://www.zhihu.com/equation?tex=x%5Cin+%5Cmathcal%7BX%7D),![[公式]](https://www.zhihu.com/equation?tex=P_s%28Y%7CX%3Dx%29%3DP_t%28Y%7CX%3Dx%29)，但是![[公式]](https://www.zhihu.com/equation?tex=P_s%28X%29%5Cne+P_t%28X%29).

