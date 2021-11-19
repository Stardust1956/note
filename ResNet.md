### [Deep Residual Learning for Image Recognition](https://zhuanlan.zhihu.com/p/54072011?utm_source=com.tencent.tim&utm_medium=social&utm_oi=41268663025664)

 [TOC]

#### Abstract

- Deeper neural networks are more difficult to train.
- They present a residual learning framework to ease the training of deep networks.

- ==They explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions==
- They prove these residual networks are easier to optimize, and can improve accuracy
- They win the 1st place in ILSVRC2015&COCO 2015.
  - ensemble of these residual nets achieves 3.57%
  - 28% improvement on the COCO object detection dataset

#### Introduction

- Deep networks is a end-to-end networks. The deeper layers contain the enricher features.

- The depth of models is of crucial importance (This is proved in VGG's paper)

- Is learning better networks as easy as stacking more layers?
  - vanishing/exploding gradients hamper convergence
  
- How to solve vanishing/exploding gradients?
  - normalized initialization
  - intermediate normalization layers

- Degradation problem: the accuracy gets saturated and even degrades rapidly with the depth of network increasing.

  - ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211010211324500-1039180260.png)
  - Not all systems are similarly easy to optimize

- a solution to construct deeper model

  - learned shallow model followed by identity mapping: the error should be no higher than shallower counterpart. But they can't find better solution with current solvers.

- They present a deep residual learning framework to address the degradation problem

  - ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211010214101948-425986562.png)
  - $H(x)\rightarrow F(x)+x$

  > $F(x) = H(x) - x$

  - they hypothesize it is easier to optimize than the original one when the optimal mapping is identity mapping
  
- How to realize $F(x)+x$

  - feedforward neural networks with shortcut connections
  - shortcut connections perform identity mapping
    - advantage:
      - neither extra parameter nor computational complexity.
      - can be trained end-to-end by SGD with back propagation

- After comprehensive experiments they find that:

  - deep residual nets are easy to optimize than plain nets
  - deep residual nets can easily enjoy accuracy gains from greatly increased depth.

- the model's generalization ability is good

- The residual learning principle is generic

##### Related Work

- ==Residual Representations.==: a good reformulation or preconditioning can simplify the optimization
- Shortcut Connections
  - The differences between highway networks and shortcut connections
    - parameter-free
    - never closed

#### Deep Residual Learning :star2:

##### Residual Learning ✨

- $H(x)$ is an underlying mapping

- $x$ is the inputs to the stack layers, and it's dimensions is equivalent to $H(x)$

- multiple nonlinear layers can asymptotically approximate complicated functions

- the ease of learning between $F(x)+x = H(x) - x+x$ and $H(x)$ is different

- residual may precondition the problem

- The dimensions of x and $F$ must be equal. If not, can perform a linear projection $W_{s}$ by the shortcut connections to match the dimensions:

  > $W_s 就是一个用于维度匹配的矩阵$

  $$
  y = F(x,\{W_i\})+W_{s}x
  $$

- $W_{s}$ is only used when matching dimensions

- if $F$ has only a single layer, $y = W_1x+x$,which have not observed advantages

- residual function is applicable to convolutional layers.

##### Network Architectures

###### Plain Network. 

- $3\times3$ filters

- two rules:
  - for the same output feature map size the layers have the same number of filters
  - if the feature map size is halved , the number of filters is doubled so as to preserve the time complexity per layer.
  
- use convolutional layers with a stride of 2 to perform downsampling.

- the network ends with a global average pooling layer and a 1000-way fully connected layer with softmax.

- ==their model has fewer filters and lower complexity than VGG?== 18%

  > the two 4096 fully connected layer is too time consuming

![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211014153151966-1368328964.png)

###### Residual Network

- insert shortcut to plain network

- if the dimension of input and output are same:

  - directly used

- else

  - two option:
    - extra zero entries padded for increasing dimensions
    - projection use $y = F(x,\{W_i\})+W_{s}x$
  - for both options,when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.
##### Implementation

- scale augmentation 
- $224\times224$  randomly crop sampled from an image or it's horizontal flip
- per-pixel mean subtracted (Alexnet)
- standard color augmentation(Alexnet)
- batch normalization right after each convolution and before activation.  
- initialize the weights as in [[13]](K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
  Surpassing human-level performance on imagenet classification. In
  ICCV, 2015.[)
- SGD with mini-batch size of 256
- learning rate start from 0.1 and is divided by 10 when error plateaus
- iterations $60\times10^4$
- weight decay 0.0001
- momentum 0.9
- no dropout
- in testing use
  - 10-crop testing
  - average the scores at multiple scales {224,256,384,480,640}

#### Experiments

##### ImageNet Classification		

![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211014214505291-2044569665.png)

|| Plain Networks | Residual Networks |
|-| -------------- | ---- |
|18-layer|  |convergence faster than plain Networks|
|34-layer| higher validation error and training  error | lower validation error and training  error |

- the degration problem is well addressed by Residual learning and can gain accuracy
- ResNet ease the optimization by providing faster convergence at the early stage
![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211014214436524-120717732.png)

###### Identity VS Projection Shortcuts

| zero-padding               | projection shortcuts     | all shortcuts are projections    |
| -------------------------- | ------------------------ | -------------------------------- |
| better than plain networks | better than zero-padding | better than projection shortcuts |

The 3 options's differences is smal,  in consider of memory/time complexity and model size , finally use identity shortcuts 

###### Deeper Bottleneck Architectures

![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211015103153893-889807853.png)

- $1\times1$ layers are responsible for reducing and then increasing dimensions
- identity shortcuts is more efficient than projeciton

###### Comparisons with sota method

- combine six models of different depth to form an ensemble lead to 3.57% top-5 error on the test set
- won 1st place in ILSVR 2015
- ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211015104939143-381635199.png)

##### CIFAR-10 and Analysis

- input image $32\times32$ with per-pixel mean subtracted
- architecture
  - ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211015110007202-1774038532.png)
- weight decay 0.0001
- momentum 0.9
- weight initialization and BN and no drop out
- minibatch size : 128
- learning rate 0.1 divide it by 10 at 32k and 48k iterations
- 45k/5k train/val split
- data augmentation : 4 pixels are padded on each side and $32\times32$ randomly crop sample from the padded image or its horizontal flip.
- the result is similar to Imagenet

###### Analysis of Layer Responses

- ResNets have generally smaller responses than their plain counterparts
- the deeper ResNet has smaller magnitudes

###### Exploring Over 1000 layers

- still fairly good but worse than 110-layer network
- reason : overfitting

##### Object Dectecion on PASCAL and MS COCO

![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211015112155765-1938501804.png)

#### Conclusion

- Deep Residual Networks:
  - easy to train
  - simply gain accuracy from depth
  - well transferrable

#### Confuse

- [x] [vanishing/exploding gradients](https://zhuanlan.zhihu.com/p/68579467)

>  because of  back propagation, the loss is multiply by the Derivative of active function many times. so when the derivative > 1, produce exploding of gradients, when the derivative <1, produce vanishing gradients

- [x] degradation problem

> with the increase of depth the training error become higher because the solver can't find solves in feasible time

- [x] Batch Normalization

> BN is added before active function and after conv. or fully connected layers. to make the distribution of value more important.
>
> ![](https://pic3.zhimg.com/80/v2-083ca0bcd0749fd0f236a690b50442e6_1440w.png)

- [x] center layer response
- [x] zero mapping
- [x] 10-crop testing 

> the method is from Alexnet . four corners  and center and horizontally flip

- [x] non-zero variances
- [x] bottleneck

> ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211015154748067-290781554.png)
>
> bottleneck means firstly use $1\times1$ filters to decrease the dimensions ,and then use $1\times1$ filters to fit the dimensions to reduce compute complexity.

- [x] avg pool

> compute the mean of every feature map

- [x] why the residual learning is efficient？

> - Adaptive depth: After using the residual structure, it is easier to fit identity mapping. Therefore, when the network does not need to be so deep, the identity mapping in the middle can be a little more, and vice versa.
> - Differential amplifier: when the optimal H(x) is close to identity mapping ,it's easier to find small fluctuations than plain structure **“差分放大器”：**假设最优 ![[公式]](https://www.zhihu.com/equation?tex=H%28x%29) 更接近恒等映射，那么网络更容易发现除恒等映射之外微小的波动
> - model assemble
> - avoid gradient vanish

#### New words

| words      | translations |
| ---------- | ------------ |
| omit       | 省略了的     |
| philosophy | 思想体系     |
| plateaus   | 停滞时期     |
| conjecture | 推测         |
| bottleneck | 瓶颈         |
| magnitude  | 数值         |



  

  