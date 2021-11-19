## MobileNets：Efficient Convolutional Neural Networks for Mobile Vision Applications

### Abstract

-  They present MobileNets which use depth-wise separable convolutions to build light weight DNN.
- Two simple global hyper- parameters to trade off between latency and accuracy

### Introduction

- Goal: build  very small、low latency  model that can be easily matched to the design requirements for mobile and embedded vision applications

- Two hyper-parameters:
  - width multiplier 
  - resolution multiplier

### Prior Work

- two method to build small network
  - compressing pretrained networks
  - train mall networks directly
- MobileNets use depthwise separable convolutions
- small network
  - Flattened networks
  - factorized network
  - Xception network
  - Squeezenet: bottleneck 
  - structure transform network
  - deep fried convnets
- Compression methods
  - product quantization
  - hashing
  - pruning
  - vector quantization
  - Huffman coding
 - Distillation: use a larger network to teach a smaller network
 - low bit networks

### MobileNet Architecture

#### Depthwise Separable Convolution

- Factorize a standard convolution into a depthwise convolution and pointwise convolution

- each input channel a single filter

- drastically reducing computation and model size

  - $$
    \frac{D_k\cdot D_k\cdot M\cdot D_F \cdot D_F + M\cdot N\cdot D_F\cdot D_F}{D_k\cdot D_k\cdot M\cdot N\cdot D_F\cdot D_F}
    \\
    =\frac{1}{N}+\frac{1}{D_K^2}
    $$

- depthwise convolutions

![](https://pic2.zhimg.com/80/v2-2bdf9cb05d9caf6c968c43610f6b8b95_1440w.jpg)

- pointwise convolutions

![](https://pic4.zhimg.com/v2-7593e8b0c43db44d62f19fec7c8795bb_r.jpg)

- less computation less accuracy

#### Network Structure and Training

- All layers are followed by BN and ReLU except the final fully connected layer
- Down sampling is handled with strided convolution in the depthwise convolutions
- 28 layers
- RMSprop
- less regularization and data augmentation techniques

![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211104170300202-71047324.png)

- Standard convolutional layer vs Depthwise Separable convolutions

![](https://img2020.cnblogs.com/blog/2143936/202111/2143936-20211104170505614-1187022037.png)

Computation and Parameters resource of Mobilenet 

![image-20211104181536378](C:\Users\Keven\AppData\Roaming\Typora\typora-user-images\image-20211104181536378.png)

#### Width Multiplier: Thinner Models

- add the parameter $\alpha$ to thin MobileNet

- width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly $\alpha^2$.

- the new network need to be trained from scratch

- cost is

- $$
  D_k\cdot D_k\cdot \alpha M\cdot D_F \cdot D_F + \alpha M\cdot \alpha N\cdot D_F\cdot D_F
  \\
  \alpha \in (0,1]
  $$

#### Resolution Multiplier: Reduced Representation

- add the parameter $\rho$ to thin MobileNet

- set $\rho$ in input resolution

- cost is:
  $$
  D_k\cdot D_k\cdot \alpha M\cdot \rho D_F \cdot \rho D_F + \alpha M\cdot \alpha N\cdot \rho D_F\cdot \rho D_F
  \\
  \rho \in (0,1]
  $$

![image-20211104184501016](C:\Users\Keven\AppData\Roaming\Typora\typora-user-images\image-20211104184501016.png)

### Experiments 

#### Model Choices

![image-20211104185026718](C:\Users\Keven\AppData\Roaming\Typora\typora-user-images\image-20211104185026718.png)

thinner vs shllower

![image-20211104185242377](picture/image-20211104185242377.png)

#### Model Shrinking Hyperparameters

- as $\alpha$ become smaller the accuracy drops off smoothly

![image-20211104185559436](picture/image-20211104185559436.png)

- as $\rho$ become smaller the accuracy drops off smoothly

  ![image-20211104185742413](picture/image-20211104185742413.png)

- trade off between accuracy and computation. Results are log linear with a jump when models get very small at $\alpha=0.25$

![image-20211104190019331](picture/image-20211104190019331.png)

- trade off between accuracy and number of parameters.

![image-20211104190221169](picture/image-20211104190221169.png)

- faster and better
![image-20211104190330837](picture/image-20211104190330837.png)
- mobilenet vs squeezenet and alexnet
 ![image-20211104190437431](picture/image-20211104190437431.png)

#### Fine Grained Recognition

![image-20211104190759996](picture/image-20211104190759996.png)

#### Large Scale Geolocalization

![image-20211104191149790](picture/image-20211104191149790.png)

#### Face Attributes

- similar mAP and 1% computation

![image-20211104191919325](picture/image-20211104191919325.png)

#### Object Detection

![image-20211104192821006](picture/image-20211104192821006-16360253020461.png)

#### Face Embedding

![image-20211104192938818](picture/image-20211104192938818.png)

### Conclusion

- Proposed MobileNet based on depthwise separable convolutions
- trade off the accuracy and size and latency by two hyper-parameters $\alpha$ and $\beta$



### Confuse

- [x] Depth wise separable convolution

