### VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
[TOC]
#### Abstract

- they investigate the effect of the depth of cnn on it's accuracy in image recognition
- they use 3$\times$3 convolution filters and have a significant improvement by pushing the depth of cnn to 16-19 weight layers
- they win the ImageNet Challenge 2014 by using these finding
- they achieved state-of-the-art in other datasets
- they make two best performing ConvNet available

#### Introduction

- why convolutional networks can be popular and made a great achievement.
  - big dataset: ImageNet
  - high-performance computing systems:GPUs
  - large-scale distributed clusters
  - ILSVRC
- there are many attempts to improve Alexnet
  - smaller receptive window size and smaller stride of the first layer
  - train and test image densely and over multiple scales
- they fix the parameter and increase the depth of the ConvNets steadily, which is feasible due to the use of 3$\times$3 convolutional filters.
- they made a great achievement and have released two best-performing models to facilitate further reseach

#### ConvNet Configurations

##### Architecture

-  input size is 224$\times$224 RGB image
-  preprocessing: each pixel subtracting the mean RGB value of the training set
-  a very small receptive field 3$\times$3
-  use 1$\times$1 convolution filters as linear transformation 
-  stride is fixed to 1 pixel
-  use spatial padding to preserved the resolution after convolution
-  some of the conv layers are followed by max pooling layer. 2$\times$2 pixel window with stride 2.
-  conv layer is followed by three fully-connected layers
  - fiist two 4096 channels
  - third one 1000 channels
-  final layer is soft-max layer
-  all hidden layer use ReLU as their activition function
-  didn't use Local Response Normalisation

##### Configurations

-  the width of conv layers starting from 64 to 512
-  ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211007094347993-249482349.png)

![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211007094609693-264845785.png)

##### Discussion

- the receptive field of VGG is $3\times3$ which is smaller than alexnet
- why we use three $3\times 3$ conv. layers instead of a single $7\times 7$ conv. layer
  - three non-linear rectification layers instead of a single one
  - [decrease the number of parameters](https://blog.csdn.net/dandelion_2/article/details/96318986)
- $1\times1$ conv. layers increase the nonlinearity of the decision function without changeing the receptive fields
- the related work
  - samll-size convolution filters,but less deep than VGG(Ciresan)
  - deep ConvNets (Goodfellow)
  - GoogleNet more complex than ours but VGG is outperforming in terms of the single-network classification accuracy

#### Classification Framework

##### Training

- follow Krizhevsky except for sampling
- using mini-batch gradient descent with momentum
- batch size :256
- momentum: 0.9
- regularised by weight decay: L2 penalty multiplier is $5\times10^{-4}$
- dropout regularisation ratio is 0.5
- the initial learning rate is $10^{-2}$ and then decreased by a factor of 10 when the validation stop improving.(3 times, 74 epochs)
- why VGG is faster converge than Alexnet
  - implicit regularisation imposed imposed by greater depth and smaller conv filter sizes
  - pre-initialisation of certain layers
- initialisation of the network weight is important
  - pretrain A and use A's weight(first four convolutional layers and the last three fully) 
  - without decrease learning rate
  - random initialisation is sampled from a normal distribution with the zero mean and $10^{-2}$ variance.
  - the biases is zero
- crop from rescaled traning images randomly (one crop per image per SGD iteration)
- training augment 
  - horizontal flipping 
  - RGB colour shift
- crop size is fiexed to $224\times224$
- two approaches for setting the training scale $S$
  - fix $S$：==note that image content within the sampled crops can still represent multiscale image statistics.==
  - multi-scale $S$​ (256,512): This can also be seen as training set augmentation by scale jittering, where a single model is trained to recognise objects over a wide range of scales. (fine-tuning all layer of a single-scale model with the same configuration)

##### [Testing](https://zhuanlan.zhihu.com/p/52766120)

- $Q$ is the test scale, using several values of $Q$ for each $S$ leads to improved performance

- apply the network densely over rescaled test image.
- fully-connected layers are first converted to convolutional layers(the first FC layer to $7\times7$ conv.layer,the last two FC layers to $1\times1$ conv. layers)
- the resulting fully-convolutionla net is then applied to the while image.
- the result is  a class score map
- convert class score map to a fixed-size vector by using spatially averaged(sum-pooled)
- augment the test set the same as training and compute their average
- pad with zero

##### Implementation Details

- C++ Caffe toolbox
- Multi-CPU training

#### Classification Experiments

##### Dataset

- ILSVRC-2012(1000 classes) training(1.3M) validation(50K) testing (100K)
- top-1:the proportion of incorrectly classified images
- top-5 : the ground-truth category is outside the top-5 predicted categories

##### Single scale evaluation

- LRN have no effect
- depth does help,but it is also important to capture spatial context by using conv. filters with non-trivial($3\times3$) receptive fields
- a deep net with small filters outperforms a shallow net with larger filters
- scale jittering is helpful for capturing multi-scale image statistics

##### Multi-scale Evaluation

- running a model over several rescaled version of a test image, and then average the result
- scale jittering at test time lead to better performance.

##### Multi-crop evaluation

- multiple crops and dense evaluation are complementary
- multiple crops are slightly better than dense evaluation

##### Convnet fusion

- combine the outputs of several models by averaging their soft-max class posteriors.

- the resulting ensemble of 7 networks has 7.3% ILSVRC test error.

![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211007221319485-263734447.png)

  ##### Comparison with the state of the art

- VGG is better than other networks at that time except GoogleNet

#### Confuse

- [x] large-scale distributed clusters

> 大规模分布式集群

- [x] receptive field

> the position where the filter can capture 

- [x] RGB colour shift

> Does this method is the same as Alexnet's PCA  operation over the training images

- [x] isotropically-rescaled training image

> 同比例缩放训练图像，因为考虑到所要检测的对象的 size 有可能是多尺度的，所以把图片缩放到不同的尺度，有可能有助于训练出一个好结果。

- [x] multi-crops

> 对同一张图片多次裁剪

- [x] non-trivial receptive fields

> $3\times3$ and $1\times1$ is trivial

- [x] class score map

> use sum-pooled in each channel to compute the class score vector, which present the score of one class

#### Appendix

##### Localisation

- special case of object detection
- a single object bouding box should be predicted for each of the top-5 classes

##### Localisation ConvNet

- the last fully connected layer predicts the bounding box lacation instead of the class scores
- the bouding box is represented by a 4-D vector storing its center coordinates,with and height
- two choice of the bouding box 
  - shared across all classes(single-class regression,SCR) 4-D
  - class-specific(per-class regression,PCR) 4000-D

