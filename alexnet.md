## ImageNet Classifification with Deep Convolutional Neural Networks

[TOC]

###  Abstract 

1. They propose a model which can achieved the minimum top-1 and top-5 error rates on the test data of ImageNet at that time. This model is a deep convolutional neural network, which has 60 million parameters and 650000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.
2. To make training faster they use
   - non-sturating neurons 
   - efficient GPU implementatoin of convolutional operation


3. To reduce overfitting they use
   - drop out
   - data augmentation

### Introduction

1. Traditional approaches use machine learning to solve object recognition. To  improve performance we can 
   - collect larger datasets
   - learn more powerful models
   - prevent overfitting
   
2. Because of the variablility of realistic setting, we need much larger training sets.

3. The model need prior knowledge to compensate for the data which the ImageNet lake for.

4. CNN's capacity can be controlled by adjust their's depth and breadth.

5. CNN is easier to train than standard feedforward neural networks with similarly-sized layers. Because CNN has fewer connection and parameters, while their theoretically-best performance is likey to be only slightly worse.

6. The contribution of this paper are:

   - they train a largest cnn on the subsets of imageNet and achived state of the art

   - they wrote a highly-optimized GPU implementtation of 2D convolution and all the other operations inherent in training convolutional neural networks.

   - they use some method to prevent overfitting.

   - Five convolutional and three fully-connected layers is very important. 

### Dataset

1. ImageNet:

- 15 milion images

- 22000 categories.

2. This experiments is conduct in the version of ILSVRC-2010

3. Top-5 error rate is the fraction of  test images for which the correct label is not among the fifive labels considered most probable by the model.

4. down-samp the image to a fixed resolution of 256$\times$256:
   - rescale the image such that the shorter side was of length 256
   - crop out the central 256$\times$256
   - ==subtracting the mean activity over the training set from each pixel.==

### Architecture ✨

##### [ReLU](Rectified Linear Units) Nonlinearity

- before $f(x) = tanh(x)$ or $f(x) = (1 + e^{-x})^{-1}$
- now $f(x) = max(0,x)$

##### Traning on Multiple GPUs

- Put half of the kernels on each GPU
- The GPUs can communicate only in certain layers
- advantage: 
  - less error rate than one GPU with half kernels
  - less training time than one GPU

##### [Local Response Normalization](https://blog.csdn.net/luoluonuoyasuolong/article/details/81750190)

- ReLus don't require input normalization to prevent saturating

- learning will happen if at least some training samples produce a positive input

- The response-normalized activity $b^{i}_{x,y}$ is given by the expression
  $$
  b^{i}_{x,y} = a^{i}_{x,y}/(k+\alpha\sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)}(a^j_{x,y})^2)^{\beta}\\
  k= 2, n=5,\alpha=10^{-4},\beta=0.75
  $$

##### Overlapping Pooling 

- summarize the outputs of neighboring groups of neurons
- s = 2 and z = 3
- less overfit

##### Overall Architecture [code](http://code.google.com/p/cuda-convnet/.)

- Eight layers: the first five are convolutional and the remaining three are fully-conneted  which follow the1000-way softmax. 
- The kernels of the second, fourth, and fififth convolutional layers are connected only to those kernel maps in the previous layer which reside on the same GPU
-  The kernels of the third convolutional layer are connected to all kernel maps in the second layer
- The neurons in the fully connected layers are connected to all neurons in the previous layer
- Response-normalization layers follow the fifirst and second convolutional layers.
- Max-pooling layers, of the kind described in Section 3.4, follow both response-normalization layers as well as the fifth convolutional layer.
- The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer.

### Reducing Overfitting 

##### Data Augmentation

- The first form of data augmentation is consists of generating image translations and horizontal reflections.: extract 224$\times$224 patches randomly from the 256$\times$256 images.(the training examples are highly dependent).

- At test time, the network makes a prediction by extracting five 224 *×* 224 patches (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the predictions made by the network’s softmax layer on the ten patches.

- The second form of data augmentation consists of altering the intensities of the RGB channels in training images.(eliminate the influence of the intensity and color of illumination):
  $$
  I_{xy} = [I^R_{xy},I^G_{xy},I^B_{xy}]^T\\
  I_{xy}^{'}=[I^R_{xy},I^G_{xy},I^B_{xy}]^T +[p_1,p_2,p_3][\alpha_1\lambda_1,\alpha_2\lambda_2,\alpha_3\lambda_3]^T
  $$

  > where $p_i$and $\lambda_i$ are *i*th eigenvector and eigenvalue of the 3 *×* 3 covariance matrix of RGB pixel values, respectively, and $\alpha_i$ is the aforementioned random variable.

##### Dropout

- By set the output of hidden neuron to zero with probability 0.5.
- so every time an input is presented, the neural network samples a different architecture, but all these architectures share weights.
- at test time , multiply their outputs by 0.5

### Details of learning 

- use stochastic gradient descent

- batch seze : 128

- momentum: 0.5

- weight decay: 0.0005

- update rule:

- $$
  v_{i+1} := 0.9\cdot v_i -0.0005\cdot\epsilon\cdot w_i-\epsilon\cdot<\frac{\partial L}{\partial w}\mid w_i>_{D_i}
  \\
  w_{i+1} := w_{i} + v_{i+1}
  $$

  > *i* is the iteration index, *v* is the momentum variable, $\epsilon$ is the learning rate, and $<\frac{\partial L}{\partial w}\mid w_i>_{D_i}$  is the average over the *i*th batch $D_i$ of the derivative of the objective with respect to *w*, evaluated at $w_i$

- initialize: 

  - weights:a zero-mean Gaussian distribution with standard deviation 0.01
  - bias of the second, fourth and fifth convolutional layers as well as the fully-conneted hidden layers, with the constant 1.
  - others 0
  - learning rate: 0.01

- divide the learning rate by 10 when the validation error stop improving

### Results

![image-20211002162103728](C:\Users\Keven\AppData\Roaming\Typora\typora-user-images\image-20211002162103728.png)

![image-20211002162545854](C:\Users\Keven\AppData\Roaming\Typora\typora-user-images\image-20211002162545854.png)

##### Qualitative Evaluations

![image-20211002165741256](C:\Users\Keven\AppData\Roaming\Typora\typora-user-images\image-20211002165741256.png)

- if the Euclidean separation of two images 's  feature activations is small, they are consider to be similar.

### DIscussion

- a large,deep convolutional neural network is capable of achieving record-breaking results on a highly challenging dataset using purely supervised learning.
- Depth is important
- they hope to use very large and deep convolutional nets on video sequences where the temporal structure provide very helpful information that is missing or far less obvious in static images.

### Confuse

- [x] non-saturating neurons

> 1. The value is not squeezed into an interval
> 2. The result of the activation function has no maximum or minimum limit

- [x] label-preserving transformations

> transform the image without changing it's label

- [ ] fully-segmented images
- [ ] kernel maps

> feature map?

- [x] Our network maximizes the multinomial logistic regression objective, which is equivalent to maximizing the average across training cases of the log-probability of the correct label under the prediction distribution.
- [x] [PCA](https://zhuanlan.zhihu.com/p/37777074)
- [x] covariance matrix

> $$
> \hat\Sigma=\frac{1}{m-1}\sum_{j=1}^{m}(x_j-\bar{x})(x_j-\bar{x})^T
> $$

- [x] frequency- and orientation-selective kernels

> The network has learned a variety of frequency- and orientation-selective kernels, as well as various colored blobs.

- [x] auto-encoder

![image-20211003161041692](C:\Users\Keven\AppData\Roaming\Typora\typora-user-images\image-20211003161041692.png)

