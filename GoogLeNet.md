## Going Deeper with convolutions

[TOC]

### Abstract

- They propose a new deep convolutional neural network architecture codenamed Inception
- win ILSVRC14 and achieve state of the art
- improve utilization of the computing resources inside the network
- increasing the depth and width of the network while keeping the computational budget constant
- The architectural decisions are based on the Hebbian principle and the intuition of muti-sacle processing

### Introduction

- GoogLeNet uses 12 $\times$ fewer parameters than AlexNet while being more accurate
- The biggest gains in object-detection have not come from the utilization of deep networks alone or bigger models, but from the synergy of deep architectures and classical computer vision ,like the R-CNN algorithm
- Because of the popular of mobile and embedded computing , the efficiency of the algorithm is becoming more and more important. So we need to considerate the efficiency rather than just consider the accuracy numbers.
- The small computational budget make it can be used to real  world.
- two "deep" meaning 
  - a new level of organization in the form of the "Inception module" 
  - increased network depth
- we can view the Inception model as a logical culmination of [12] while taking inspiration and guidance from the theoretical work by Arora.

### Related Work

- Inspired by a neuroscience model of the primate visual cortex

> 受灵长类视觉皮层的神经模型启发

- max-pooling layers can result in loss of accurate spatial information

- Inception layers are repeated many times, and all filters in the inception model are learned
- Network-in-Network is an approach in order to increase the representational power of neural networks. When applied to convolutional layers, the method could be viewed as additional $1\times1$ convolutional layers followed typically by the rectified linear activation.
- Network-in-Network can be easily integrated in the current CNN pipelines, and they use this approach heavily in this architecture.

- two purpose of $1\times1$ convolutions
  - as dimension reduction modules to remove computational bottlenecks and break the limit of network's size
  - increasing the depth and the width of networks without significant performance penalty
- RCNN is a leading approach for object detection. It has two steps.
  1. utilize low-level cues such as color and superpixel consistency for potential object proposals in a category-agnostic fashion
  2. use CNN classifiers to identify object categories at those location.
- They adopt similar pipeline in the detection submission, but have explored enhancements in both stages, such as multi-box prediction for higher object bounding box recall, and ensemble approaches for better categorization of bounding box proposals.

### Motivation and High Level Considerations

- There are two ways to improve the performance of DNN. 
  - increase the depth of DNN,
  -  increase the width of the layer.
- But these way has two major drawbacks. 
  - overfitting need high quality training sets which is tricky and expensive
  - High computational resources
- The fundamental  way of solving both issues is moving from fully connected to sparsely connected architectures, even inside the convolutions.
- if the probability distribution of the data-set is representable by a large, very sparse deep neural network, then the optimal network topology can be constructed layer by layer by analyzing the correlation statics of the activations of the last layer and clustering neurons with highly corrected outputs.
- Buy the sparse data structures is inefficient on todays computing infrasturctures
- whether the proposed architecture's quality can be attributed to the guiding principles

### Architectural Details

-  The main idea of the Inception architecture is based on finding out how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components.
- In order to avoid patch-alignment issues, filter sizes in Inception architecture are restricted to $1\times1 and 3\times3 and 5\times5$
- The pooling is useful in cnn , so they add pooling path in Inception architecture.
- As features of higher abstraction are captured by higher layers, so the ratio of $3\times3$ and $5\times5$ convolutions should increase as we move to higher layers
- But even a modest number of $5\times5$ filters is expensive, so they add $1\times1$ before the $3\times3$ and $5\times5$ convolutions to reduce computation, which have another purpose that can include the use of rectified linear activation.  
- ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211027163431794-1075645174.png)

- lower layer in traditional convolutional fashion and higher layer in Inception module
- Advantage of Inception module
  - allow increasing the number of units significantly without an uncontrolled blow-up in computational complexity. 
  - visual information should be processed at various scales and then aggregated so that the next stage can abstract features from different scales simultaneously

### GooLeNet（an homage to LeNet）:star:

![image-20211027194517545](C:\Users\Keven\AppData\Roaming\Typora\typora-user-images\image-20211027194517545.png)

- ALL the convolution use Relu
- receptive field is $224\times224$
- RGB color channels with mean subtraction
- ALL the reduction/projection layers use rectified linear actiation
- In table 1. "#3$\times$3" and "#5$\times$5" stands for the number of 1$\times$1 filters in the reduction layer used before the 3$\times$3 and 5$\times$5 convolutions.
-  And "**pool proj**" stand for the number of 1$\times$1 filters after the max-pooling layer.
- The network is efficient so it can be run on individual devices including even those with limited computational resources, especially with low-memory footprint.
- 22 layers deep with parameters
- average pooling before the classifier, better than fully connected layers
- extra linear layer enables adapting and fine-tuning the network for other label sets easily.
- remain dropout even after removing the fully connected layers
- How to propagate gradients back through all the layers in an effective manner?
  - add auxiliary classifier connected to the intermediate layers
  - put on top of the output of the Inception4a and 4d modules.
  - during training, their loss gets added to the total loss of the network with a discount weight----0.3, and at inference time , these auxiliary networks are discarded.
  - ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211028093528635-911476840.png)
  - ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211028093408393-433758449.png)

- ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211028093447244-1784928760.png)

- exact structure of the extra network on the side, including the auxiliary classifier, is as follows:
  - An average pooling layer with $5\times5$ filter size and stride 3, resulting in an $4\times4\times512$ output for the (4a) and $4\times4\times528$ for 4d.
  - A $1\times1$ convolution with 128 filters for dimension reduction and rectified linear activation
  - A fully connected layer with 1024 units and rectified linear activation
  - A dropout layer with 70% ratio of dropped outputs
  - A linear layer with softmax loss as the classifier (predicting the same 1000 classes as the main classifier, but removed at inference time)

### Training Methodology

- They use DistBelief distributed machine learning system using modest amount of model and data-parallelism.
- CPU training but could be trained to convergence within a week.
- SGD with 0.9 momentum
- fixed learning rate schedule(decreasing the learning rate by 4% every 8 epochs)

- Polyak averaging was used to create the final model used at inference time.
- some of the model were trained on smaller relative crops others on larger ones.
- sampling of various sized patches of the image (8% and 100% of the image area and aspect ratio is chosen randomly between 3/4 and 4/3)
- photometric distortions are useful to combat overfitting
- they don't know the exact effect of random interpolation methods

### ILSVRC 2014 Classification Challenge Setup and Results

- adopt a set of techniques during testing to obtain a higher performance
  - ensemble 7 versions of the same GoogLeNet model which only differ in sampling methodologies and the random order in which they see input images
  - cropping approach
    - 4 scales where the shorter dimension  is 256,288,320 and 352.
    - take the left, center and right square 
    - portrait image: top, center and bottom squares
    - For each square, take the 4 corner and the center 224$\times$224 crop as well as the square resized to $224\times224$ and their mirrored versions
    - The total numbers of crops is $4\times3\times6\times2 = 144$ per image 
  - averaged over multiple crops and over all the individual classifiers to obtain the final prediction
- Results

![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211028114142688-2025798437.png)

- vary the number of models and the number of crops have different result.
- when use one model, they choose the one with the lowest top-1 error rate on the validation data 

### ILSVRC 2014 Detection Challenge Setup and Results

- Detection task is to produce bounding boxes around objects in images among 200 possible classes.

- Detected objects count as correct if they match the class of the groundtruth and their bounding boxes overlap by at least 50%.

- The image in detection task may have 0 object.

- Result are reported using the mean average precision (mAP)

- The method is similar to RCNN, but is augmented with the Inception model as the region classifier. And combing the Selective Search approach with multi-box predictions for higher object bounding box recall.

- The superpixel size was increased by 2$\times$ to cut down the number of false positives. By doing this and combing the selective search algorithm the proposal become half. And then, they added back 200 region proposals coming from multi-box resulting, in total, in about 60% of the proposals used by RCNN.

- 1% improvement of mAP for single model

- improve the accuracy from 40% to 43.9% by using an ensemble of 6 ConvNets.

- no bounding box regression

- ![](https://img2020.cnblogs.com/blog/2143936/202110/2143936-20211029085823887-411026812.png) 

  

### Conclusions

- They prove approximate the expected optimal sparse structure by readily available dense building blocks ia s viable method for improving neural networks for CV.
- The main advantage of this method is a significant quality gain at a modest increase of computational requirements compared to shallower and less wide networks.
- The Inception architecture is powerful
- This suggest promising future work toward creating sparser and more refined structures in automated ways on the bias of []. 

### Questions

- [ ] Hebbian principle

> neurons that fire together, wire together。

- [ ] the intuition of multi-scale processing
- [ ] contract normalization
- [ ] super consistency 
- [ ] Polyak averaging 
- [ ] jaccard index
- [ ] SeLective Search
- [ ] superpixel size
- [ ] mAP

> **mAP定义及相关概念**
>
> - mAP: mean Average Precision, 即各类别AP的平均值
> - AP: PR曲线下面积，后文会详细讲解
> - PR曲线: Precision-Recall曲线
> - Precision: TP / (TP + FP)
> - Recall: TP / (TP + FN)
> - TP: IoU>0.5的检测框数量（同一Ground Truth只计算一次）
> - FP: IoU<=0.5的检测框，或者是检测到同一个GT的多余检测框的数量
> - FN: 没有检测到的GT的数量
>
> 


## [资料](https://zhuanlan.zhihu.com/p/32702031)

#### 1$\times$ 1 卷积模块的作用

> 1. 在相同尺寸的感受野中叠加更多的卷积，能提取到更丰富的特征。这个观点来自于Network in Network, Inception 结构里的 三个1$\times$ 1 卷积都起到了该作用。
> 2. 降维，降低计算复杂度

#### Network in network 和传统神经网络的区别

> NIN 结构是在同一个尺度上的多层计算，从而在相同的感受野范围能提取更强的非线性。传统神经网络是跨越了不同尺寸的感受野，从而在更高尺度上提取出特征。

#### 多个尺寸上卷积再聚合

> - 在直观感觉上在多个尺度上同时进行卷积，能提取到不同尺度的特征。特征更为丰富也意味着最后分类判断时更加准确。
>
> - 利用稀疏矩阵分解成密集矩阵计算的原理来加快收敛速度。举个例子下图左侧是个稀疏矩阵（很多元素都为0，不均匀分布在矩阵中），和一个2x2的矩阵进行卷积，需要对稀疏矩阵中的每一个元素进行计算；如果像右图那样把稀疏矩阵分解成2个子密集矩阵，再和2x2矩阵进行卷积，稀疏矩阵中0较多的区域就可以不用计算，计算量就大大降低。**这个原理应用到inception上就是要在特征维度上进行分解！**传统的卷积层的输入数据只和一种尺度（比如3x3）的卷积核进行卷积，输出固定维度（比如256个特征）的数据，所有256个输出特征基本上是均匀分布在3x3尺度范围上，这可以理解成输出了一个稀疏分布的特征集；而inception模块在多个尺度上提取特征（比如1x1，3x3，5x5），输出的256个特征就不再是均匀分布，而是相关性强的特征聚集在一起（比如1x1的的96个特征聚集在一起，3x3的96个特征聚集在一起，5x5的64个特征聚集在一起），这可以理解成多个密集分布的子特征集。这样的特征集中因为相关性较强的特征聚集在了一起，不相关的非关键特征就被弱化，同样是输出256个特征，inception方法输出的特征“冗余”的信息较少。用这样的“纯”的特征集层层传递最后作为反向计算的输入，自然收敛的速度更快。
>
>    ![]( https://pic4.zhimg.com/v2-eacad5957624f2b0dec823af256817cf_r.jpg)
>
> - Hebbin赫布原理。Hebbin原理是神经科学上的一个理论，解释了在学习的过程中脑中的神经元所发生的变化，用一句话概括就是*fire togethter, wire together*。赫布认为“两个神经元或者神经元系统，如果总是同时兴奋，就会形成一种‘组合’，其中一个神经元的兴奋会促进另一个的兴奋”。比如狗看到肉会流口水，反复刺激后，脑中识别肉的神经元会和掌管唾液分泌的神经元会相互促进，“缠绕”在一起，以后再看到肉就会更快流出口水。用在inception结构中就是要把相关性强的特征汇聚到一起。这有点类似上面的解释2，把1x1，3x3，5x5的特征分开。因为训练收敛的最终目的就是要提取出独立的特征，所以预先把相关性强的特征汇聚，就能起到加速收敛的作用。
>
> - 在inception模块中有一个分支使用了max pooling，作者认为pooling也能起到提取特征的作用，所以也加入模块中。注意这个pooling的stride=1，pooling后没有减少数据的尺寸。



