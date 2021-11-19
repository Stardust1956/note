# Pytorch 学习笔记

[TOC]

 ## Tensor(张量)

Tensor可以被理解为一个pytorch专用的表示多维数组的数据结构，用于保存输入、输出以及模型的参数，和Numpy很像，而且共享存储位置，可以互相转换。

### Tensor初始化的几种常用方法

- 直接由python中的列表转化

```python
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
```

- 和numpy互转

``` python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
np_array = x_np.numpy()
```

- 指定shape和初始化的方式

```pyhon
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
x_ones = torch.ones_like(x_data) # retains the properties of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
```

### Tensor的操作

``` python
#操作一：利用GPU加速
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
#操作二：赋值
tensor = torch.ones(4, 4)
tensor[:,1] = 0
#操作三：连接
t1 = torch.cat([tensor, tensor, tensor], dim=1)#dim=1表示在第一维连接即横向连接
#操作四：运算
tensor.mul(tensor)#tensor * tensor 对应位置乘积的两种表示方法
tensor.matmul(tensor.T)#tensor @ tensor.T 矩阵乘积
#如果需要就地运算，即运算后直接保存在原来的数组中，直接在运算操作后加_，例如tensor.add_(5),不推荐这样做，会覆盖历史，丢失数据
```

## AutoGrad

即利用计算图自动计算在模型中的每个参数相对于误差的梯度

```PYTHON
#例子
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
prediction = model(data) # forward pass
loss = (prediction - labels).sum()
loss.backward() # backward pass
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent
```

- 如果一个变量的被设置为requires_grad = True，那么依赖这个变量计算的其他变量，的这个属性也是True
- 可以将requires_grad设置为False，用于冻结部分参数，用于迁移学习冻结模型，只训练全连接层

```PYT
#例子
from torch import nn, optim
model = torchvision.models.resnet18(pretrained=True)
# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 10)
# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

## Torch.nn

利用torch.nn可以自己定义一个网络模型用于深度学习，先定义要用到的层，然后再forward（）中使用这些层，backward不需要自己定义，由autograd自动计算每个参数的梯度。torch.nn只支持min-batch的输入，他的张量为（nsamples，nchannels，height，width）

```PYTHON
import torch
import torch.nn as nn
import torch.nn.functional as F
#需要继承nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)#输入为1个channel，输出6个channels，卷积核5*5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension 16个卷积核的5*5个参数全部和全连接层连接
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
#显示模型的所有参数
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
#输入一张随机生成的图片
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
#先把梯度清零，因为梯度是累加的，然后调用backward（），就能反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))
#定义损失函数
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)


#反向传播
net.zero_grad()     # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
#更新权重
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    
#优化器的写法
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```

## 完整的训练流程

```python
import torch
import torchvision
import torchvision.transforms as transforms
#第一步:加载,归一化数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#第二步:定义一个卷积神经网络
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
#第三步:定义损失函数以及优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#第四步:训练网络
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
#第五步:测试网络,计算精度
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
#也可以计算每一类的精度
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))
```

## 数据集可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
```

## 保存与加载模型

```python
#保存模型
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
#加载模型
net = Net()
net.load_state_dict(torch.load(PATH))
```

## 利用GPU加速

需要把网络和数据都放进GPU才能调用GPU加速

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
net.to(device)
inputs, labels = data[0].to(device), data[1].to(device)
```

## 为什么要在训练时使用net.train()，评估时用net.eval()

根据所处的模式，它们的行为可能会有所不同。例如，BatchNorm模块在train()期间维护运行平均值和方差，当模型处于evalue模式时，这些平均值和方差不会更新。一般来说， 模型在训练期间应处于训练模式，并且仅切换到评估模式进行推理或评估。

## Tensor Views

张量的视图，共享底层，避免显示的复制，可以高效的重新设置形状，切片，对元素进行操作

有一个参数是-1时，该维度可以由其他维度推出来

## torch.utils.data.Data==L==oader

torch.utils.data.Data==L==oader 注意这个l要大写，否则找不到，不能用torch.utils.data.dataloader



