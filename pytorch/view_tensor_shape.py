#!/usr/bin/python3
# coding=utf-8
"""
How To Use The view Method To Manage Tensor Shape In PyTorch
Use the PyTorch view method to manage Tensor Shape within a Convolutional Neural Network
"""



"""
A common issue that arises when designing any type of neural network is the output tensor of one layer having the wrong shape to act as the input tensor to the next layer.

Sometimes, this issue will raise an error, but in more insidious cases, the error will only be noticeable when one evaluates the performance of their trained model on a test set.

For the most part, careful management of layer arguments will prevent these issues.


However, there are cases where it is necessary to explicitly reshape tensors as they move through the network
"""

import torch.nn as nn

# when the output tensor of a convolutional layer is feeding into a fully connected output layer 

"""
1. assume the input images: 32*32 with 3 color channels 
2. input --> The batches are fed into this one elongated tensor by 3, the number of channels, by 32, the height of the images, and by 32, the width of the images.
3. layter1 --> The first layer takes the input tensor and applies the Conv2d operation to it with a kernel size of 3 and a padding of 1 producing 16 feature maps from the original 3 channels.
"""


class Convolutional(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(Convolutional, self).__init__()

        # layer 1 
        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv1", nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1))
        self.layer1.add_module("Relu1", nn.ReLU(inplace=False))

        # layer 2 
        self.layer2 = nn.Sequential()
        self.layer2.add_module("Conv2", nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1, stride=2))
        self.layer2.add_module("Relu2", nn.ReLU(inplace=False))

        self.fully_connected = nn.Linear(32*16*16, num_classes)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 32*16*16)
        x = self.fully_connected(x)
        return x
