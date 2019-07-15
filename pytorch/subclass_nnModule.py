#!/usr/bin/python3
# coding=utf-8

"""
How to Subclass The nn.Module Class in PyTorch

Construct A Custom PyTorch Model by creating your own custom PyTorch module by subclassing the PyTorch nn.Module class

"""


"""
The recommended method of constructing a custom model in PyTorch is to defind your own subclass of the PyTorch module class.
"""

import torch.nn as nn 
from collections import OrderedDict

class Convolutional(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(Convolutional, self).__init__()
        """
        The first sequential container defines the first layer of our convolutional neural network. In this case, a 2D convolutional layer and a ReLU layer are being considered as one layer
        """
        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv1", nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1))
        self.layer1.add_module("Relu1", nn.ReLU(inplace=False))
        """
        can also define our second layer in an equivalent but slightly cleaner way by passing an Ordered Dictionary from the collections library to the sequential container when we initialize.
        """
        self.layer2 = nn.Sequential(OrderedDict(
            [('Conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)),
             ('Relu2', nn.ReLU(inplace=False))]
            ))

    def forward(self, x):
        """
        define the forward method 
        The forward method takes a single argument, x, which starts as simply the input data.

We then define what happens to x as it passes through our network.

In our case, we simply apply a layer1 and layer2 to x.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        return x


"""
This network is still not fully functional as it requires a reshaping step, an output layer, and could do with some max pooling in the convolutional layers.

However, defining our network in this way makes these steps much easier to add.    
"""


