#!/usr/bin/python3
# coding=utf-8

"""
How To Define A Sequential Neural Network Container In PyTorch
Use PyTorch's nn.Sequential and add_module operations to define a sequential neural network container
"""

"""
Once our data has been imported and pre-processed, the next step is to build the neural network that we'll be training and testing using the data.

Though our ultimate goal is to use a more complex model to process the data, such as a residual neural network, we will start with a simple convolutional neural network or CNN.
"""

import torch 
import torch.nn as nn 
from collections import OrderedDict


"""
For the convolutional neural network, first we will need to define a container.
Containers can be defined as sequential, module list, module dictionary, parameter list, or parameter dictionary.
The sequential, module list, and module dictionary containers are the highest level containers and can be thought of as neural networks with no layers added in.
"""

# The sequential container can be defined as
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

'''
- OrderdDict: though it is not necessary, it is good practice to use OrderedDict when creating a sequential model as well as passing a name string as a parameter when using add_module as this one will make your model much, much easier to inspect and debug.

'''


# can also add layers to the end --> using add_module
model.add_module('conv3', nn.Conv2d(64, 64, 5))

