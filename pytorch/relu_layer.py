#!/usr/bin/python3
# coding=utf-8
"""
How To Define A ReLU Layer In PyTorch
Use PyTorch's nn.ReLU and add_module operations to define a ReLU layer
"""

"""
how to define the activator layers that we will place between our convolutional layers
"""



"""
Two issues that can arise when optimizing a neural network are second order effects in activation functions and saturation of an activated unit.


Second order effects cause issues because linear functions are more easily optimized than their non-linear counterparts.


Saturation occurs when two conditions are satisfied: One, the activator function is asymptotically flat and two, the absolute value of the input to the unit is large enough to cause the output of the activator function to fall in the asymptotically flat region.

Since the gradient in the flat region is close to zero, it is unlikely that training via stochastic gradient descent will continue to update the parameters of the function in an appropriate way.

This often arises when using tanh and sigmoid activation functions.

A popular unit that avoids these two issues is the rectified linear unit or ReLU.

We use the activation function g(z) = max(0, z).

These units are linear almost everywhere which means they do not have second order effects and their derivative is 1 anywhere that the unit is activated.
"""


import torch.nn as nn
import random

"""
ReLU function: applied element-wise 
no need to specify input or output dimensions 
argument 'inplace' --> determines how the function treats the input 
    inplace = True --> the input is replaced by the output in memory (reduce memory usage, but may not be valid for the particular use case, )
    inplace = False --> both the input and the ouput are stored separately in memory
"""


# first, we will define the sequential container.
model = nn.Sequential()

# layer 1
# define our first convolutional layer 
first_conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
# use add module to add the first convolutional layer
model.add_module("Conv1", first_conv_layer)
# define the activator layer --> place between 2 convolutional layers 
relu1 = nn.ReLU(inplace=False)
# add module again to add our first rectified linear unit layer.
model.add_module("Relu1", relu1)


# layer 2
# Alternatively, we can use add module and define the layer in the same line.
model.add_module("Conv2", nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]))
# second ReLU layer
model.add_module("Relu2", nn.ReLU(inplace=False))

print(model)
