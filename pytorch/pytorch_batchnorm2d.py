#!/usr/bin/python3
# coding=utf-8
"""
BatchNorm2d: How to use the BatchNorm2d Module in PyTorch
BatchNorm2d - Use the PyTorch BatchNorm2d Module to accelerate Deep Network training by reducing internal covariate shift
"""


"""
can improve the learning rate of a neural network

by minimizing internal covariate shift which is essentially the phenomenon of each layer’s input distribution changing as the parameters of the layer above it change during training.

grants us the freedom to use larger learning rates while not worrying as much about internal covariate shift.

This, in turn, means that our network can be trained slightly faster.
"""

import torch.nn as nn 


class Convolutional(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(Convolutional, self).__init__()

        # layer 1 
        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv1", nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1))
        # batch normalization is applied to both the first and the second layer with minimal modification from default arguments.
        """
        num_features: how many features are in the output of the function above it 
        BatchNorm2d.num_features = Conv2d.out_channels = 16
        eps: a value added to the denominator of the batch normalization calculation --> improve numerical stability (should only be modified with good reason)
        track_running_stats = True --> keep a running estimate of its computed mean and variance during training for use during evaluation of the network
        momentum: determines the reate at which the running estimates are updated (if None: the running estimates will be simple averating)
        affine: set to true indicates the BatchNorm should have learnable affine parameters
        """
        self.layer1.add_module("BN1", nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layer1.add_module("Relu1", nn.ReLU(inplace=False))
        
        
        # layer 2
        self.layer2 = nn.Sequential()
        self.layer2.add_module("Conv2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2))
        self.layer2.add_module("BN2", nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layer2.add_module("Relu2", nn.ReLU(inplace=False))
        self.fully_connected = nn.Linear(32 * 16 * 16, num_classes)



    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fully_connected(x)
        return x
