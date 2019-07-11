#!/usr/bin/python3
# coding=utf-8


import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 

from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # in_channels: 1
        # out_channels: 2
        # kernel_size: 5
        # stride: 1 -- controls the stride for the cross-correlation
        # use Conv2d because image data is 2 dimensional (3D: video)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
