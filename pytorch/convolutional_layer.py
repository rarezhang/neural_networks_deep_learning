#!/usr/bin/python3
# coding=utf-8
"""
How To Define A Convolutional Layer In PyTorch
Use PyTorch nn.Sequential and PyTorch nn.Conv2d to define a convolutional layer in PyTorch
"""


import torch
import torchvision
import torch.nn as nn

# The sequential container object in PyTorch is designed to make it simple to build up a neural network layer by layer.
model = nn.Sequential()

# once defined a sequential container, can start adding layers to the network 
first_conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
"""
- Conv2d: because image data is 2 dimensional (An example of 3D data would be a video with time acting as the third dimension)

- in_channels: needs to be equal to the number of channels in the layer above or in the case of the first layer, the number of channels in the data ( In the case of image data, the most common cases are grayscale images which will have one channel, black, or color images that will have three channels – red, green, and blue)

- out_channels: a matter of preference (Firstly, a larger number of out_channels allows the layer to potentially learn more useful features about the input data, though this is not a hard rule.
Secondly, the size of your CNN is a function of the number of in_channels/out_channels in each layer of your network and the number of layers.
If you have a limited dataset, then you should aim to have a smaller network so that it can extract useful features from the data without overfitting.
Lastly, if you’re finding yourself running out of RAM on training your network, thinning the layers is one of the best ways to solve this problem while still having a useful model, other than getting more RAM.)

- kernel_size: the size of the filter that is run over the images 

- stride: how far the filter is moved after each computation 
(With a kernel size of 3 and a stride of 1, features for each pixel are calculated locally in the context of the pixel itself and every pixel adjacent to it.
If I were to change the kernel_size to 5, then the context would be expanded to include pixels adjacent to the pixels adjacent to the central pixel.
The kernel size can also be given as a tuple of two numbers indicating the height and width of the filter respectively if a square filter is not desired.)

(With a stride of 1 in the first convolutional layer, a computation will be done for every pixel in the image.
With a stride of 2, every second pixel will have computation done on it, and the output data will have a height and width that is half the size of the input data.
The stride argument can also be a tuple if different horizontal and vertical strides are desired.
I would not recommend changing the stride from 1 without a thorough understanding of how this impacts the data moving through the network.)

- padding: how much 0 padding is added to the edges of the dta during computation --> prevents the image shrinking as it moves through the layers
(Without good reason to change this, the padding should be equal to the kernel size minus 1 divided by 2.)
"""

# add it to our sequential container using the add module function, giving it the name Conv1.
model.add_module("Conv1", first_conv_layer)

print(model)
