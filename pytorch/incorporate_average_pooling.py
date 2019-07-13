#!/usr/bin/python3
# coding=utf-8
"""
AvgPool2D: How to Incorporate Average pooling into a PyTorch Neural Network
"""

"""
It is common practice to use either max pooling or average pooling at the end of a neural network but before the output layer in order to reduce the features to a smaller, summarized form.


Max pooling strips away all information of the specified kernel except for the strongest signal.


Average pooling summarizes the signal in the kernel to a single average.


"""

import torch.nn as nn

class Convolutional(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(Convolutional, self).__init__()
        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv1", nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1))
        self.layer1.add_module("BN1", nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True,
            track_running_stats=True))
        self.layer1.add_module("Relu1", nn.ReLU(inplace=False))
        self.layer2 = nn.Sequential()
        self.layer2.add_module("Conv2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2))
        self.layer2.add_module("BN2", nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True,
            track_running_stats=True))
        self.layer2.add_module("Relu2", nn.ReLU(inplace=False))

        """
        To perform average pooling on the output of the second convolutional layer.
        kernel_size -- determines how large of an area the average pooling has taken over
        stride -- equal to the kernel size: ensures that there is no overlap in the output averages (can be specified smaller or larger than the kernel size)
        padding -- the implicit zero padding to be added to the edges of the inputs before calculation (useful if your kernel size does not evenly divide the height and width of the input features)
        """
        self.avg_pool("AvgPool1", nn.AvgPool2D(kernel_size=4, stride=4, padding=0, ceil_mode=False, count_include_pad=False))
        # the input dimensions of the fully connected output layer need to be changed to match average pool as average pool changes the shape of layer2’s outputs.
        self.fully_connected = nn.Linear(32 * 4 * 4, num_classes)



    def forward(self, x):
        y = x.clone()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)  # After the average pool layer is set up, we simply need to add it to our forward method.
        x = x.view-1, 32*4*4() # the input dimension is now 32x4x4 because average pool has reduced the height and width of each feature map to 4.
        x = self.fully_connected(x)
        return x
