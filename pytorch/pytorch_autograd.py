#!/usr/bin/python3
# coding=utf-8
"""
PyTorch Autograd: Define A PyTorch Tensor With Autograd
PyTorch Autograd - Use PyTorch's requires_grad to define a PyTorch Tensor with Autograd
"""


import torch 


# create a Python variable non_grad_tensor
# use torch.rand to create a PyTorch tensor that is 2x3x4.
non_grad_tensor = torch.rand(2, 3, 4)


# created a PyTorch autograd tensor
yes_grad_tensor = torch.rand(2, 3, 4, requires_grad=True)


print(non_grad_tensor)

print(yes_grad_tensor)
