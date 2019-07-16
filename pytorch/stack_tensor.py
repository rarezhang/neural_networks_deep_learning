#!/usr/bin/python3
# coding=utf-8
"""
PyTorch Stack: Turn A List Of PyTorch Tensors Into One Tensor
PyTorch Stack - Use the PyTorch Stack operation (torch.stack) to turn a list of PyTorch Tensors into one tensor
"""

import torch 


tensor_one = torch.tensor([[1,2,3],[4,5,6]])

print(tensor_one)

tensor_two = torch.tensor([[7,8,9],[10,11,12]])
tensor_tre = torch.tensor([[13,14,15],[16,17,18]])


tensor_list = [tensor_one, tensor_two, tensor_tre]

print(tensor_list)

stacked_tensor = torch.stack(tensor_list)

print(stacked_tensor)

print(tensor_one.shape, stacked_tensor.shape)
