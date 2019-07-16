#!/usr/bin/python3
# coding=utf-8
"""
PyTorch Matrix Multiplication: How To Do A PyTorch Dot Product
PyTorch Matrix Multiplication - Use torch.mm to do a PyTorch Dot Product
"""

import torch

tensor_example_one = torch.Tensor(
[
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3]
])


print(tensor_example_one)

tensor_example_two = torch.Tensor(
[
    [4, 5, 6],
    [4, 5, 6],
    [4, 5, 6]
])

print(tensor_example_two)

tensor_dot_product = torch.mm(tensor_example_one, tensor_example_two)

print(tensor_dot_product)
