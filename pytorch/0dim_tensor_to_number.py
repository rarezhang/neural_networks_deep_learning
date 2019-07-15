#!/usr/bin/python3
# coding=utf-8


"""
PyTorch item: Convert A 0-dim PyTorch Tensor To A Python Number
PyTorch item - Use PyTorch's item operation to convert a 0-dim PyTorch Tensor to a Python number

use PyTorch’s item operation to convert a zero-dimensional PyTorch tensor to a Python number.
"""


import torch 

zero_dim_example_tensor = torch.tensor(10)

# zero_dim_example_tensor = torch.tensor([10])
'''
Notice that we are not putting the 10 inside of brackets.
If we had, we would then be creating a one-dimensional tensor.
However, since we are creating a zero-dimensional example tensor, we did not use any bracket.
'''

print(zero_dim_example_tensor)
print(type(zero_dim_example_tensor))
print(zero_dim_example_tensor.shape)  # it is an empty list which signifies that it has zero dimensions.

# convert the zero-dimensional PyTorch tensor to a Python number
converted_python_number = zero_dim_example_tensor.item()

print(converted_python_number)
print(type(converted_python_number))
