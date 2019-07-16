#!/usr/bin/python3
# coding=utf-8
"""
PyTorch Min: Get Minimum Value Of A PyTorch Tensor
PyTorch Min - Use PyTorch's min operation to calculate the min of a PyTorch tensor

how to use PyTorch’s min operation to calculate the minimum of a PyTorch tensor.
"""

import torch 


tensor_min_example = torch.tensor(
[
  [
      [1,-10, 1],
      [2, 2, 2],
      [3, 3, 3]
  ],
  [
      [4, 4, 4],
      [5,50, 5],
      [6, 6, 6]
  ]
]
)


print(tensor_min_example)

tensor_min_value = torch.min(tensor_min_example) # returns the answer as a 0-dimensional tensor with a value of -10 inside of it.

print(tensor_min_value)
print(type(tensor_min_value))


#To get the number 10 from the tensor, we’re going to use PyTorch’s item operation.

print(tensor_min_value.item())
print(type(tensor_min_value.item()))
