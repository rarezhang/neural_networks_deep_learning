#!/usr/bin/python3
# coding=utf-8

"""
PyTorch Tensor To List: Convert a PyTorch Tensor To A Python List
PyTorch Tensor To List: Use PyTorch tolist() to convert a PyTorch Tensor into a Python list
"""

import torch 

# create a pytorch tensor 
pytorch_tensor = torch.tensor(
[
  [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
      [0.7, 0.8, 0.9]
  ],
  [
      [1.1, 1.2, 1.3],
      [1.4, 1.5, 1.6],
      [1.7, 1.8, 1.9]
  ]
]
)

print(pytorch_tensor)
print(type(pytorch_tensor))

# use the PyTorch tolist operation to convert our example PyTorch tensor to a Python list.
python_list_from_pytorch_tensor = pytorch_tensor.tolist()
print(python_list_from_pytorch_tensor)
print(type(python_list_from_pytorch_tensor))

#  check to make sure that the numbers are still floating point numbers 
print(type(python_list_from_pytorch_tensor[0][1][2]))
