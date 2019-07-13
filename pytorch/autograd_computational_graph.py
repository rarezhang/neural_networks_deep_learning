#!/usr/bin/python3
# coding=utf-8


"""
Make A Simple PyTorch Autograd Computational Graph
"""

import torch 
print(torch.__version__)

# to create a computational graph need 3 tensors 
grad_tensor_a = torch.randn(3, 3, requires_grad=True) # 3x3 tensor 
print(grad_tensor_a)

grad_tensor_b = torch.randn(3, 3, requires_grad=True) 
grad_tensor_c = torch.randn(3, 3, requires_grad=True) 

# build computational graph 
# matrix multiplication 
grad_tensor_multiplication = torch.mm(grad_tensor_a, grad_tensor_b)
print(grad_tensor_multiplication)


grad_tensor_sum = grad_tensor_multiplication + grad_tensor_c
print(grad_tensor_sum)

print(grad_tensor_sum.grad_fn)

print(grad_tensor_sum.grad_fn.next_functions)


