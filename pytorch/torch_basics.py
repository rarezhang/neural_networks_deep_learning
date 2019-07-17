#!/usr/bin/python3
# coding=utf-8
import torch 
import torchvision
import torch.nn as nn 
import numpy as np 
import torchvision.transforms as transforms 


# 1. basic autograd 
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# build a computational graph 
y = w * x + b 

print(y)

# compute gradients 
y.backward()
print(y.backward)

# print out the gradients 
print(x.grad)
print(w.grad)
print(b.grad)

# 2. autograd 
x = torch.randn(10, 3) # shape (10, 3)
y = torch.randn(10, 2) # shape (10, 2)

# a fully connected layer 
linear = nn.Linear(3, 2)
print('w:', linear.weight)
print('b:', linear.bias)

# build loss function and optimizer 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# forward pass 
pred = linear(x)

# compute loss 
loss = criterion(pred, y)
print('loss:', loss.item())


# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# 3. loading data from numpy 
x = np.array([[1,2], [3,4]])

# convert the numpy array to a torch tensor 
y = torch.from_numpy(x)

# tensor to a numpy array 
z = y.numpy()

print(x, y, z)


# 4. input pipline 
# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass
