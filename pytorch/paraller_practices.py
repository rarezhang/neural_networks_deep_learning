#!/usr/bin/python3
# coding=utf-8

import torch
import torch.nn as nn 
import torch.optim as optim 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# basic usage 
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to(device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(device)

    def forward(self, x):
        x = self.relu(self.net1(x.to(device)))
        return self.net2(x.to(device))




model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to(device)
loss_fn(outputs, labels).backward()
optimizer.step()   

print(model)
