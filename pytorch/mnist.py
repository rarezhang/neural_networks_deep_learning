#!/usr/bin/python3
# coding=utf-8
"""
https://github.com/pytorch/examples/tree/master/mnist
"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 

from torchvision import datasets, transforms

# Utils 
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # in_channels: needs to be equal to the number of channels in the layer above or in the case of the first layer, the number of channels in the data --> grayscale images which will have one channel, black, or color images that will have three channels – red, green, and blue

        # out_channels: matter of preference (larger number: learn more useful features; limited dataset: smaller network; limited by RAM)

        # kernel_size: the size of the filter that is run over the images; 
        # e.g., kernel_size=3, stride=1 --> features for each pixel are calculated locally in the context of the pixel itself and every pixel adjacent to it.
        # e.g., kernel_size=5, stride=5 --> the context would be expanded to include pixels adjacent to the pixels adjacent to the central pixel.
        # The kernel size can also be given as a tuple of two numbers indicating the height and width of the filter respectively if a square filter is not desired.
 

        # stride: 1 -- controls the stride for the cross-correlation, how far the filter is moved after each computation 
        

        # use Conv2d because image data is 2 dimensional (3D: video)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)

        # conv2.in_channels = conv1.out_channels = 20
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)

        # fully connected layter 
        self.fc1 = nn.Linear(in_features=4*4*50, out_features=500)
        # fc2.in_features = fc1.out_features = 500
        # fc2.out_features = 10 --> class number  
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        # applies a 2d max pooling over an input signal composed of several input planes 
        # kernel_size: the size of teh window to take a max over 
        # stride: stride of the window 
        # padding: implicit 0 paddinn to be added on both sides 
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
       
        optimizer.zero_grad()
        output = model(data)

        # the negative log likelihood loss 
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


args = {'batch_size': 64, 'test_batch_size': 1000, 'epochs':10, 'lr':0.01, 'momentum':0.5, 'no_cuda':False, 'seed':1, 'log_interval':10}
args = dotdict(args)


use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)

if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")
