from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class CNN_small(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def call_bn(bn, x):
    return bn(x)

class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

    def forward(self, x):
        return self.out_conv(x)


class CNN(nn.Module):
    def __init__(self,n_outputs=10):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            ConvBrunch(3, 64, 3),
            ConvBrunch(64, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(
            ConvBrunch(64, 128, 3),
            ConvBrunch(128, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3 = nn.Sequential(
            ConvBrunch(128, 196, 3),
            ConvBrunch(196, 196, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(3136, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.global_avg_pool(x)
        x = x.view(-1, 3136)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class NewsNet(nn.Module):
    def __init__(self, weights_matrix, context_size=1000, hidden_size=300, num_classes=7):
        super(NewsNet, self).__init__()
        n_embed, d_embed = weights_matrix.shape
        self.embedding = nn.Embedding(n_embed, d_embed)
        self.embedding.weight.data.copy_(torch.Tensor(weights_matrix))
        self.avgpool=nn.AdaptiveAvgPool1d(16*hidden_size)
        self.fc1 = nn.Linear(16*hidden_size, 4*hidden_size)
        self.bn1=nn.BatchNorm1d(4*hidden_size)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.bn2=nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  
        embed = self.embedding(x) # input (128, 1000)
        embed = embed.detach()    # embed (128, 1000, 300)
        out = embed.view((1, embed.size()[0], -1)) # (1, 128, 300 000)
        out = self.avgpool(out)
        out = out.squeeze(0)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.fc3(out)
        return out

