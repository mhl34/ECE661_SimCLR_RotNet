import argparse
import os, sys
import time
import datetime
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import time

import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchsummary import summary
import numpy as np

# define the ResNet mode;
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # first convolution
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16,
                               kernel_size = 3, stride = 1, padding = 1)

        self.batchNorm1 = nn.BatchNorm2d(16)

        # layer 1: 2 layers
        self.layer1a = layerBlock(in_channels = 16, out_channels = 16,
                                  kernel_size = 3, stride = 1, padding = 1)

        self.layer1b = layerBlock(in_channels = 16, out_channels = 16,
                                  kernel_size = 3, stride = 1, padding = 1)

        # layer 2: 2 layers
        self.layer2a = layerBlock(in_channels = 16, out_channels = 32,
                                  kernel_size = 3, stride = 2, padding = 1)

        self.layer2b = layerBlock(in_channels = 32, out_channels = 32,
                                  kernel_size = 3, stride = 1, padding = 1)

        # layer 3: 2 layers
        self.layer3a = layerBlock(in_channels = 32, out_channels = 64,
                                  kernel_size = 3, stride = 2, padding = 1)

        self.layer3b = layerBlock(in_channels = 64, out_channels = 64,
                                  kernel_size = 3, stride = 1, padding = 1)

        self.pooling = nn.AvgPool2d(8)

        self.fullyConnect   = nn.Linear(in_features = 64,
                               out_features = 10)

    def forward(self, x):
        out = F.relu(self.batchNorm1(self.conv1(x)))
        out = self.layer1a(out)
        out = self.layer1b(out)
        out = self.layer2a(out)
        out = self.layer2b(out)
        out = self.layer3a(out)
        out = self.layer3b(out)
        out = self.pooling(out)
        out = self.fullyConnect(torch.reshape(out,(out.size(0),-1)))
        return out

    def swish(self, x):
        return x * torch.sigmoid(x)

class layerBlock(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super(layerBlock, self).__init__()

        self.stride = stride

        # layer 1
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                               kernel_size = kernel_size, stride = stride, padding = padding)

        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                               kernel_size = kernel_size, stride = 1, padding = padding)

         # layer 2
        self.conv3 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                               kernel_size = kernel_size, stride = 1, padding = padding)

        self.conv4 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                               kernel_size = kernel_size, stride = 1, padding = padding)

        # layer 3
        self.conv5 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                               kernel_size = kernel_size, stride = 1, padding = padding)

        self.conv6 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                               kernel_size = kernel_size, stride = 1, padding = padding)

        # shortcuts
        self.shortcut1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                               kernel_size = 1, stride = stride, padding = 0)
        self.shortcut2 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                               kernel_size = 1, stride = stride, padding = 0)
        self.shortcut3 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                               kernel_size = 1, stride = stride, padding = 0)

        # layer 1
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        self.batchNorm3 = nn.BatchNorm2d(out_channels)

        # layer 2
        self.batchNorm4 = nn.BatchNorm2d(out_channels)
        self.batchNorm5 = nn.BatchNorm2d(out_channels)
        self.batchNorm6 = nn.BatchNorm2d(out_channels)

        # layer 3
        self.batchNorm7 = nn.BatchNorm2d(out_channels)
        self.batchNorm8 = nn.BatchNorm2d(out_channels)
        self.batchNorm9 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        # layer 1
        out = F.relu(self.batchNorm1(self.conv1(x)))
        out = F.relu(self.batchNorm2(self.conv2(out)))
        sc1 = self.batchNorm3(self.shortcut1(x)) if self.stride != 1 else x
        out = F.relu(out + sc1)

        # layer 2
        out = F.relu(self.batchNorm4(self.conv3(out)))
        out = F.relu(self.batchNorm5(self.conv4(out)))
        sc2 = self.batchNorm6(self.shortcut1(x)) if self.stride != 1 else x
        out = F.relu(out + sc2)

        # layer 3
        out = F.relu(self.batchNorm7(self.conv5(out)))
        out = F.relu(self.batchNorm8(self.conv6(out)))
        sc3 = self.batchNorm9(self.shortcut1(x)) if self.stride != 1 else x
        out = F.relu(out + sc3)
        return out

    def swish(self, x):
        return x * torch.sigmoid(x)