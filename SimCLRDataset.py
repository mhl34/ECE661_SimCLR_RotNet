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
from ResNet import ResNet

# Custom Dataset for Self-Supervised Learning
class SimCLRDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
            # transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get two augmented versions of the same image
        img1 = self.transform(self.to_pil_image(self.dataset[index][0]))
        img2 = self.transform(self.to_pil_image(self.dataset[index][0]))

        return img1, img2

    @staticmethod
    def to_pil_image(tensor):
        # Convert torch Tensor to PIL Image
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy().transpose((1, 2, 0))  # Channels last
            tensor = (tensor * 255).astype('uint8')
            return transforms.ToPILImage()(tensor)
        else:
            return tensor