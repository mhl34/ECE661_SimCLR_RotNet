import argparse
import os, sys
import time
import datetime
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import time
import cv2
from PIL import Image

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
from ContrastiveLoss import ContrastiveLoss
from SimCLRDataset import SimCLRDataset
from hyperParams import hyperParams
from SimCLRModel import SimCLRModel
import matplotlib.pyplot as plt
from utils import extract_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize SimCLR Model
hyperparams = hyperParams()
base_encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False).to(device)
simclr_model = SimCLRModel(base_encoder, hyperparams.projection_dim)

# Data Augmentation
data_transform = transforms.Compose([
    # transforms.Resize(224),
    # transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
    # transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
    transforms.ToTensor()
])

# Initialize DataLoader and SimCLRDataset
dataset = datasets.CIFAR10(root='./data', download=True, train=True, transform=data_transform)
simclr_dataset = SimCLRDataset(dataset)
indices = list(range(len(dataset)))
subset_size = hyperparams.batch_size * 20

# store the best loss
best_loss = float('inf')

# Shuffle the indices randomly
torch.manual_seed(42)  # Set a seed for reproducibility
shuffled_indices = torch.randperm(len(indices))

# Take a random subset of indices
subset_indices = indices[:subset_size]

# Use SubsetRandomSampler to create a DataLoader with the random subset
subset_sampler = SubsetRandomSampler(subset_indices)
dataloader = DataLoader(dataset=simclr_dataset, batch_size=hyperparams.batch_size, sampler=subset_sampler)

criterion = ContrastiveLoss(hyperparams.temperature)
optimizer = optim.Adam(simclr_model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)

# Training Loop
for epoch in range(hyperparams.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}', unit='batch')

    # Loop over batches
    for batch_idx, (img1, img2) in progress_bar:
        # Forward pass
        h1, z1 = simclr_model(img1.to(device))
        h2, z2 = simclr_model(img2.to(device))

        # Compute contrastive loss
        loss = criterion(z1, z2)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # file.write(f"{loss.item()}\n")

    if loss.item() < best_loss:
        print("Saving ...")
        # Save the learned representation for downstream tasks
        state = {'state_dict': simclr_model.state_dict(),
                 'epoch': epoch,
                 'lr': hyperparams.learning_rate}
        torch.save(state, f'simclr_encoder{hyperparams.batch_size}_{hyperparams.epochs}.pth')
        best_loss = loss.item()
        
    print(f'Epoch [{epoch+1}/{hyperparams.epochs}], Train Loss: {loss.item():.4f}')
