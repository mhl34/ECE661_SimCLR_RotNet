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
from hyperParams import hyperParams
import matplotlib.pyplot as plt
from RotNetDataset import RotatedCIFAR10
from RotNetLoss import RotNetLoss
from RotNet import RotNet

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create RotatedCIFAR10 dataset
root = "./data"  # Change this to your desired data directory
rotated_cifar_dataset = RotatedCIFAR10(root=root, transform=transform)

# Create DataLoader
hyperparams = hyperParams()
torch.manual_seed(42)  # Set a seed for reproducibility

indices = list(range(len(rotated_cifar_dataset)))
subset_size = hyperparams.batch_size * 20
shuffled_indices = torch.randperm(len(indices))
subset_indices = indices[:subset_size]
subset_sampler = SubsetRandomSampler(subset_indices)

cifar_dataloader = DataLoader(rotated_cifar_dataset, batch_size=hyperparams.batch_size, sampler=subset_sampler)
device = hyperparams.device

# Example usage
base_encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False).to(device)
base_encoder.fullyConnect = nn.Identity()  # Remove the final classification layer
rotnet_model = RotNet(base_encoder, hyperparams.num_classes)
rotnet_model.to(device)

torch.nn.utils.clip_grad_norm_(base_encoder.parameters(), max_norm=1.0) 

criterion = RotNetLoss()
optimizer = optim.Adam(rotnet_model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)

best_loss = float('inf')

rotnet_model.train()

file = open("trainLosses.txt", "w")

# Example: forward pass
for epoch in range(hyperparams.epochs):
    progress_bar = tqdm(enumerate(cifar_dataloader), total=len(cifar_dataloader), desc=f'Epoch {epoch+1}', unit='batch')
    
    if epoch in [30, 60, 80]:
        print(f"updating lr ... ")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 1/5

    # Loop over batches
    for batch_idx, batch in progress_bar:
        rotated_image, rotated_label = batch
        rotated_image = rotated_image.to(device)
        rotated_image_norm = F.normalize(rotated_image)
        rotated_label = rotated_label.to(device)
        
        rotnet_output = rotnet_model(rotated_image_norm)
        rotnet_output = rotnet_output.to(device)
        
        loss = criterion(rotnet_output, rotated_label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        file.write(f"{loss.item()}\n")
        
    if loss.item() < best_loss:
        print("Saving ...")
        # Save the learned representation for downstream tasks
        state = {'state_dict': rotnet_model.state_dict(),
                 'epoch': epoch,
                 'lr': hyperparams.learning_rate}
        torch.save(state, f'rotnet_encoder{hyperparams.batch_size}.pth')
        best_loss = loss.item()
        
    print(f'Epoch [{epoch+1}/{hyperparams.epochs}], Train Loss: {loss.item():.4f}')
        
