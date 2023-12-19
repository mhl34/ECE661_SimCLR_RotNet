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
from utils import extract_features
from RotNetDataset import RotatedCIFAR10
from RotNetLoss import RotNetLoss
from RotNet import RotNet


# Data Augmentation
data_transform = transforms.Compose([
    # transforms.Resize(224),
    # transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
    # transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
    transforms.ToTensor()
])

hyperparams = hyperParams()
root = "./data"  # Change this to your desired data directory
dataset = RotatedCIFAR10(root=root, transform=data_transform)
indices = list(range(len(dataset)))
device = 'cuda' if torch.cuda.is_available else 'cpu'

subset_labeled_size = 640
subset_labeled_indices = indices[:subset_labeled_size]
subset_labeled_sampler = SubsetRandomSampler(subset_labeled_indices)
labeled_dataloader = DataLoader(dataset=dataset, batch_size=hyperparams.batch_size, sampler=subset_labeled_sampler)

print("Extracting Features...")
rotnet_encoder = RotNet(torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True).to(device), hyperparams.num_classes)
rotnet_encoder.load_state_dict(torch.load('rotnet_encoder256.pth', map_location=torch.device('cuda'))['state_dict'])
rotnet_encoder.to(device)
rotnet_encoder.rotation_head = nn.Identity()
features, labels = extract_features(labeled_dataloader, rotnet_encoder)

# Step 3: Linear classifier
# Train a linear classifier (logistic regression in this example)
linear_classifier = nn.Linear(features.size(1), hyperparams.num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(linear_classifier.parameters(), lr=0.01)

print("Training Linear Classifier...")
# Train the linear classifier
num_epochs = 1000
for epoch in tqdm(range(num_epochs)):
    outputs = linear_classifier(features)
    targets = F.one_hot(labels, hyperparams.num_classes).float().to(device)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# Fine-Tuning RotNet
print("Finetuning RotNet...")
finetune_data_transform = transforms.Compose([
    # transforms.Resize(224),
    # transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
finetune_dataset = dataset

# Define the labeling fraction for fine-tuning
labeling_fraction = 0.1  # Example: 10% of the dataset is labeled
if labeling_fraction > 0:
    # Split dataset into labeled and unlabeled subsets
    num_total_samples = len(finetune_dataset)
    num_labeled_samples = int(labeling_fraction * num_total_samples)
    labeled_subset, unlabeled_subset = torch.utils.data.random_split(finetune_dataset, [num_labeled_samples, num_total_samples - num_labeled_samples])

    # DataLoader for labeled and unlabeled subsets
    labeled_data_loader = DataLoader(labeled_subset, batch_size=hyperparams.batch_size, shuffle=True)
    unlabeled_data_loader = DataLoader(unlabeled_subset, batch_size=hyperparams.batch_size, shuffle=True)

    optimizer_finetune = optim.SGD(rotnet_encoder.parameters(), lr=0.05 * hyperparams.batch_size / 256, momentum=0.9, nesterov=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    num_epochs_finetune = 60 if labeling_fraction == 0.01 else 30
    
    rotnet_encoder.train()

    for epoch in tqdm(range(num_epochs_finetune)):
        for batch in labeled_data_loader:
            finetune_inputs, finetune_labels = batch
            finetune_inputs = finetune_inputs.to(device)
            finetune_labels = finetune_labels.to(device)

            # Forward pass with SimCLR encoder
            features, labels = extract_features(labeled_dataloader, rotnet_encoder)

            # Fine-tuning: replace this part with your downstream task-specific head
            finetune_predictions = linear_classifier(features)
            targets = F.one_hot(finetune_labels, hyperparams.num_classes).float().to(device)
            # Compute supervised loss
            finetune_loss = criterion(finetune_predictions, labels)

            # Backward pass and optimization
            optimizer_finetune.zero_grad()
            finetune_loss.backward()
            optimizer_finetune.step()

    
# Step 4: Evaluation
# Evaluate the linear classifier on a test set
rotnet_encoder.eval()
subset_test_size = 640
subset_test_indices = indices[:subset_test_size]
subset_test_sampler = SubsetRandomSampler(subset_test_indices)
test_dataloader = DataLoader(dataset=dataset, batch_size=hyperparams.batch_size, sampler=subset_test_sampler)
test_features, test_labels = extract_features(test_dataloader, rotnet_encoder)

print("Evaluating with Linear Classifier...")
linear_classifier.eval()
with torch.no_grad():
    test_outputs = linear_classifier(test_features)
    _, predicted = torch.max(test_outputs, 1)
    print(predicted.shape)
    print(test_outputs.shape)
    top1accuracy = (predicted == test_labels).float().mean().item()
    print(f"Linear Classifier Top-1 Accuracy: {top1accuracy * 100:.2f}%")
