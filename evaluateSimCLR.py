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

# Data Augmentation
data_transform = transforms.Compose([
    # transforms.Resize(224)
    transforms.ToTensor()
])

hyperparams = hyperParams()
dataset = datasets.CIFAR10(root='./data', download=True, train=False, transform=data_transform)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * .8), int(len(dataset) * .2)])
train_dataloader = DataLoader(dataset=train_set, batch_size=hyperparams.batch_size, shuffle=True)

print("Extracting Features...")
simclr_model = SimCLRModel(torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False).to(device))
simclr_model.load_state_dict(torch.load('simclr_encoder.pth', map_location=torch.device('cuda'))['state_dict'])
simclr_encoder = simclr_model.encoder
simclr_encoder.projection_head = nn.Identity()
linearClassifier = nn.Linear(1000, hyperparams.num_classes).to(device)

# Step 3: Linear classifier
# Train a linear classifier (logistic regression in this example)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(linearClassifier.parameters(), lr=0.01)

simclr_model.eval()
print("Training Linear Classifier...")
# Train the linear classifier
num_epochs = 10
for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        features = simclr_encoder(inputs)
        outputs = linearClassifier(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss: {loss.item()}")
    
# Fine-Tuning SimCLR
print("Finetuning SimCLR...")
finetune_data_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Define the labeling fraction for fine-tuning
labeling_fraction = hyperparams.labeling_fraction # Example: 10% of the dataset is labeled
finetune_dataset = datasets.CIFAR10(root='./data', download=True, train=False, transform=finetune_data_transform)


if labeling_fraction > 0:
    # Split dataset into labeled and unlabeled subsets
    num_total_samples = len(finetune_dataset)
    num_labeled_samples = int(labeling_fraction * num_total_samples)
    labeled_subset, unlabeled_subset = torch.utils.data.random_split(finetune_dataset, [num_labeled_samples, num_total_samples -  num_labeled_samples])

    # DataLoader for labeled and unlabeled subsets
    labeled_dataloader = DataLoader(labeled_subset, batch_size=hyperparams.batch_size, shuffle=True)

    optimizer_finetune = optim.SGD(simclr_encoder.parameters(), lr=0.05 * hyperparams.batch_size / 256, momentum=0.9, nesterov=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    num_epochs_finetune = 60 if labeling_fraction == 0.01 else 30
    
    simclr_model.train()

    for epoch in tqdm(range(num_epochs_finetune)):
        for batch in labeled_dataloader:
            finetune_inputs, finetune_labels = batch
            finetune_inputs = finetune_inputs.to(device)
            finetune_labels = finetune_labels.to(device)

            # Forward pass with SimCLR encoder
            features = simclr_encoder(finetune_inputs)
            outputs = linearClassifier(features)
            loss = criterion(outputs, finetune_labels)

            # Backward pass and optimization
            optimizer_finetune.zero_grad()
            loss.backward()
            optimizer_finetune.step()

    
# Step 4: Evaluation
# Evaluate the linear classifier on a test set
test_dataloader = DataLoader(dataset=test_set, batch_size=hyperparams.batch_size, shuffle=True)

print("Evaluating with Linear Classifier...")
top1 = []
top5 = []
simclr_encoder.eval()
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        features = simclr_encoder(inputs)
        outputs = linearClassifier(features)
        _, predicted = torch.max(outputs, 1)
        top5guesses = torch.topk(outputs, k=5, dim=1).indices
        top5accuracy = torch.sum(torch.Tensor([labels[i] in top5guesses[i] for i in range(len(labels))])).item()/len(labels)
        top1accuracy = (predicted == labels).float().mean().item()
        top5.append(top5accuracy)
        top1.append(top1accuracy)
    print(f"Linear Classifier Top-1 Accuracy: {sum(top1)/len(top1) * 100:.2f}%")
    print(f"Linear Classifier Top-5 Accuracy: {sum(top5)/len(top5) * 100:.2f}%")
