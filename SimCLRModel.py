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

# SimCLR Model Definition
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLRModel, self).__init__()

        # Base encoder (e.g., ResNet18 without the final classification layer)
        self.encoder = base_encoder

        self.projection_dim = projection_dim
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Projection head
        self.projection_head = nn.Sequential(
            # nn.Linear(self.encoder.fc.in_features, 512),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        ).to(self.device)
        
        

    def forward(self, x):
        # Forward pass through the encoder
        h = self.encoder(x).to(self.device)

        # Forward pass through the projection head
        z = self.projection_head(h).to(self.device)

        return h, z

    def get_fc_in_features(self):
        # Extract the number of input features for the projection head
        return self.encoder.fc.in_features

