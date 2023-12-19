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

# Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i and z_j are the projections of augmented views of the same image.
        The inputs are expected to have shape (batch_size, embedding_size).
        """
        # Normalize the vectors along the embedding dimension
        z_i_norm = torch.norm(z_i, dim=1).reshape(-1,1)
        z_j_norm = torch.norm(z_j, dim=1).reshape(-1,1)
        
        z_i_normed = torch.div(z_i, z_i_norm)
        z_j_normed = torch.div(z_j, z_j_norm)
        
        z_ij = torch.cat([z_i_normed, z_j_normed], dim=0)
        z_ji = torch.cat([z_j_normed, z_i_normed], dim=0)
        
        # Compute the similarity matrix (dot product) and divide by temperature
        sim_matrix = torch.div(torch.mm(z_ij, z_ij.t()), self.temperature)
        
        exp_sum_sim_matrix = torch.sum(torch.exp(sim_matrix), dim=1)
        
        numerator = torch.exp(torch.div(nn.CosineSimilarity()(z_ij, z_ji), self.temperature))

        # Exclude diagonal elements (similarity of each sample with itself)
        denominator = exp_sum_sim_matrix - torch.diag(torch.exp(sim_matrix))

        loss = -torch.log(torch.div(numerator, denominator))

        return torch.mean(loss)
  
