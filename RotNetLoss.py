import torch
import torch.nn as nn
import torch.nn.functional as F

class RotNetLoss(nn.Module):
    def __init__(self):
        super(RotNetLoss, self).__init__()
        self.num_classes = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, rotation_logits, rotation_labels):
        loss = nn.CrossEntropyLoss()(rotation_logits,  F.one_hot(rotation_labels, self.num_classes).float().to(self.device))
        return loss
