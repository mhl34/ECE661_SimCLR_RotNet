import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class RotNet(nn.Module):
    def __init__(self, base_encoder, projection_dim=4):
        super(RotNet, self).__init__()

        self.encoder = base_encoder
        
        self.rotation_head = nn.Sequential(
            nn.Linear(1000, 640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Linear(320, 240),
            nn.ReLU(),
            nn.Linear(240, projection_dim)  # Four rotation classes: 0, 90, 180, 270 degrees
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        x = self.encoder(x.to(self.device))
        
        rotation_logits = self.rotation_head(x)
        rotation_pos_logits = torch.min(rotation_logits) + rotation_logits
        rotation_final_logits = rotation_pos_logits
        return rotation_final_logits.to(self.device)
