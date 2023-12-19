import math
import torch

# Hyperparameters for RotNet
class hyperParams:
    def __init__(self):
        self.batch_size = 128
        self.projection_dim = 128
        self.temperature = 0.5
        self.learning_rate = 0.0001
        self.weight_decay = 0.0005
        self.epochs = 50
        self.num_classes = 4
        self.momentum = 0.9
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'