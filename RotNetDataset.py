import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class RotatedCIFAR10(Dataset):
    def __init__(self, root, transform=None):
        self.cifar_dataset = CIFAR10(root, train=True, download=True, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, index):
        original_image, original_label = self.cifar_dataset[index]

        rotation_angle = np.random.choice([0, 90, 180, 270]).item()

        # Rotate image
        rotated_image = transforms.functional.rotate(original_image, rotation_angle)

        # Convert labels to tensor
        rotated_label = torch.tensor(rotation_angle // 90)

	# Allow for further transformation
        if self.transform:
            rotated_image = self.transform(rotated_image)

        return rotated_image, rotated_label

