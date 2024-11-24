from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms

from model_training.architectures import Architecture
from model_training.architectures.transforms import KeepImageChannels


class SimpleConvolutionNet(nn.Module):
    def __init__(self, in_channels=3, out_features=30):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 72 * 97, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleConvolutionArchitecture(Architecture):
    def __init__(self, rgba_channels: List[int] = []):
        self.rgba_channels = rgba_channels

    def get_model(self):
        return SimpleConvolutionNet(
            in_channels=len(self.rgba_channels) if self.rgba_channels else 3
        )

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((300, 400)),
                transforms.ToTensor(),
                *(
                    [KeepImageChannels(self.rgba_channels)]
                    if self.rgba_channels
                    else []
                ),
            ]
        )
