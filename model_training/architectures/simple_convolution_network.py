import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms


class SimpleConvolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 72 * 97, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleConvolutionArchitecture:
    def get_model(self):
        return SimpleConvolutionNet()

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((300, 400)),
                transforms.ToTensor(),
            ]
        )
