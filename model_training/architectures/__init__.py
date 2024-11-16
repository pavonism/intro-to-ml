from torch import nn
from torchvision import transforms


class Architecture:
    def get_model(self) -> nn.Module:
        pass

    def get_transform(self) -> transforms.Compose:
        pass
