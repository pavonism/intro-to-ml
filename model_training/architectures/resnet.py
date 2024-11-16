import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from model_training.architectures import Architecture


class ResNetArchitecture(Architecture):
    def __init__(self, out_features=30):
        self.__model = models.resnet50(pretrained=True)

        # Freeze all layers except the final fully connected layer
        for param in self.__model.parameters():
            param.requires_grad = False

        self.__model.fc = nn.Linear(self.__model.fc.in_features, out_features)

    def get_model(self):
        return self.__model

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=3, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
