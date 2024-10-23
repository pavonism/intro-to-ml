import torch.nn as nn
import timm


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # TODO: Remove pretrained model
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(enet_out_size, 2))

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
