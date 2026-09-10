import torch
from torch import nn
from torchvision import models

class BrainTumorModel(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        # Load pretrained ResNet-50
        self.resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Freeze pretrained layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the original classifier
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.resnet(x)
