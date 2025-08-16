import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet():
    model = models.resnet18(weights = ResNet18_Weights.DEFAULT)
    OUTPUTS = 2
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS)

    return model