import torch.nn as nn
from torchvision import models

def getresnet18(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False  # transfer learning

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model