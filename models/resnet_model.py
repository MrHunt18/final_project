import torch.nn as nn
import torchvision

def get_model(num_classes=4):
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model
