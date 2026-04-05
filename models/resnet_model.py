import torch.nn as nn
import torchvision

def get_model(num_classes=4, weights=None):
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model
