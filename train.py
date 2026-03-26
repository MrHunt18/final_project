import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import get_loaders
from utils.trainer import train_model
from models.resnet_model import get_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Load Data
    train_loader, val_loader, test_loader, dataset = get_loaders(
        "dataset",
        batch_size=20
    )

    # Compute class weights automatically
    class_counts = []
    for i in range(len(dataset.classes)):
        class_counts.append(
            len([label for _, label in dataset.samples if label == i])
        )

    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    # Model
    model = get_model(num_classes=4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_hist, val_hist = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=20,
        patience=5
    )

    print("Training Complete!")


if __name__ == "__main__":
    main()
