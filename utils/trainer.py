import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, patience=5):

    best_val_acc = 0
    patience_counter = 0

    train_history = []
    val_history = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader, device)

        train_history.append(train_acc)
        val_history.append(val_acc)

        print(f"Epoch {epoch+1}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    return train_history, val_history


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
