import matplotlib.pyplot as plt

def plot(train_hist, val_hist):
    plt.plot(train_hist, label="Train Accuracy")
    plt.plot(val_hist, label="Validation Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
