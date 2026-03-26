import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from models.resnet_model import get_model
from utils.dataloader import get_loaders


CLASS_NAMES = ["abnormal", "history_mi", "mi", "normal"]


def collect_predictions(model, loader, device):
    """Run the model on the entire loader and collect all predictions, labels, and probabilities."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return all_labels, all_preds, all_probs


def print_overall_accuracy(labels, preds):
    """Print overall accuracy."""
    correct = (labels == preds).sum()
    total = len(labels)
    acc = 100.0 * correct / total
    print("=" * 60)
    print(f"  OVERALL TEST ACCURACY:  {acc:.2f}%  ({correct}/{total})")
    print("=" * 60)
    return acc


def print_per_class_accuracy(labels, preds):
    """Print accuracy for each class."""
    print("\n--- Per-Class Accuracy ---")
    for idx, name in enumerate(CLASS_NAMES):
        mask = labels == idx
        class_total = mask.sum()
        if class_total == 0:
            print(f"  {name:>12s}:  N/A (no samples)")
            continue
        class_correct = (preds[mask] == idx).sum()
        class_acc = 100.0 * class_correct / class_total
        print(f"  {name:>12s}:  {class_acc:.2f}%  ({class_correct}/{class_total})")
    print()


def print_classification_report(labels, preds):
    """Print precision, recall, F1-score per class and averages."""
    print("--- Classification Report ---")
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report)


def print_misclassification_summary(labels, preds):
    """Show the most common misclassification pairs."""
    cm = confusion_matrix(labels, preds)
    print("--- Most Common Misclassifications ---")
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i][j] > 0:
                pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    if not pairs:
        print("  No misclassifications!")
    for true_cls, pred_cls, count in pairs[:10]:
        print(f"  {true_cls:>12s}  →  {pred_cls:<12s}:  {count} samples")
    print()


def plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0],
    )
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14)

    # Percentages
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Oranges",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1],
    )
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)
    axes[1].set_title("Confusion Matrix (% per class)", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix saved to: {save_path}")


def plot_roc_curves(labels, probs, save_path="roc_curves.png"):
    """Plot per-class ROC curves and compute AUC."""
    n_classes = len(CLASS_NAMES)
    # Binarize labels for one-vs-rest
    labels_bin = np.eye(n_classes)[labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    auc_scores = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[CLASS_NAMES[i]] = roc_auc
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})")

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color="navy", lw=2, linestyle="--",
            label=f"Micro-avg (AUC = {roc_auc_micro:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[✓] ROC curves saved to: {save_path}")
    print("\n--- ROC-AUC Scores (One-vs-Rest) ---")
    for name, score in auc_scores.items():
        print(f"  {name:>12s}:  {score:.4f}")
    print(f"  {'Micro-avg':>12s}:  {roc_auc_micro:.4f}")

    # Also compute macro AUC
    try:
        macro_auc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")
        print(f"  {'Macro-avg':>12s}:  {macro_auc:.4f}")
    except Exception:
        pass
    print()

    return auc_scores


def plot_precision_recall_curves(labels, probs, save_path="precision_recall_curves.png"):
    """Plot per-class Precision-Recall curves."""
    n_classes = len(CLASS_NAMES)
    labels_bin = np.eye(n_classes)[labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
        ap = average_precision_score(labels_bin[:, i], probs[:, i])
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f"{CLASS_NAMES[i]} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Precision-Recall curves saved to: {save_path}")


def plot_confidence_distribution(labels, preds, probs, save_path="confidence_dist.png"):
    """Plot confidence distribution for correct vs incorrect predictions."""
    max_probs = probs.max(axis=1)
    correct_mask = labels == preds

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(max_probs[correct_mask], bins=30, alpha=0.6, color="#2ecc71",
            label=f"Correct ({correct_mask.sum()})", density=True, edgecolor="white")
    ax.hist(max_probs[~correct_mask], bins=30, alpha=0.6, color="#e74c3c",
            label=f"Incorrect ({(~correct_mask).sum()})", density=True, edgecolor="white")
    ax.set_xlabel("Prediction Confidence", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Confidence Distribution: Correct vs Incorrect", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Confidence distribution saved to: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load data and model
    _, _, test_loader, dataset = get_loaders("dataset", batch_size=20)

    model = get_model(num_classes=4)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)

    print(f"Class mapping: {dataset.class_to_idx}")
    print(f"Test set size: {len(test_loader.dataset)} images\n")

    # Collect all predictions
    labels, preds, probs = collect_predictions(model, test_loader, device)

    # ── Metrics ──
    print_overall_accuracy(labels, preds)
    print_per_class_accuracy(labels, preds)
    print_classification_report(labels, preds)
    print_misclassification_summary(labels, preds)

    # ── Plots ──
    plot_confusion_matrix(labels, preds)
    plot_roc_curves(labels, probs)
    plot_precision_recall_curves(labels, probs)
    plot_confidence_distribution(labels, preds, probs)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE — All plots saved to project folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
