import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from tqdm import tqdm

from .config import DEVICE, CLASS_NAMES, RESULTS_DIR


def evaluate_model(model, test_loader):
    """Run model on test set, return predictions and ground truth."""
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred):
    """Compute accuracy, macro F1, weighted F1, and per-class report."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    class_names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"\n{report}")

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plot and optionally save confusion matrix heatmap."""
    class_names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_training_curves(history, title="Training Curves", save_path=None):
    """Plot loss and accuracy curves for training and validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
