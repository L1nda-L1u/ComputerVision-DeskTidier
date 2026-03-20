"""
Visualize ResNet18 classifier results:
- Training curves (loss + accuracy)
- Confusion matrix heatmap
- Per-class metrics bar chart
- Sample predictions with images
"""

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_DIR = Path("../jpg_images")
LABELS_CSV = Path("desk_labels.csv")
MODEL_PATH = Path("desk_classifier.pth")
IMG_SIZE = 224
SEED = 42
BATCH_SIZE = 16

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

display_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

CLASS_NAMES = ["tidy", "untidy"]


class DeskDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_labels():
    label_map = {"tidy": 0, "untidy": 1}
    samples = []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = IMAGE_DIR / row["image"]
            if p.exists() and row["label"] in label_map:
                samples.append((p, label_map[row["label"]]))
    return samples


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def get_all_predictions(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_preds, all_labels, all_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples = load_labels()
    random.seed(SEED)
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples, val_samples = samples[:split], samples[split:]

    # ── Run on full dataset for visualization ──
    full_ds = DeskDataset(samples, val_transform)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False)

    val_ds = DeskDataset(val_samples, val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(device)

    # Full dataset predictions
    all_preds, all_labels, all_probs = get_all_predictions(model, full_loader, device)
    val_preds, val_labels, _ = get_all_predictions(model, val_loader, device)

    # ────────── Figure 1: Training curves ──────────
    # Re-train metrics from the training log
    train_losses = [0.4473, 0.1243, 0.0582, 0.0416, 0.0186, 0.0158, 0.0243, 0.0211,
                    0.0147, 0.0271, 0.0457, 0.0224, 0.0075, 0.0092, 0.0094, 0.0070,
                    0.0229, 0.0056, 0.0046, 0.0082]
    val_accs = [0.9286, 0.9286, 0.9286, 0.9286, 0.9286, 0.9286, 0.9286, 0.9286,
                0.8929, 0.9286, 0.9643, 0.9643, 0.9643, 0.9643, 0.9643, 0.9643,
                0.9643, 0.9643, 0.9286, 0.9286]
    epochs = range(1, 21)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("ResNet18 Desk Classifier — Training Results", fontsize=16, fontweight="bold")

    # (1) Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, "o-", color="#2196F3", linewidth=2, markersize=5)
    ax.fill_between(epochs, train_losses, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.set_xlim(1, 20)
    ax.grid(True, alpha=0.3)

    # (2) Validation Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [a * 100 for a in val_accs], "o-", color="#4CAF50", linewidth=2, markersize=5)
    ax.fill_between(epochs, [a * 100 for a in val_accs], alpha=0.15, color="#4CAF50")
    ax.axhline(y=96.43, color="#FF5722", linestyle="--", alpha=0.7, label="Best: 96.43%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy")
    ax.set_xlim(1, 20)
    ax.set_ylim(85, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) Confusion Matrix (Full dataset)
    ax = axes[1, 0]
    cm = confusion_matrix(all_labels, all_preds)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (All 139 images)")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=24, fontweight="bold", color=color)

    # (4) Per-class metrics
    ax = axes[1, 1]
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    tidy_vals = [report["tidy"][m] for m in metrics]
    untidy_vals = [report["untidy"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.3
    bars1 = ax.bar(x - width / 2, [v * 100 for v in tidy_vals], width, label="tidy", color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + width / 2, [v * 100 for v in untidy_vals], width, label="untidy", color="#FF5722", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-Score"])
    ax.set_ylabel("Score (%)")
    ax.set_title("Per-Class Metrics (All 139 images)")
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: results_summary.png")

    # ────────── Figure 2: Sample Predictions ──────────
    fig2, axes2 = plt.subplots(2, 5, figsize=(18, 8))
    fig2.suptitle("Sample Predictions (Random 10 images)", fontsize=16, fontweight="bold")

    random.seed(42)
    indices = random.sample(range(len(samples)), 10)

    for i, idx in enumerate(indices):
        row, col = i // 5, i % 5
        ax = axes2[row, col]
        path, true_label = samples[idx]
        pred_label = all_preds[idx]
        prob = all_probs[idx]

        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ax.imshow(img)

        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_label]
        conf = prob[pred_label] * 100
        correct = pred_label == true_label

        color = "#4CAF50" if correct else "#FF0000"
        symbol = "Correct" if correct else "WRONG"
        ax.set_title(f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)\n{symbol}",
                     fontsize=9, color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
    print("Saved: sample_predictions.png")

    # Print summary
    total_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    print(f"\nOverall Accuracy (all 139): {total_correct}/{len(all_labels)} = {total_correct/len(all_labels):.2%}")
    print(f"Val Accuracy (28 images):   {sum(1 for p, l in zip(val_preds, val_labels) if p == l)}/{len(val_labels)}")
    print(f"\nClassification Report (all 139 images):")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
