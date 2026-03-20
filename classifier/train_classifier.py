"""
ResNet18 Transfer Learning: Tidy vs Untidy Desk Classifier
Uses GPU (CUDA) if available.

Install: pip install torch torchvision scikit-learn matplotlib
"""

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# ──────────────────── Config ────────────────────
IMAGE_DIR = Path("../jpg_images")
LABELS_CSV = Path("desk_labels.csv")
MODEL_SAVE_PATH = Path("desk_classifier.pth")

NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42
IMG_SIZE = 224


# ──────────────────── Dataset ────────────────────
class DeskDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
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


# ──────────────────── Transforms ────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_labels() -> list[tuple[Path, int]]:
    """Load CSV labels: returns [(image_path, 0=tidy/1=untidy), ...]"""
    label_map = {"tidy": 0, "untidy": 1}
    samples = []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = IMAGE_DIR / row["image"]
            if img_path.exists() and row["label"] in label_map:
                samples.append((img_path, label_map[row["label"]]))
    return samples


def split_data(samples, val_ratio=VAL_SPLIT):
    random.seed(SEED)
    random.shuffle(samples)
    split = int(len(samples) * (1 - val_ratio))
    return samples[:split], samples[split:]


def build_sampler(train_samples):
    """WeightedRandomSampler to handle tidy/untidy imbalance (28 vs 111)."""
    labels = [s[1] for s in train_samples]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_model(device):
    """ResNet18 pretrained, replace final FC for binary classification."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze early layers, only train later layers + FC
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze layer4 + FC for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    samples = load_labels()
    print(f"Total samples: {len(samples)}")
    tidy_count = sum(1 for _, l in samples if l == 0)
    print(f"  tidy: {tidy_count}, untidy: {len(samples) - tidy_count}")

    train_samples, val_samples = split_data(samples)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = DeskDataset(train_samples, train_transform)
    val_ds = DeskDataset(val_samples, val_transform)

    sampler = build_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # Training loop
    best_val_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)

        # ── Validate ──
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = correct / total
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nBest Val Accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    # ── Final evaluation on best model ──
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=["tidy", "untidy"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # ── Plot training curves ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, NUM_EPOCHS + 1), train_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")

    ax2.plot(range(1, NUM_EPOCHS + 1), val_accs)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("\nTraining curves saved to: training_curves.png")


if __name__ == "__main__":
    train()
