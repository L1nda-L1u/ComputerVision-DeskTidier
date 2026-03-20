"""
Grad-CAM: Visualize what the ResNet18 classifier actually learned.
Shows heatmaps of which regions the model focuses on for its predictions.
"""

import csv
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGE_DIR = Path("../jpg_images")
LABELS_CSV = Path("desk_labels.csv")
MODEL_PATH = Path("desk_classifier.pth")
IMG_SIZE = 224
SEED = 42
CLASS_NAMES = ["tidy", "untidy"]

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def load_labels():
    label_map = {"tidy": 0, "untidy": 1}
    samples = []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = IMAGE_DIR / row["image"]
            if p.exists() and row["label"] in label_map:
                samples.append((p, label_map[row["label"]]))
    return samples


class GradCAM:
    """Grad-CAM for ResNet18 layer4."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.model.layer4.register_forward_hook(self._save_activation)
        self.model.layer4.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output


def overlay_heatmap(img_np, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return alpha * heatmap + (1 - alpha) * img_np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    grad_cam = GradCAM(model)

    samples = load_labels()
    random.seed(SEED)
    random.shuffle(samples)

    # Pick 4 tidy + 4 untidy for analysis
    tidy_samples = [(p, l) for p, l in samples if l == 0]
    untidy_samples = [(p, l) for p, l in samples if l == 1]
    random.shuffle(tidy_samples)
    random.shuffle(untidy_samples)
    selected = tidy_samples[:3] + untidy_samples[:3]

    fig, axes = plt.subplots(6, 4, figsize=(18, 26))
    fig.suptitle("Grad-CAM: What does the model actually look at?",
                 fontsize=18, fontweight="bold")

    col_titles = ["Original", "Grad-CAM Heatmap", "Overlay", "Prediction Detail"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for i, (path, true_label) in enumerate(selected):
        img_pil = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img_pil) / 255.0

        input_tensor = val_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        cam, output = grad_cam.generate(input_tensor)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        pred = output.argmax(dim=1).item()

        overlay = overlay_heatmap(img_np, cam)

        row = i if i < 4 else i
        if i >= 4:
            row = i

        # Col 0: Original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(f"{'TIDY' if true_label == 0 else 'UNTIDY'}\n{path.name}",
                              fontsize=11, fontweight="bold",
                              color="#4CAF50" if true_label == 0 else "#FF5722")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Col 1: Heatmap
        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Col 2: Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # Col 3: Prediction bar chart
        ax = axes[i, 3]
        colors = ["#4CAF50", "#FF5722"]
        bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, alpha=0.85, height=0.5)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{p*100:.1f}%", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 115)
        correct = pred == true_label
        ax.set_title(f"{'Correct' if correct else 'WRONG'}",
                     color="#4CAF50" if correct else "#FF0000",
                     fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("gradcam_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved: gradcam_analysis.png")

    # ── Data leakage analysis ──
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS - What did it learn?")
    print("=" * 60)

    print(f"\nDataset: {len(samples)} images")
    print(f"  tidy: {sum(1 for _, l in samples if l == 0)}")
    print(f"  untidy: {sum(1 for _, l in samples if l == 1)}")

    print("\n--- Potential Concerns ---")
    print("1. ALL images from same desk, same day, same camera angle")
    print("   -> Model may learn lighting/background, not 'tidiness' concept")
    print("2. Class imbalance: 28 tidy vs 111 untidy (1:4 ratio)")
    print("   -> WeightedRandomSampler helps but not perfect")
    print("3. No held-out test set from different desk/environment")
    print("   -> 99% accuracy may not generalize to other desks")

    print("\n--- What ResNet18 likely learned ---")
    print("- Visual complexity / edge density (messy = more edges)")
    print("- Object density in the image (more stuff = untidy)")
    print("- Color variety (more objects = more diverse colors)")
    print("- Overall 'clutter texture' patterns")

    print("\n--- What it probably CANNOT learn ---")
    print("- Semantic 'overlap' or 'stacking' concepts")
    print("- Object alignment / orientation rules")
    print("- Whether specific objects are 'in the right place'")
    print("- Tidiness of a completely different desk setup")

    print("\n--- For coursework, this is fine because ---")
    print("- Transfer learning baseline with good results")
    print("- Grad-CAM shows interpretable attention regions")
    print("- Can discuss limitations in report")
    print("- Could improve with: more data, different desks, cross-val")


if __name__ == "__main__":
    main()
