"""
Create train/val folders from labels.csv for PyTorch training.
Run after label_desk.py is done.

Output:
  dataset/
    tidy/      <- tidy images
    untidy/    <- untidy images
"""

import shutil
from pathlib import Path

import csv

IMAGE_DIR = Path("../jpg_images")
LABELS_CSV = Path("desk_labels.csv")
OUTPUT_DIR = Path("dataset")


def main():
    if not LABELS_CSV.exists():
        print(f"Run label_desk.py first to create {LABELS_CSV}")
        return

    labels = {}
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["image"]] = row["label"]

    for label in ("tidy", "untidy"):
        (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

    for img_name, label in labels.items():
        src = IMAGE_DIR / img_name
        if not src.exists():
            print(f"Skip (missing): {img_name}")
            continue
        dst = OUTPUT_DIR / label / img_name
        shutil.copy2(src, dst)
        print(f"  {img_name} -> {label}/")

    tidy_count = sum(1 for l in labels.values() if l == "tidy")
    untidy_count = sum(1 for l in labels.values() if l == "untidy")
    print(f"\nDone. dataset/tidy: {tidy_count}, dataset/untidy: {untidy_count}")


if __name__ == "__main__":
    main()
