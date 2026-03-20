"""
Quick labeling tool for tidy/untidy binary classification.
Shows each image, press key to label.

Usage: python label_desk.py
Keys: 1 or t = tidy,  2 or u = untidy,  s = skip,  q = quit (saves)
"""

import csv
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Install: pip install opencv-python")
    exit(1)

IMAGE_DIR = Path("../jpg_images")
LABELS_CSV = Path("desk_labels.csv")


def load_existing_labels() -> dict[str, str]:
    """Load already-labeled entries from CSV."""
    labels = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    labels[row[0]] = row[1]
    return labels


def save_labels(labels: dict[str, str]) -> None:
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "label"])
        for img, lbl in sorted(labels.items()):
            w.writerow([img, lbl])


def main():
    if not IMAGE_DIR.exists():
        print(f"Error: {IMAGE_DIR} not found")
        return

    images = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.JPG"))
    if not images:
        print(f"No images in {IMAGE_DIR}")
        return

    labels = load_existing_labels()
    print(f"Loaded {len(labels)} existing labels. {len(images)} images total.")
    print("Keys: 1/t = tidy, 2/u = untidy, s = skip, q = quit & save\n")

    for path in images:
        name = path.name
        if name in labels:
            continue  # skip already labeled

        img = cv2.imread(str(path))
        if img is None:
            print(f"Skip (cannot read): {name}")
            continue

        # Resize to fit screen if too large
        h, w = img.shape[:2]
        max_h = 800
        if h > max_h:
            scale = max_h / h
            img = cv2.resize(img, (int(w * scale), max_h))

        cv2.imshow("Label: 1=tidy 2=untidy s=skip q=quit", img)
        key = chr(cv2.waitKey(0) & 0xFF).lower()

        if key == "q":
            break
        elif key in ("1", "t"):
            labels[name] = "tidy"
            print(f"  {name} -> tidy")
        elif key in ("2", "u"):
            labels[name] = "untidy"
            print(f"  {name} -> untidy")
        elif key == "s":
            pass
        else:
            print(f"  {name} -> (unknown key, skipped)")

    cv2.destroyAllWindows()
    save_labels(labels)
    print(f"\nSaved {len(labels)} labels to {LABELS_CSV}")


if __name__ == "__main__":
    main()
