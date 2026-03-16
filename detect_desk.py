"""
Desk Object Detection using Ultralytics YOLO (no training)
For Computer Vision coursework - testing desk images.
Aligned with Scoring Framework: Core Work + Study + Temporary items only.

Install: pip install ultralytics
"""

from pathlib import Path

from ultralytics import YOLO

# Desk-relevant COCO classes only (Scoring Framework)
# Note: COCO has no pen, pencil, earphones, notebook - fine-tune for those
DESK_CLASSES = {
    "laptop", "mouse", "keyboard", "cell phone", "book",
    "bottle", "cup", "bowl", "scissors", "clock", "remote",
    "dining table",  # desk surface
}


def detect_objects(
    image_path: str,
    output_dir: str = "detection_output",
    confidence_threshold: float = 0.25,
) -> None:
    """
    Run pretrained YOLO detection on a single image.
    Draws boxes, saves result, and prints detection summary.
    """
    # Validate image path
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    # Load pretrained model (YOLOv8n = nano, fast and good for beginners)
    model = YOLO("yolov8n.pt")

    # Restrict to desk-relevant classes only (Scoring Framework: laptop, phone, book, cup, etc.)
    allowed_ids = [i for i, name in model.names.items() if name in DESK_CLASSES]
    active_classes = [model.names[i] for i in allowed_ids]

    # Run inference (no training)
    results = model.predict(
        source=str(path),
        conf=confidence_threshold,
        classes=allowed_ids,
        save=True,
        project=str(Path.cwd()),
        name=output_dir,
        exist_ok=True,
    )

    # Get the first result (single image)
    r = results[0]

    # Print detected objects
    print(f"\nDetecting only: {', '.join(sorted(active_classes))}")
    print("\n--- Detected Objects ---")
    if r.boxes is None or len(r.boxes) == 0:
        print("No objects detected.")
        return

    # Collect per-class counts
    class_counts = {}
    detections = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = r.names[cls_id]
        detections.append((name, conf))
        class_counts[name] = class_counts.get(name, 0) + 1

    # Print each detection
    for name, conf in detections:
        print(f"  {name}: {conf:.2%}")

    # Print per-class counts
    print("\n--- Counts per Class ---")
    for name, count in sorted(class_counts.items()):
        print(f"  {name}: {count}")

    print(f"\nTotal detections: {len(detections)}")
    save_path = getattr(r, "save_dir", None) or f"{output_dir}/run"
    print(f"\nOutput saved to: {save_path}")


if __name__ == "__main__":
    # Config
    IMAGE_PATH = "jpg_images/desk_015.jpg"
    OUTPUT_DIR = "detection_output"
    CONFIDENCE = 0.05  # For overhead desk shots, use 0.05–0.1 (COCO model trained on side views)

    try:
        detect_objects(
            image_path=IMAGE_PATH,
            output_dir=OUTPUT_DIR,
            confidence_threshold=CONFIDENCE,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
