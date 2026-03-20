from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


def load_tidy_module(script_path: Path):
    module_name = "tidy_scoring_system"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_classifier_model(model_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def classify_need_tidy(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> tuple[str, bool]:
    """
    Returns (pred_label, need_tidy).
    Label mapping from training script: 0=tidy, 1=untidy.
    """

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
    pred_label = "untidy" if pred == 1 else "tidy"
    return pred_label, pred == 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tidy scores with classifier gating.")
    parser.add_argument(
        "--classifier-model",
        type=str,
        default="",
        help="Path to classifier .pth file (required for gating).",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    module_root = repo_dir.parent
    project_root = module_root.parent
    tidy_script = repo_dir / "Tidy Scoring System.py"
    tidy = load_tidy_module(tidy_script)

    model_path = project_root / "runs" / "detect" / "desk_tidy_runs" / "v4_yolov8m_roboflow_style" / "weights" / "best.pt"
    source_dir = Path(r"C:\Users\caoyi\Desktop\DwBS\DVS\final Project\DeskTidier.v3i.yolov8")
    output_csv = module_root / "outputs" / "tidy_scores_conf0.4_v4.csv"

    conf = 0.4
    iou = 0.5
    imgsz = 640
    device = "cpu"
    cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_path = Path(args.classifier_model) if args.classifier_model else (project_root / "classifier" / "desk_classifier.pth")

    classifier_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in exts])
    if not images:
        raise FileNotFoundError(f"No images found under: {source_dir}")
    if not classifier_path.exists():
        raise FileNotFoundError(
            "Classifier model not found. "
            f"Expected: {classifier_path}. "
            "Please pass --classifier-model \"<path to desk_classifier.pth>\"."
        )

    model = YOLO(str(model_path))
    classifier = load_classifier_model(classifier_path, cls_device)
    rows: List[Dict[str, Any]] = []

    for img_path in images:
        pred_label, need_tidy = classify_need_tidy(classifier, img_path, classifier_transform, cls_device)
        if not need_tidy:
            rows.append(
                {
                    "image_name": img_path.name,
                    "classifier_pred": pred_label,
                    "need_tidy": 0,
                    "scoring_skipped": 1,
                    "detected_count": "",
                    "score": "",
                    "level": "",
                    "object_load_penalty": "",
                    "category_penalty": "",
                    "workspace_obstruction_penalty": "",
                    "spatial_overlap_penalty": "",
                    "spatial_dispersion_penalty": "",
                    "alignment_penalty": "",
                    "total_penalty": "",
                }
            )
            continue

        pred = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        detections = []
        if getattr(pred, "boxes", None) is not None and len(pred.boxes) > 0:
            names = pred.names
            for xyxy, score, cls_id in zip(pred.boxes.xyxy.tolist(), pred.boxes.conf.tolist(), pred.boxes.cls.tolist()):
                label = names.get(int(cls_id), str(int(cls_id)))
                detections.append(
                    tidy.Detection(
                        label=label,
                        confidence=float(score),
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        angle_deg=None,
                    )
                )

        h, w = pred.orig_shape
        res = tidy.tidy_score(
            detections,
            image_size=(w, h),
            desk_orientation_deg=0.0,
            alignment_misalignment_threshold_deg=15.0,
        )
        penalties = res["penalties"]
        rows.append(
            {
                "image_name": img_path.name,
                "classifier_pred": pred_label,
                "need_tidy": 1,
                "scoring_skipped": 0,
                "detected_count": len(detections),
                "score": res["tidy_score"],
                "level": res["tidy_level"],
                "object_load_penalty": penalties["object_load_penalty"],
                "category_penalty": penalties["category_penalty"],
                "workspace_obstruction_penalty": penalties["workspace_obstruction_penalty"],
                "spatial_overlap_penalty": penalties["spatial_overlap_penalty"],
                "spatial_dispersion_penalty": penalties["spatial_dispersion_penalty"],
                "alignment_penalty": penalties["alignment_penalty"],
                "total_penalty": res["total_penalty"],
            }
        )

    fieldnames = [
        "image_name",
        "classifier_pred",
        "need_tidy",
        "scoring_skipped",
        "detected_count",
        "score",
        "level",
        "object_load_penalty",
        "category_penalty",
        "workspace_obstruction_penalty",
        "spatial_overlap_penalty",
        "spatial_dispersion_penalty",
        "alignment_penalty",
        "total_penalty",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scored_rows = [r for r in rows if r["scoring_skipped"] == 0]
    avg_score = sum(float(r["score"]) for r in scored_rows) / len(scored_rows) if scored_rows else 0.0
    print(f"Exported {len(rows)} rows to: {output_csv}")
    print(f"Scored rows: {len(scored_rows)} | Skipped rows: {len(rows) - len(scored_rows)}")
    print(f"Average score (scored rows only): {avg_score:.2f}")


if __name__ == "__main__":
    main()

