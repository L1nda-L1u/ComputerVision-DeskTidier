"""
Unified Desk Tidier Pipeline
=============================
Integrates all project components into a single end-to-end workflow:

  1. Binary classifier (ResNet18)  → decides if tidying is needed
  2. Custom YOLO (v4 YOLOv8m)      → detects desk objects
  3. Tidy Scoring System           → computes score + penalties
  4. Language Recommendations      → rule-based text suggestions
  5. Visual Recommendations        → plan image + relayout image

Usage:
    python run_pipeline.py --image jpg_images/desk_065.jpg
    python run_pipeline.py --image jpg_images/desk_065.jpg --skip-classifier
    python run_pipeline.py --image jpg_images/ --conf 0.3
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# ──────────────────────── Path setup ──────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
SCORING_SCRIPT = PROJECT_ROOT / "scoring_module" / "scripts" / "Tidy Scoring System.py"
CLASSIFIER_MODEL = PROJECT_ROOT / "classifier" / "desk_classifier.pth"
YOLO_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "desk_tidy_runs"
    / "v4_yolov8m_roboflow_style"
    / "weights"
    / "best.pt"
)

# ──────────────────────── Load scoring module ─────────────────────────────────


def _load_scoring_module():
    spec = importlib.util.spec_from_file_location("tidy_scoring_system", str(SCORING_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scoring module from {SCORING_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tidy_scoring_system"] = mod
    spec.loader.exec_module(mod)
    return mod


_scoring = _load_scoring_module()
Detection = _scoring.Detection
tidy_score = _scoring.tidy_score
infer_category = _scoring.infer_category
estimate_object_angle_deg = _scoring.estimate_object_angle_deg

# ──────────────────────── Classifier ──────────────────────────────────────────

_CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_classifier(model_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def classify_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
) -> tuple[str, bool]:
    """Returns (pred_label, need_tidy). 0=tidy, 1=untidy."""
    img = Image.open(image_path).convert("RGB")
    x = _CLASSIFIER_TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
    label = "untidy" if pred == 1 else "tidy"
    return label, pred == 1


# ──────────────────────── YOLO detection ──────────────────────────────────────


def run_yolo(
    model: YOLO,
    image_path: str,
    conf: float = 0.4,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str = "cpu",
    estimate_angles: bool = True,
) -> List[Detection]:
    pred = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    names = getattr(pred, "names", None) or getattr(model, "names", None) or {}
    orig_img = getattr(pred, "orig_img", None)

    detections: List[Detection] = []
    if getattr(pred, "boxes", None) is not None and len(pred.boxes) > 0:
        for xyxy, score, cls_id in zip(
            pred.boxes.xyxy.tolist(),
            pred.boxes.conf.tolist(),
            pred.boxes.cls.tolist(),
        ):
            label = names.get(int(cls_id), str(int(cls_id)))
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            angle_deg = None
            if estimate_angles and isinstance(orig_img, np.ndarray):
                h_img, w_img = orig_img.shape[:2]
                ix1 = max(0, min(w_img, int(round(x1))))
                iy1 = max(0, min(h_img, int(round(y1))))
                ix2 = max(0, min(w_img, int(round(x2))))
                iy2 = max(0, min(h_img, int(round(y2))))
                if ix2 > ix1 and iy2 > iy1:
                    roi = orig_img[iy1:iy2, ix1:ix2]
                    angle_deg = estimate_object_angle_deg(label, roi)

            detections.append(Detection(
                label=label,
                confidence=float(score),
                bbox=(x1, y1, x2, y2),
                angle_deg=angle_deg,
            ))

    return detections, pred.orig_shape


# ──────────────────────── Pipeline ────────────────────────────────────────────


def process_image(
    image_path: Path,
    yolo_model: YOLO,
    classifier: nn.Module | None,
    cls_device: torch.device,
    conf: float = 0.4,
    iou: float = 0.5,
    imgsz: int = 640,
    yolo_device: str = "cpu",
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run the full pipeline on one image. Returns a result dict."""

    stem = image_path.stem
    if output_dir is None:
        output_dir = PROJECT_ROOT / "pipeline_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {"image": str(image_path)}

    # ── Step 1: Classifier gating ──
    if classifier is not None:
        pred_label, need_tidy = classify_image(classifier, image_path, cls_device)
        result["classifier_pred"] = pred_label
        result["need_tidy"] = need_tidy
        print(f"\n{'='*60}")
        print(f"Image: {image_path.name}")
        print(f"Classifier: {pred_label}")
        if not need_tidy:
            print("Desk is tidy — no further analysis needed.")
            print(f"{'='*60}")
            return result
    else:
        result["classifier_pred"] = "skipped"
        result["need_tidy"] = True
        print(f"\n{'='*60}")
        print(f"Image: {image_path.name}")
        print("Classifier: skipped")

    # ── Step 2: YOLO detection ──
    detections, orig_shape = run_yolo(
        yolo_model, str(image_path),
        conf=conf, iou=iou, imgsz=imgsz, device=yolo_device,
    )
    h, w = orig_shape
    result["detections"] = detections
    result["image_size"] = (w, h)

    print(f"Detected {len(detections)} objects:")
    for d in detections:
        cat = infer_category(d.label)
        angle_str = f", angle={d.angle_deg:.1f}°" if d.angle_deg is not None else ""
        print(f"  {d.label:18s} conf={d.confidence:.2f}  [{cat}]{angle_str}")

    # ── Step 3: Tidy scoring ──
    score_result = tidy_score(
        detections,
        image_size=(w, h),
        desk_orientation_deg=0.0,
        alignment_misalignment_threshold_deg=15.0,
    )
    result["tidy_score"] = score_result["tidy_score"]
    result["tidy_level"] = score_result["tidy_level"]
    result["penalties"] = score_result["penalties"]
    result["explanation"] = score_result["explanation"]

    print(f"\nTidy Score: {score_result['tidy_score']} ({score_result['tidy_level']})")
    print(f"Total Penalty: {score_result['total_penalty']}")
    for reason in score_result["explanation"]["reasons"]:
        print(f"  - {reason}")

    # ── Step 4: Language recommendations ──
    from desk_language_recommend import generate_language_recommendations

    rec = generate_language_recommendations(detections, score_result)
    result["recommendations"] = rec

    print(f"\nDecision: {rec['decision']}")
    print("Reasons:")
    for r in rec["reasons"]:
        print(f"  - {r}")
    print("Suggestions:")
    for s in rec["suggestions"]:
        print(f"  - {s}")

    # ── Step 5: Visual recommendations ──
    from desk_recommend import make_default_zones, plan_actions, draw_plan_image, draw_after_image
    from desk_relayout_viz import DeskRelayoutVisualizer

    zones = make_default_zones(w, h)
    plans = plan_actions(detections, zones)

    plan_path = str(output_dir / f"{stem}_plan.png")
    after_path = str(output_dir / f"{stem}_after.png")
    relayout_path = str(output_dir / f"{stem}_relayout.png")

    draw_plan_image(str(image_path), plans, zones, plan_path)
    draw_after_image(str(image_path), plans, zones, after_path)

    viz = DeskRelayoutVisualizer(str(image_path), detections)
    viz.generate(relayout_path)

    result["output_files"] = {
        "plan": plan_path,
        "after": after_path,
        "relayout": relayout_path,
    }

    print(f"\nOutput files:")
    for name, path in result["output_files"].items():
        print(f"  {name}: {path}")
    print(f"{'='*60}")

    return result


# ──────────────────────── CLI ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Unified Desk Tidier Pipeline")
    parser.add_argument("--image", type=str, required=True,
                        help="Image file or folder of images")
    parser.add_argument("--yolo-model", type=str, default=str(YOLO_MODEL),
                        help="Path to YOLO .pt model")
    parser.add_argument("--classifier-model", type=str, default=str(CLASSIFIER_MODEL),
                        help="Path to classifier .pth model")
    parser.add_argument("--skip-classifier", action="store_true",
                        help="Skip the binary classifier gating step")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference image size")
    parser.add_argument("--device", type=str, default="cpu",
                        help="YOLO device (cpu or 0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for generated images")
    args = parser.parse_args()

    # Load YOLO model
    yolo_path = Path(args.yolo_model)
    if not yolo_path.exists():
        print(f"ERROR: YOLO model not found at {yolo_path}")
        print("Please place best.pt at:")
        print(f"  {YOLO_MODEL}")
        print("Or specify --yolo-model <path>")
        sys.exit(1)

    yolo_model = YOLO(str(yolo_path))
    print(f"YOLO model loaded: {yolo_path.name}")
    print(f"  Classes: {list(yolo_model.names.values())}")

    # Load classifier (optional)
    classifier = None
    cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.skip_classifier:
        cls_path = Path(args.classifier_model)
        if cls_path.exists():
            classifier = load_classifier(cls_path, cls_device)
            print(f"Classifier loaded: {cls_path.name} (device: {cls_device})")
        else:
            print(f"WARNING: Classifier not found at {cls_path}, skipping classifier gating")

    # Collect images
    source = Path(args.image)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)
    elif source.exists():
        images = [source]
    else:
        print(f"ERROR: Image not found: {source}")
        sys.exit(1)

    if not images:
        print(f"ERROR: No images found at {source}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "pipeline_output"

    print(f"\nProcessing {len(images)} image(s)...")
    print(f"Config: conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
    print(f"Output: {output_dir}")

    all_results = []
    for img_path in images:
        result = process_image(
            image_path=img_path,
            yolo_model=yolo_model,
            classifier=classifier,
            cls_device=cls_device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            yolo_device=args.device,
            output_dir=output_dir,
        )
        all_results.append(result)

    # Summary
    if len(all_results) > 1:
        scored = [r for r in all_results if "tidy_score" in r]
        if scored:
            avg = sum(r["tidy_score"] for r in scored) / len(scored)
            print(f"\n{'='*60}")
            print(f"Summary: {len(images)} images, {len(scored)} scored")
            print(f"Average tidy score: {avg:.1f}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
