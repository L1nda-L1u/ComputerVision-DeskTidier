"""
Tidy Scoring System

Implements the "Desktop Tidy Scoring Framework" (0–100 score).

GitHub: https://github.com/L1nda-L1u/ComputerVision-DeskTidier.git
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GITHUB_REPO_URL = "https://github.com/L1nda-L1u/ComputerVision-DeskTidier.git"


# Matches the categories defined in `Scoring Framework.md`
CATEGORY_PENALTY_PER_OBJECT: Dict[str, int] = {
    "Core Work Items": 0,
    "Study Items": 0,
    "Temporary Items": 2,
    "Clutter Items": 6,
}

WORKSPACE_PENALTY_IF_IN_WORKSPACE: Dict[str, int] = {
    "Core Work Items": 0,
    "Study Items": 2,
    "Temporary Items": 6,
    "Clutter Items": 10,
}

# Framework section 6. "Penalty Rules" (alignment / orientation)
ALIGNMENT_PENALTY_PER_OBJECT: Dict[str, int] = {
    "Core Work Items": 8,
    "Study Items": 5,
    "Temporary Items": 3,
    "Clutter Items": 1,
}


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def central_workspace_rect(image_w: float, image_h: float) -> Tuple[float, float, float, float]:
    """
    Central 60% area of the desk image.

    We interpret "central 60%" as x in [0.2w, 0.8w] and y in [0.2h, 0.8h].
    """

    left = 0.2 * image_w
    right = 0.8 * image_w
    top = 0.2 * image_h
    bottom = 0.8 * image_h
    return left, top, right, bottom


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass(frozen=True)
class Detection:
    """
    Single object detection required by the scoring system.

    bbox: (x1, y1, x2, y2) in pixels
    angle_deg: optional. If provided, used for alignment penalty.
    """

    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    angle_deg: Optional[float] = None


def object_load_penalty(num_objects: int) -> int:
    """
    Table (Scoring Framework section 2).
    """

    if num_objects <= 8:
        return 0
    if 9 <= num_objects <= 12:
        return 5
    if 13 <= num_objects <= 15:
        return 10
    if 16 <= num_objects <= 18:
        return 15
    return 20


def infer_category(label: str) -> str:
    """
    Best-effort mapping from detection label to one of the scoring categories.

    The framework uses: laptop/mouse/book/notebook/phone/pen ... etc.
    For COCO-only labels, you may want to adjust this mapping.
    """

    l = label.strip().lower()

    core = {
        "laptop",
        "mouse",
        "keyboard",
        "book",
        "notebook",
        "clock",
        "remote",
    }
    study = {
        "eraser",
        "pen",
        "pencil",
        "cell phone",
        "earphones",
        "sticky note",
        "marker",
        "phone",
    }
    temporary = {
        "coffee cup",
        "mug",
        "cup",
        "bowl",
        "bottle",
        "scissors",
        "scissor",
        "tape",
    }
    clutter = {
        "cable",
        "spitball",
        "ring-pull can",
        "food packaging",
        "trash",
        "packaging",
        "food",
        "dining table",
    }

    if l in core:
        return "Core Work Items"
    if l in study:
        return "Study Items"
    if l in temporary:
        return "Temporary Items"
    if l in clutter:
        return "Clutter Items"

    # Unknown labels default to Temporary to avoid over-penalizing.
    return "Temporary Items"


def bbox_center_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def dispersion_level(detections: List[Detection], image_w: float, image_h: float) -> Tuple[str, int]:
    """
    Heuristic dispersion scoring (Framework section 5.2 "Object Dispersion").

    - Low: objects are clustered
    - Medium: moderate spread
    - High: widely scattered
    """

    if len(detections) <= 1:
        return "Low", 0

    centers = [bbox_center_xyxy(d.bbox) for d in detections]
    # Normalize by desk diagonal (so thresholds work across image sizes)
    diag = (image_w * image_w + image_h * image_h) ** 0.5
    if diag <= 0:
        return "Low", 0

    # Mean pairwise distance
    dists = []
    for (x1, y1), (x2, y2) in combinations(centers, 2):
        dx = x1 - x2
        dy = y1 - y2
        dists.append((dx * dx + dy * dy) ** 0.5)

    mean_dist = sum(dists) / len(dists)
    norm = mean_dist / diag

    # These thresholds are heuristic; adjust if you have a better dispersion metric.
    if norm < 0.18:
        return "Low", 0
    if norm < 0.35:
        return "Medium", 5
    return "High", 10


def tidy_level(score: int) -> str:
    """
    Framework section 7.
    """

    if 85 <= score <= 100:
        return "Tidy"
    if 70 <= score <= 84:
        return "Slightly Messy"
    if 50 <= score <= 69:
        return "Messy"
    return "Very Messy"


def tidy_score(
    detections: List[Detection],
    image_size: Tuple[int, int],
    desk_orientation_deg: float = 0.0,
    alignment_misalignment_threshold_deg: float = 15.0,
) -> Dict[str, Any]:
    """
    Compute the tidy score and an explanation dictionary.

    Alignment requires `angle_deg` per detection (if you can estimate orientation).
    If `angle_deg` is missing for an object, that object contributes 0 alignment penalty.
    """

    image_w, image_h = float(image_size[0]), float(image_size[1])
    num_objects = len(detections)

    # 2) Object Load Penalty
    load_penalty = object_load_penalty(num_objects)

    # 3) Category Penalty
    per_cat_counts: Dict[str, int] = {}
    category_penalty = 0
    for d in detections:
        cat = infer_category(d.label)
        per_cat_counts[cat] = per_cat_counts.get(cat, 0) + 1
        category_penalty += CATEGORY_PENALTY_PER_OBJECT[cat]

    # 4) Workspace Obstruction Penalty
    left, top, right, bottom = central_workspace_rect(image_w, image_h)
    workspace_penalty = 0
    workspace_blocked_objects = 0
    for d in detections:
        cx, cy = bbox_center_xyxy(d.bbox)
        if left <= cx <= right and top <= cy <= bottom:
            workspace_blocked_objects += 1
            cat = infer_category(d.label)
            workspace_penalty += WORKSPACE_PENALTY_IF_IN_WORKSPACE[cat]

    # 5.1) Object Overlap Penalty
    overlap_pairs = 0
    overlap_penalty = 0
    for a, b in combinations(detections, 2):
        if iou_xyxy(a.bbox, b.bbox) > 0.3:
            overlap_pairs += 1
    overlap_penalty = min(10, overlap_pairs * 2)

    # 5.2) Object Dispersion Penalty
    dispersion_label, dispersion_penalty = dispersion_level(detections, image_w, image_h)

    # 6) Alignment Penalty
    alignment_penalty = 0
    misaligned_count = 0
    for d in detections:
        if d.angle_deg is None:
            continue
        diff = abs((d.angle_deg - desk_orientation_deg + 180) % 360 - 180)  # shortest angle distance
        if diff > alignment_misalignment_threshold_deg:
            misaligned_count += 1
            alignment_penalty += ALIGNMENT_PENALTY_PER_OBJECT[infer_category(d.label)]

    # Total
    total_penalty = (
        load_penalty
        + category_penalty
        + workspace_penalty
        + overlap_penalty
        + dispersion_penalty
        + alignment_penalty
    )

    score = int(round(clamp(100 - total_penalty, 0, 100)))

    # Explanation output (Framework section 8)
    reasons: List[str] = []
    reasons.append(f"{num_objects} objects detected -> object load penalty {load_penalty}")
    if per_cat_counts:
        cat_str = ", ".join([f"{k.split()[0]}={v}" for k, v in sorted(per_cat_counts.items())])
        reasons.append(f"Category counts: {cat_str}; category penalty {category_penalty}")
    reasons.append(f"{workspace_blocked_objects} objects in central workspace -> workspace penalty {workspace_penalty}")
    reasons.append(f"Overlap pairs (IoU>0.30): {overlap_pairs} -> overlap penalty {overlap_penalty}")
    reasons.append(f"Dispersion: {dispersion_label} -> dispersion penalty {dispersion_penalty}")
    if misaligned_count > 0:
        reasons.append(f"Misaligned objects: {misaligned_count} -> alignment penalty {alignment_penalty}")
    else:
        reasons.append("Alignment penalty: 0 (no angles available or all aligned)")

    return {
        "github_repo": GITHUB_REPO_URL,
        "tidy_score": score,
        "tidy_level": tidy_level(score),
        "total_penalty": total_penalty,
        "penalties": {
            "object_load_penalty": load_penalty,
            "category_penalty": category_penalty,
            "workspace_obstruction_penalty": workspace_penalty,
            "spatial_overlap_penalty": overlap_penalty,
            "spatial_dispersion_penalty": dispersion_penalty,
            "alignment_penalty": alignment_penalty,
        },
        "explanation": {
            "reasons": reasons,
            "workspace_center_rect": (left, top, right, bottom),
            "dispersion_label": dispersion_label,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Desktop Tidy Score.")
    parser.add_argument("--model", type=str, default=None, help="YOLO model path, e.g. best.pt")
    parser.add_argument("--source", type=str, default=None, help="Image file or folder of images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold for NMS (e.g. 0.5). If omitted, Ultralytics default is used.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="cpu", help="e.g. cpu or 0")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated prediction images (with bounding boxes).",
    )
    parser.add_argument("--save-project", type=str, default="tidy_score_predicts", help="Ultralytics project dir")
    parser.add_argument("--save-name", type=str, default="desk_065", help="Ultralytics run name")
    parser.add_argument("--desk-orientation", type=float, default=0.0, help="Desk orientation angle (deg)")
    parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=15.0,
        help="Misalignment threshold angle (deg) for applying alignment penalty",
    )
    args = parser.parse_args()

    # If no YOLO args are provided, run the built-in demo.
    if not args.model or not args.source:
        demo_image_size = (640, 360)
        demo_detections = [
            Detection(label="laptop", confidence=0.92, bbox=(250, 140, 430, 250), angle_deg=0.0),
            Detection(label="mouse", confidence=0.80, bbox=(440, 180, 490, 220), angle_deg=2.0),
            Detection(label="bowl", confidence=0.65, bbox=(60, 170, 140, 250), angle_deg=18.0),
            Detection(label="cup", confidence=0.55, bbox=(300, 60, 350, 120), angle_deg=25.0),
        ]

        result = tidy_score(
            demo_detections,
            image_size=demo_image_size,
            desk_orientation_deg=0.0,
            alignment_misalignment_threshold_deg=15.0,
        )

        print(f"Tidy Score: {result['tidy_score']}")
        print(f"Tidy Level: {result['tidy_level']}")
        print("\nReasons:")
        for r in result["explanation"]["reasons"]:
            print(f"- {r}")
        raise SystemExit(0)

    # Otherwise, use YOLO predictions to build Detection objects and score them.
    from ultralytics import YOLO  # local import so demo still works without ultralytics

    model = YOLO(args.model)
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"--source not found: {source_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source_path.is_dir():
        images = sorted([p for p in source_path.rglob("*") if p.suffix.lower() in exts])
    else:
        images = [source_path]

    if not images:
        raise FileNotFoundError(f"No images found under: {source_path}")

    scores: List[int] = []
    for img_path in images:
        predict_kwargs: Dict[str, Any] = {
            "source": str(img_path),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "device": args.device,
            "verbose": False,
            "save": bool(args.save),
            "save_txt": False,
        }
        if args.iou is not None:
            predict_kwargs["iou"] = float(args.iou)
        if args.save:
            predict_kwargs["project"] = str(args.save_project)
            predict_kwargs["name"] = str(args.save_name)
            predict_kwargs["exist_ok"] = True

        preds = model.predict(
            **predict_kwargs,
        )
        r = preds[0]
        names = getattr(r, "names", None) or getattr(model, "names", None) or {}

        detections: List[Detection] = []
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            # ultralytics boxes: xyxy, conf, cls
            for xyxy, conf, cls_id in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                cls_id_int = int(cls_id)
                label = names.get(cls_id_int, str(cls_id_int))
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(conf),
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        angle_deg=None,  # alignment angle is not available from YOLO detection alone
                    )
                )

        # ultralytics gives orig_shape as (h, w)
        h, w = r.orig_shape
        res = tidy_score(
            detections,
            image_size=(w, h),
            desk_orientation_deg=float(args.desk_orientation),
            alignment_misalignment_threshold_deg=float(args.alignment_threshold),
        )
        scores.append(res["tidy_score"])

        print(
            f"{img_path.name}: tidy_score={res['tidy_score']} "
            f"({res['tidy_level']}) total_penalty={res['total_penalty']}"
        )
        # Print reasons for transparency (can be noisy for many images)
        for reason in res["explanation"]["reasons"][:4]:
            print(f"  - {reason}")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nSummary: {len(scores)} images, avg_tidy_score={avg:.2f}")