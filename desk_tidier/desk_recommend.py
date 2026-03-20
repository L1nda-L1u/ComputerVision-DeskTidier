"""
Desk Recommendation Visualizations

Generates two output images after YOLO detection + tidy scoring:
  1. Rearrangement Plan  — arrows showing where objects should move
  2. Organised Result    — clean "after" layout diagram

Reuses the Detection dataclass from "Tidy Scoring System.py".

Usage (standalone):
    python -m desk_tidier.desk_recommend --image data/images/desk_065.jpg

Integration (in your pipeline):
    from desk_recommend import make_default_zones, plan_actions
    from desk_recommend import draw_plan_image, draw_after_image

    zones = make_default_zones(w, h)
    plans = plan_actions(detections, zones)
    draw_plan_image(img_path, plans, zones, "desk_move_plan.png")
    draw_after_image(img_path, plans, zones, "desk_after_layout.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Re-export Detection so callers don't need a second import
try:
    import importlib.util
    import sys
    from pathlib import Path as _Path

    _scoring_path = _Path(__file__).resolve().parents[1] / "scoring_module" / "scripts" / "Tidy Scoring System.py"
    _spec = importlib.util.spec_from_file_location("tidy_scoring_system", str(_scoring_path))
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Failed to load scoring module")
    _scoring = importlib.util.module_from_spec(_spec)
    sys.modules["tidy_scoring_system"] = _scoring
    _spec.loader.exec_module(_scoring)
    Detection = _scoring.Detection
    infer_category = _scoring.infer_category
except Exception:
    @dataclass(frozen=True)
    class Detection:
        label: str
        confidence: float
        bbox: Tuple[float, float, float, float]
        angle_deg: Optional[float] = None

    def infer_category(label: str) -> str:
        return "Temporary Items"


# ───────────────────────── Zone colours (pastel, BGR) ─────────────────────────
ZONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "workspace":  (235, 206, 135),   # pastel blue
    "reference":  (180, 229, 180),   # pastel green
    "stationery": (200, 200, 245),   # pastel pink
    "temporary":  (180, 235, 255),   # pastel yellow
    "remove":     (190, 190, 210),   # pastel grey
}

ZONE_LABELS = {
    "workspace":  "Workspace",
    "reference":  "Reference",
    "stationery": "Stationery",
    "temporary":  "Temporary",
    "remove":     "Remove",
}

ACTION_COLORS = {
    "keep":   (80, 180, 80),    # green
    "move":   (50, 165, 255),   # orange
    "remove": (80, 80, 220),    # red
}


# ──────────────────────────── Helper functions ────────────────────────────────

def point_in_rect(px: float, py: float, rect: Tuple[int, int, int, int]) -> bool:
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def rect_center(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def draw_transparent_rect(
    img: np.ndarray,
    rect: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.25,
) -> None:
    x, y, w, h = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def draw_label_card(
    img: np.ndarray,
    text: str,
    center: Tuple[int, int],
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (40, 40, 40),
    font_scale: float = 0.5,
) -> None:
    """Draw a rounded card with text at the given center."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    pad = 8
    x1 = center[0] - tw // 2 - pad
    y1 = center[1] - th // 2 - pad
    x2 = center[0] + tw // 2 + pad
    y2 = center[1] + th // 2 + pad

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), tuple(max(0, c - 40) for c in bg_color), 1)

    tx = center[0] - tw // 2
    ty = center[1] + th // 2
    cv2.putText(img, text, (tx, ty), font, font_scale, text_color, 1, cv2.LINE_AA)


def zone_slots(
    rect: Tuple[int, int, int, int], n: int, cols: int = 3
) -> List[Tuple[int, int]]:
    """Compute grid positions for n label cards inside a zone rectangle."""
    x, y, w, h = rect
    rows = max(1, (n + cols - 1) // cols)
    positions = []
    for i in range(n):
        r, c = divmod(i, cols)
        cx = x + int((c + 0.5) * w / cols)
        cy = y + 28 + int((r + 0.5) * (h - 36) / rows)
        positions.append((cx, cy))
    return positions


# ──────────────────────── 1. Zone definitions ─────────────────────────────────

def make_default_zones(w: int, h: int) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Default desk zones for a right-handed study desk (top-down view).
    Returns {zone_name: (x, y, width, height)}.

    Layout:
    ┌──────────┬─────────────────────┬──────────┐
    │reference │     workspace       │stationery│
    │  (left)  │     (center)        │  (right) │
    ├──────────┼─────────────────────┼──────────┤
    │          │     temporary       │  remove  │
    └──────────┴─────────────────────┴──────────┘
    """
    margin = int(0.03 * min(w, h))
    left_w = int(w * 0.2)
    right_w = int(w * 0.18)
    center_w = w - left_w - right_w - margin * 4
    top_h = int(h * 0.62)
    bot_h = h - top_h - margin * 3

    return {
        "reference":  (margin, margin, left_w, top_h),
        "workspace":  (margin + left_w + margin, margin, center_w, top_h),
        "stationery": (w - right_w - margin, margin, right_w, top_h),
        "temporary":  (margin + left_w + margin, margin + top_h + margin, center_w, bot_h),
        "remove":     (w - right_w - margin, margin + top_h + margin, right_w, bot_h),
    }


# ──────────────────────── 2. Label → zone mapping ────────────────────────────

def target_zone_for_label(label: str) -> str:
    l = label.strip().lower()

    workspace = {"laptop", "mouse", "keyboard", "clock", "remote"}
    reference = {"book", "notebook", "sticky note", "paper"}
    stationery = {
        "pen", "pencil", "marker", "eraser", "scissors", "scissor",
        "tape", "phone", "cell phone", "earphones",
    }
    temporary = {"cup", "mug", "bowl", "bottle", "cable", "charger"}
    remove = {"spitball", "food packaging", "trash", "packaging", "food", "dining table", "ring-pull can"}

    if l in workspace:
        return "workspace"
    if l in reference:
        return "reference"
    if l in stationery:
        return "stationery"
    if l in temporary:
        return "temporary"
    if l in remove:
        return "remove"
    return "temporary"


# ──────────────────────── 3. Plan actions ─────────────────────────────────────

def plan_actions(
    detections: List[Detection],
    zones: Dict[str, Tuple[int, int, int, int]],
) -> List[Dict[str, Any]]:
    """
    For each detection, decide: keep / move / remove.
    Returns list of action dicts.
    """
    plans = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        target = target_zone_for_label(d.label)
        target_pt = rect_center(zones[target])

        if target == "remove":
            action = "remove"
        elif point_in_rect(cx, cy, zones[target]):
            action = "keep"
        else:
            action = "move"

        plans.append({
            "label": d.label,
            "confidence": d.confidence,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": (int(cx), int(cy)),
            "target_zone": target,
            "target_point": target_pt,
            "action": action,
        })
    return plans


# ──────────────────────── 4. Plan image ───────────────────────────────────────

def draw_plan_image(
    image_path: str,
    plans: List[Dict[str, Any]],
    zones: Dict[str, Tuple[int, int, int, int]],
    out_path: str,
) -> None:
    """Draw rearrangement plan: zones + arrows + labels on the original image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    # Draw zones
    for zname, zrect in zones.items():
        color = ZONE_COLORS.get(zname, (200, 200, 200))
        draw_transparent_rect(img, zrect, color, alpha=0.20)
        zx, zy, zw, zh = zrect
        cv2.putText(img, ZONE_LABELS.get(zname, zname),
                    (zx + 8, zy + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)

    # Draw each plan
    for p in plans:
        x1, y1, x2, y2 = p["bbox"]
        cx, cy = p["center"]
        action = p["action"]
        color = ACTION_COLORS[action]
        label = p["label"]

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if action == "keep":
            tag = f"{label}: keep"
            cv2.putText(img, tag, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

        elif action == "move":
            tx, ty = p["target_point"]
            cv2.arrowedLine(img, (cx, cy), (tx, ty), color, 2, tipLength=0.04)
            tag = f"{label} -> {p['target_zone']}"
            mid_x, mid_y = (cx + tx) // 2, (cy + ty) // 2
            cv2.putText(img, tag, (mid_x + 5, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        elif action == "remove":
            tx, ty = p["target_point"]
            cv2.arrowedLine(img, (cx, cy), (tx, ty), color, 2, tipLength=0.04)
            tag = f"{label} -> remove"
            cv2.putText(img, tag, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Legend
    ly = h - 70
    for i, (act, col) in enumerate(ACTION_COLORS.items()):
        lx = 20 + i * 160
        cv2.rectangle(img, (lx, ly), (lx + 14, ly + 14), col, -1)
        cv2.putText(img, act.upper(), (lx + 20, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    print(f"Plan image saved: {out_path}")


# ──────────────────────── 5. After image ──────────────────────────────────────

def draw_after_image(
    image_path: str,
    plans: List[Dict[str, Any]],
    zones: Dict[str, Tuple[int, int, int, int]],
    out_path: str,
) -> None:
    """Draw organised result: faded background + zone cards for kept/moved objects."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Fade background
    white = np.full_like(img, 240)
    img = cv2.addWeighted(img, 0.3, white, 0.7, 0)

    h, w = img.shape[:2]

    # Draw zones
    for zname, zrect in zones.items():
        color = ZONE_COLORS.get(zname, (200, 200, 200))
        draw_transparent_rect(img, zrect, color, alpha=0.30)
        zx, zy, zw, zh = zrect
        cv2.putText(img, ZONE_LABELS.get(zname, zname),
                    (zx + 8, zy + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, tuple(max(0, c - 60) for c in color), 2, cv2.LINE_AA)

    # Group objects by target zone (exclude removed)
    zone_objects: Dict[str, List[str]] = {z: [] for z in zones}
    removed_items: List[str] = []

    for p in plans:
        if p["action"] == "remove":
            removed_items.append(p["label"])
        else:
            zone_objects[p["target_zone"]].append(p["label"])

    # Place label cards inside each zone
    for zname, items in zone_objects.items():
        if not items:
            continue
        zrect = zones[zname]
        slots = zone_slots(zrect, len(items), cols=2 if zrect[2] < 300 else 3)
        color = ZONE_COLORS.get(zname, (220, 220, 220))
        for i, item_label in enumerate(items):
            if i < len(slots):
                draw_label_card(img, item_label, slots[i], bg_color=color)

    # Removed items note
    if removed_items:
        note = f"Off-desk: {', '.join(removed_items)}"
        cv2.putText(img, note, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 100, 100), 1, cv2.LINE_AA)

    # Title
    cv2.putText(img, "Organised Desk Layout", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    print(f"After image saved: {out_path}")


# ──────────────────────── Standalone runner ────────────────────────────────────

_DEFAULT_YOLO = str(Path(__file__).resolve().parent / "runs" / "detect" / "desk_tidy_runs"
                     / "v4_yolov8m_roboflow_style" / "weights" / "best.pt")


def run_standalone(image_path: str, model_path: str = _DEFAULT_YOLO, conf: float = 0.4):
    """Run YOLO detection → plan → draw both images."""
    from ultralytics import YOLO

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    h, w = img.shape[:2]

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False, agnostic_nms=True)
    r = results[0]
    names = getattr(r, "names", None) or {}

    detections: List[Detection] = []
    if r.boxes is not None and len(r.boxes) > 0:
        for xyxy, c, cls_id in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            label = names.get(int(cls_id), str(int(cls_id)))
            detections.append(Detection(
                label=label, confidence=float(c),
                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
            ))

    print(f"Detected {len(detections)} objects: {[d.label for d in detections]}")

    zones = make_default_zones(w, h)
    plans = plan_actions(detections, zones)

    for p in plans:
        print(f"  {p['label']:15s} -> {p['action']:6s} (target: {p['target_zone']})")

    stem = Path(image_path).stem
    draw_plan_image(image_path, plans, zones, f"{stem}_plan.png")
    draw_after_image(image_path, plans, zones, f"{stem}_after.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Desk recommendation visualization")
    parser.add_argument("--image", type=str, default="jpg_images/desk_065.jpg")
    parser.add_argument("--model", type=str, default=_DEFAULT_YOLO)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    run_standalone(args.image, args.model, args.conf)
