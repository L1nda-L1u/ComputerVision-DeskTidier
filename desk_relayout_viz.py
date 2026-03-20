"""
DeskRelayoutVisualizer — Reorganised Layout Image Generator

Creates a clean 1400x900 canvas showing where each detected object
should be placed after tidying.  Objects are cropped from the original
photo and arranged in a neat grid inside functional zones.

Usage (standalone):
    python desk_relayout_viz.py --image jpg_images/desk_065.jpg

Integration:
    from desk_relayout_viz import DeskRelayoutVisualizer
    viz = DeskRelayoutVisualizer(original_image_path, detections)
    viz.generate("desk_relayout.png")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────── Zone / label mapping ────────────────────────────────

ZONE_ASSIGNMENT: Dict[str, str] = {
    "laptop":       "workspace",
    "mouse":        "workspace",
    "keyboard":     "workspace",
    "clock":        "workspace",
    "remote":       "workspace",

    "book":         "reference",
    "notebook":     "reference",
    "sticky note":  "reference",
    "paper":        "reference",

    "pen":          "stationery",
    "pencil":       "stationery",
    "marker":       "stationery",
    "eraser":       "stationery",
    "scissors":     "stationery",
    "scissor":      "stationery",
    "tape":         "stationery",
    "phone":        "stationery",
    "cell phone":   "stationery",
    "earphones":    "stationery",

    "cup":          "temporary",
    "mug":          "temporary",
    "coffee cup":   "temporary",
    "bowl":         "temporary",
    "bottle":       "temporary",
    "ring-pull can": "temporary",
    "cable":        "temporary",
    "charger":      "temporary",

    "spitball":       "remove",
    "food packaging": "remove",
    "trash":          "remove",
    "packaging":      "remove",
    "food":           "remove",
    "dining table":   "remove",
}

ZONE_COLORS_RGB = {
    "stationery": (230, 240, 255),
    "reference":  (235, 255, 235),
    "temporary":  (255, 245, 225),
    "workspace":  (240, 248, 255),
    "remove":     (245, 240, 240),
}

ZONE_BORDER_RGB = {
    "stationery": (140, 170, 220),
    "reference":  (130, 190, 130),
    "temporary":  (210, 170, 100),
    "workspace":  (120, 160, 200),
    "remove":     (180, 150, 150),
}

ZONE_DISPLAY_NAMES = {
    "stationery": "Stationery Zone",
    "reference":  "Reference Zone",
    "temporary":  "Temporary Zone",
    "workspace":  "Workspace Zone",
    "remove":     "Removed Items",
}

CANVAS_W, CANVAS_H = 1400, 900
BG_COLOR = (245, 245, 245)

# Zone rectangles: (x, y, w, h) on the 1400x900 canvas
ZONE_RECTS: Dict[str, Tuple[int, int, int, int]] = {
    "stationery": (30,  100, 260, 580),
    "reference":  (320, 30,  760, 180),
    "workspace":  (320, 240, 760, 440),
    "temporary":  (1110, 100, 260, 580),
    "remove":     (320, 720, 380, 150),
}


# ──────────────────────── Helper: Detection ───────────────────────────────────

@dataclass
class _Item:
    label: str
    bbox: Tuple[int, int, int, int]
    zone: str
    patch: Optional[np.ndarray] = None


# ──────────────────────── Grid layout computation ─────────────────────────────

def _grid_layout(n: int, zone_w: int, zone_h: int) -> Tuple[int, int]:
    """Choose cols x rows for n items that best fits the zone aspect ratio."""
    if n == 0:
        return 1, 1
    if n == 1:
        return 1, 1
    if n == 2:
        return (2, 1) if zone_w >= zone_h else (1, 2)
    if n == 3:
        return (3, 1) if zone_w >= zone_h * 1.5 else (1, 3) if zone_h >= zone_w * 1.5 else (3, 1)

    aspect = zone_w / max(zone_h, 1)
    best_cols, best_rows = 1, n
    best_waste = float("inf")
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w = zone_w / cols
        cell_h = zone_h / rows
        cell_aspect = cell_w / max(cell_h, 1)
        waste = abs(cell_aspect - 1.0) + 0.1 * (cols * rows - n)
        if waste < best_waste:
            best_waste = waste
            best_cols, best_rows = cols, rows
    return best_cols, best_rows


def _slot_centers(
    zone_rect: Tuple[int, int, int, int],
    n: int,
    header_h: int = 30,
    padding: int = 12,
) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (cx, cy, slot_w, slot_h) for n items inside a zone,
    leaving space for the zone header label.
    """
    zx, zy, zw, zh = zone_rect
    inner_x = zx + padding
    inner_y = zy + header_h + padding
    inner_w = zw - 2 * padding
    inner_h = zh - header_h - 2 * padding

    if n == 0:
        return []

    cols, rows = _grid_layout(n, inner_w, inner_h)
    cell_w = inner_w / cols
    cell_h = inner_h / rows
    gap = 6

    slots = []
    for i in range(n):
        r, c = divmod(i, cols)
        cx = int(inner_x + (c + 0.5) * cell_w)
        cy = int(inner_y + (r + 0.5) * cell_h)
        sw = int(cell_w - gap * 2)
        sh = int(cell_h - gap * 2)
        slots.append((cx, cy, max(sw, 20), max(sh, 20)))
    return slots


# ──────────────────────── Crop, resize, paste ─────────────────────────────────

def _crop_patch(img: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 10) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    return img[y1:y2, x1:x2].copy()


def _resize_keep_aspect(patch: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    ph, pw = patch.shape[:2]
    if pw == 0 or ph == 0:
        return patch
    scale = min(max_w / pw, max_h / ph, 1.0)
    nw, nh = max(1, int(pw * scale)), max(1, int(ph * scale))
    return cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)


def _paste_center(canvas: np.ndarray, patch: np.ndarray, cx: int, cy: int) -> None:
    ph, pw = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    x1 = cx - pw // 2
    y1 = cy - ph // 2
    x2 = x1 + pw
    y2 = y1 + ph

    # Clip to canvas bounds
    sx = max(0, -x1)
    sy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(cw, x2)
    y2 = min(ch, y2)
    ex = sx + (x2 - x1)
    ey = sy + (y2 - y1)

    if x2 > x1 and y2 > y1 and ex <= pw and ey <= ph:
        canvas[y1:y2, x1:x2] = patch[sy:ey, sx:ex]


# ──────────────────────── Drawing helpers ─────────────────────────────────────

def _draw_rounded_rect(
    img: np.ndarray,
    rect: Tuple[int, int, int, int],
    fill_rgb: Tuple[int, int, int],
    border_rgb: Tuple[int, int, int],
    radius: int = 16,
    alpha: float = 0.5,
) -> None:
    x, y, w, h = rect
    fill_bgr = fill_rgb[::-1]
    border_bgr = border_rgb[::-1]

    overlay = img.copy()

    # Filled rounded rect via 3 rects + 4 circles
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), fill_bgr, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), fill_bgr, -1)
    for (cx, cy) in [(x + radius, y + radius), (x + w - radius, y + radius),
                      (x + radius, y + h - radius), (x + w - radius, y + h - radius)]:
        cv2.circle(overlay, (cx, cy), radius, fill_bgr, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Border (thin rounded)
    pts_top = [(x + radius, y), (x + w - radius, y)]
    pts_bot = [(x + radius, y + h), (x + w - radius, y + h)]
    pts_lft = [(x, y + radius), (x, y + h - radius)]
    pts_rgt = [(x + w, y + radius), (x + w, y + h - radius)]
    for p1, p2 in [pts_top, pts_bot, pts_lft, pts_rgt]:
        cv2.line(img, p1, p2, border_bgr, 2)
    for (cx, cy), sa, ea in [
        ((x + radius, y + radius), 180, 270),
        ((x + w - radius, y + radius), 270, 360),
        ((x + radius, y + h - radius), 90, 180),
        ((x + w - radius, y + h - radius), 0, 90),
    ]:
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, sa, ea, border_bgr, 2)


def _put_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color_rgb: Tuple[int, int, int] = (80, 80, 80),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color_rgb[::-1], thickness, cv2.LINE_AA)


# ──────────────────────── Main class ──────────────────────────────────────────

class DeskRelayoutVisualizer:
    """
    Generate a reorganised desk layout image.

    Parameters
    ----------
    image_path : str
        Path to the original desk photo.
    detections : list
        Detection objects with .label and .bbox (x1, y1, x2, y2).
    """

    def __init__(self, image_path: str, detections: list):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        self.items: List[_Item] = []
        for d in detections:
            label = d.label.strip().lower()
            zone = ZONE_ASSIGNMENT.get(label, "temporary")
            bbox = tuple(int(v) for v in d.bbox)
            patch = _crop_patch(self.original, bbox)
            self.items.append(_Item(label=label, bbox=bbox, zone=zone, patch=patch))

    def generate(self, out_path: str) -> str:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR[::-1], dtype=np.uint8)

        # Title
        _put_text(canvas, "Reorganised Desk Layout", (30, 40),
                  color_rgb=(50, 50, 50), scale=0.9, thickness=2)
        _put_text(canvas, f"Source: {Path(self.image_path).name}", (30, 70),
                  color_rgb=(140, 140, 140), scale=0.45)

        # Group items by zone
        zone_items: Dict[str, List[_Item]] = {z: [] for z in ZONE_RECTS}
        for item in self.items:
            zone_items.setdefault(item.zone, []).append(item)

        # Draw zones
        for zname, zrect in ZONE_RECTS.items():
            fill = ZONE_COLORS_RGB.get(zname, (240, 240, 240))
            border = ZONE_BORDER_RGB.get(zname, (180, 180, 180))
            _draw_rounded_rect(canvas, zrect, fill, border, radius=14, alpha=0.45)

            zx, zy, zw, zh = zrect
            display = ZONE_DISPLAY_NAMES.get(zname, zname)
            count = len(zone_items.get(zname, []))
            label_text = f"{display} ({count})" if count else display
            _put_text(canvas, label_text, (zx + 12, zy + 22), color_rgb=border, scale=0.5, thickness=1)

        # Place object patches inside zones
        for zname, items in zone_items.items():
            if zname == "remove" or not items:
                continue

            zrect = ZONE_RECTS[zname]
            slots = _slot_centers(zrect, len(items))

            for item, (cx, cy, sw, sh) in zip(items, slots):
                if item.patch is None or item.patch.size == 0:
                    continue
                resized = _resize_keep_aspect(item.patch, sw, sh)
                _paste_center(canvas, resized, cx, cy)

                # Small label below the patch
                ph = resized.shape[0]
                label_y = cy + ph // 2 + 14
                _put_text(canvas, item.label, (cx - 20, min(label_y, CANVAS_H - 5)),
                          color_rgb=(100, 100, 100), scale=0.38, thickness=1)

        # Removed items section
        removed = zone_items.get("remove", [])
        if removed:
            zrect = ZONE_RECTS["remove"]
            zx, zy, zw, zh = zrect

            # Small "trash" icon (text-based)
            _put_text(canvas, "Items removed from desk:", (zx + 12, zy + 50),
                      color_rgb=(160, 120, 120), scale=0.42)

            removed_labels = [item.label for item in removed]
            text = ", ".join(removed_labels)
            _put_text(canvas, text, (zx + 12, zy + 75),
                      color_rgb=(140, 140, 140), scale=0.4)

            # Optionally show tiny thumbnails
            thumb_x = zx + 12
            for item in removed[:5]:
                if item.patch is not None and item.patch.size > 0:
                    thumb = _resize_keep_aspect(item.patch, 50, 50)
                    th, tw = thumb.shape[:2]
                    _paste_center(canvas, thumb, thumb_x + tw // 2, zy + 115)
                    thumb_x += tw + 8

        # Stats footer
        total = len(self.items)
        kept = sum(1 for i in self.items if i.zone != "remove")
        removed_n = total - kept
        _put_text(canvas, f"Total: {total} objects | Placed: {kept} | Removed: {removed_n}",
                  (730, CANVAS_H - 20), color_rgb=(160, 160, 160), scale=0.42)

        cv2.imwrite(out_path, canvas)
        print(f"Relayout image saved: {out_path}")
        return out_path


# ──────────────────────── Standalone runner ────────────────────────────────────

_DEFAULT_YOLO = str(Path(__file__).resolve().parent / "runs" / "detect" / "desk_tidy_runs"
                     / "v4_yolov8m_roboflow_style" / "weights" / "best.pt")


def run_standalone(image_path: str, model_path: str = _DEFAULT_YOLO, conf: float = 0.4):
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)
    r = results[0]
    names = getattr(r, "names", None) or {}

    @dataclass
    class Det:
        label: str
        confidence: float
        bbox: Tuple[float, float, float, float]

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for xyxy, c, cls_id in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            detections.append(Det(
                label=names.get(int(cls_id), str(int(cls_id))),
                confidence=float(c),
                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
            ))

    print(f"Detected {len(detections)} objects")
    for d in detections:
        zone = ZONE_ASSIGNMENT.get(d.label.strip().lower(), "temporary")
        print(f"  {d.label:15s} -> {zone}")

    viz = DeskRelayoutVisualizer(image_path, detections)
    stem = Path(image_path).stem
    viz.generate(f"{stem}_relayout.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate reorganised desk layout")
    parser.add_argument("--image", type=str, default="jpg_images/desk_065.jpg")
    parser.add_argument("--model", type=str, default=_DEFAULT_YOLO)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    run_standalone(args.image, args.model, args.conf)
