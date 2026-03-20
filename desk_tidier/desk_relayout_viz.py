"""
DeskRelayoutVisualizer — Reorganised Layout Image Generator

Creates a clean canvas showing where each detected object should be placed
after tidying.  Objects are:
  - Segmented from the background using SAM2 (pixel-accurate masks)
  - Rotated to be aligned with desk edges
  - Placed at proportionally correct sizes (relative to each other)

Usage (standalone):
    python desk_relayout_viz.py --image jpg_images/desk_065.jpg
    python desk_relayout_viz.py --image jpg_images/desk_065.jpg --left
        # --left: left-handed layout (stationery right, temporary left, mouse bottom-left)

Integration:
    from desk_relayout_viz import DeskRelayoutVisualizer
    viz = DeskRelayoutVisualizer(original_image_path, detections, sam_masks=masks)
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

    "book":         "workspace",
    "notebook":     "workspace",
    "sticky note":  "reference",
    "paper":        "reference",

    "pen":          "stationery",
    "pencil":       "stationery",
    "marker":       "stationery",
    "eraser":       "stationery",
    "scissors":     "reference",
    "scissor":      "reference",
    "tape":         "stationery",
    "phone":        "stationery",
    "cell phone":   "stationery",
    "earphones":    "stationery",

    "cup":          "temporary",
    "mug":          "temporary",
    "coffee cup":   "temporary",
    "bowl":         "temporary",
    "bottle":       "temporary",
    "ring-pull can": "remove",
    "cable":        "temporary",
    "charger":      "temporary",

    "spitball":       "remove",
    "food packaging": "remove",
    "trash":          "remove",
    "packaging":      "remove",
    "food":           "remove",
    "dining table":   "remove",
    "battery":        "remove",
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

CANVAS_W, CANVAS_H = 3600, 2200
BG_COLOR = (255, 255, 255)

ZONE_RECTS: Dict[str, Tuple[int, int, int, int]] = {
    "stationery": (30,  80, 750, 2080),
    "reference":  (810, 80, 1950, 450),
    "workspace":  (810, 560, 1950, 1000),
    "temporary":  (2790, 80, 780, 2080),
    "remove":     (810, 1590, 1250, 500),
}

MOUSE_AREA_RECT = (2090, 1590, 670, 500)


def _mirror_rect_h(
    x: int, y: int, w: int, h: int, canvas_w: int = CANVAS_W
) -> Tuple[int, int, int, int]:
    """Mirror a rect horizontally (left↔right) for left-handed desk layout."""
    return (canvas_w - x - w, y, w, h)


# Left-handed layout: stationery on the right, temporary on the left, mouse bottom-left of workspace row
ZONE_RECTS_LEFT: Dict[str, Tuple[int, int, int, int]] = {
    k: _mirror_rect_h(*v) for k, v in ZONE_RECTS.items()
}
MOUSE_AREA_RECT_LEFT = _mirror_rect_h(*MOUSE_AREA_RECT)


# ──────────────────────── Item dataclass ──────────────────────────────────────

@dataclass
class _Item:
    label: str
    bbox: Tuple[int, int, int, int]
    zone: str
    angle_deg: Optional[float] = None
    orig_w: int = 0
    orig_h: int = 0
    patch_bgra: Optional[np.ndarray] = None


# ──────────────────────── SAM-based segmentation ──────────────────────────────

def _smooth_mask(mask_uint8: np.ndarray, blur_radius: int = 7) -> np.ndarray:
    """
    Smooth mask edges with Gaussian blur for anti-aliased cutouts.
    Also applies slight morphological closing to fill small holes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Gaussian blur on the edges only (dilated - eroded region)
    dilated = cv2.dilate(smoothed, kernel, iterations=1)
    eroded = cv2.erode(smoothed, kernel, iterations=1)
    edge_band = dilated - eroded

    blurred = cv2.GaussianBlur(smoothed, (blur_radius, blur_radius), 0)

    result = smoothed.copy()
    edge_mask = edge_band > 0
    result[edge_mask] = blurred[edge_mask]

    return result


def _crop_with_mask(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask: Optional[np.ndarray] = None,
    pad: int = 6,
) -> Tuple[np.ndarray, int, int]:
    """
    Crop the bbox region and apply SAM mask to create a BGRA patch with
    smooth transparent edges. If no mask is provided, falls back to opaque crop.
    """
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x1c = max(0, x1 - pad)
    y1c = max(0, y1 - pad)
    x2c = min(iw, x2 + pad)
    y2c = min(ih, y2 + pad)

    crop = img[y1c:y2c, x1c:x2c].copy()
    ch, cw = crop.shape[:2]

    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)

    if mask is not None:
        crop_mask = mask[y1c:y2c, x1c:x2c]
        alpha = (crop_mask * 255).astype(np.uint8)
        alpha = _smooth_mask(alpha)
        bgra[:, :, 3] = alpha
    else:
        bgra[:, :, 3] = 255

    return bgra, cw, ch


# ──────────────────────── Rotation ────────────────────────────────────────────

def _angle_from_mask(alpha: np.ndarray) -> Optional[float]:
    """Estimate orientation angle from the alpha channel using minAreaRect."""
    binary = (alpha > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 100:
        return None
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    angle = float(rect[2])
    if w < h:
        angle += 90.0
    angle = ((angle + 90.0) % 180.0) - 90.0
    if abs(angle) < 2.0:
        return None
    return angle


def _rotate_bgra(patch: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a BGRA patch by -angle_deg to straighten it. Expands canvas to fit."""
    if abs(angle_deg) < 2.0:
        return patch
    h, w = patch.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(patch, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    return rotated


def _trim_transparent(patch: np.ndarray) -> np.ndarray:
    """Crop away fully transparent borders from a BGRA patch."""
    if patch.shape[2] != 4:
        return patch
    alpha = patch[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return patch
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return patch[y1:y2 + 1, x1:x2 + 1]


# ──────────────────────── Alpha paste ─────────────────────────────────────────

def _paste_bgra(
    canvas: np.ndarray,
    patch: np.ndarray,
    cx: int,
    cy: int,
    alpha_scale: float = 1.0,
) -> None:
    """Paste a BGRA patch onto a BGR canvas with alpha blending.

    alpha_scale: multiply patch alpha (0–1) for e.g. semi-transparent removed-item thumbs.
    """
    ph, pw = patch.shape[:2]
    x1 = cx - pw // 2
    y1 = cy - ph // 2
    x2 = x1 + pw
    y2 = y1 + ph

    ch, cw = canvas.shape[:2]
    sx = max(0, -x1)
    sy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(cw, x2)
    y2 = min(ch, y2)
    ex = sx + (x2 - x1)
    ey = sy + (y2 - y1)

    if x2 <= x1 or y2 <= y1 or ex > pw or ey > ph:
        return

    region = patch[sy:ey, sx:ex]
    alpha = region[:, :, 3:4].astype(np.float32) / 255.0 * float(alpha_scale)
    alpha = np.clip(alpha, 0.0, 1.0)
    bgr = region[:, :, :3].astype(np.float32)
    dst = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (bgr * alpha + dst * (1.0 - alpha)).astype(np.uint8)


# Opacity for cutouts in the "Removed Items" zone (same scale as other objects, ghosted)
REMOVED_THUMB_ALPHA = 0.48


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
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), fill_bgr, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), fill_bgr, -1)
    for (cx, cy) in [(x + radius, y + radius), (x + w - radius, y + radius),
                      (x + radius, y + h - radius), (x + w - radius, y + h - radius)]:
        cv2.circle(overlay, (cx, cy), radius, fill_bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

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


# ──────────────────────── Orientation helpers ─────────────────────────────────

VERTICAL_LABELS = {
    "pen", "pencil", "marker", "phone", "cell phone", "mouse",
    "bottle", "mug", "cup", "coffee cup",
}

# Items placed first (left side) in stationery zone
STATIONERY_PRIORITY = {"pen": 0, "pencil": 1, "marker": 2, "phone": 3, "cell phone": 3}


def _stationery_sort_key(item: _Item) -> int:
    return STATIONERY_PRIORITY.get(item.label.strip().lower(), 99)


def _ensure_orientation(patch: np.ndarray, label: str, vertical: bool) -> np.ndarray:
    """
    Rotate the patch so that its long axis matches the desired orientation.
    vertical=True means the long axis should be up-down (h > w).
    """
    if patch is None or patch.size == 0:
        return patch
    ph, pw = patch.shape[:2]
    is_tall = ph >= pw
    if vertical and not is_tall:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif not vertical and is_tall:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    return patch


# ──────────────────────── Layout engine ───────────────────────────────────────

def _pack_items_no_overlap(
    items: List[_Item],
    zone_rect: Tuple[int, int, int, int],
    global_scale: float,
    vertical_orientation: bool = True,
    header_h: int = 30,
    padding: int = 12,
    gap: int = 6,
) -> Tuple[List[Tuple[_Item, int, int, np.ndarray]], List[_Item]]:
    """
    Pack items into a zone with no overlap, row by row.
    Items that don't fit are returned as overflow.

    Returns (placed, overflow) where placed is [(item, cx, cy, scaled_patch), ...]
    """
    zx, zy, zw, zh = zone_rect
    inner_x = zx + padding
    inner_y = zy + header_h + padding
    inner_w = zw - 2 * padding
    inner_h = zh - header_h - 2 * padding

    if not items:
        return [], []

    placed: List[Tuple[_Item, int, int, np.ndarray]] = []
    overflow: List[_Item] = []

    cursor_x = inner_x
    cursor_y = inner_y
    row_h = 0

    for item in items:
        if item.patch_bgra is None or item.patch_bgra.size == 0:
            continue

        label = item.label.strip().lower()
        want_vertical = vertical_orientation and label in VERTICAL_LABELS
        patch = _ensure_orientation(item.patch_bgra.copy(), label, want_vertical)

        ph, pw = patch.shape[:2]
        sw = max(8, int(pw * global_scale))
        sh = max(8, int(ph * global_scale))
        scaled = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_AREA)

        if cursor_x + sw > inner_x + inner_w and cursor_x > inner_x:
            cursor_x = inner_x
            cursor_y += row_h + gap
            row_h = 0

        if cursor_y + sh > inner_y + inner_h:
            overflow.append(item)
            continue

        cx = cursor_x + sw // 2
        cy = cursor_y + sh // 2
        placed.append((item, cx, cy, scaled))
        cursor_x += sw + gap
        row_h = max(row_h, sh)

    return placed, overflow


def _workspace_scale_for_fit(
    items: List[_Item],
    zone_rect: Tuple[int, int, int, int],
    base_scale: float,
    vertical_orientation: bool = False,
    header_h: int = 50,
    padding: int = 24,
) -> float:
    """
    Shrink scale so the largest workspace item (e.g. laptop) fits inside the zone.
    Without this, a tall/wide laptop at global_scale can overflow and never get drawn.
    """
    zx, zy, zw, zh = zone_rect
    inner_w = max(8, zw - 2 * padding)
    inner_h = max(8, zh - header_h - 2 * padding)
    max_sw, max_sh = 0, 0
    for item in items:
        if item.patch_bgra is None or item.patch_bgra.size == 0:
            continue
        label = item.label.strip().lower()
        want_vertical = vertical_orientation and label in VERTICAL_LABELS
        patch = _ensure_orientation(item.patch_bgra.copy(), label, want_vertical)
        ph, pw = patch.shape[:2]
        sw = max(8, int(pw * base_scale))
        sh = max(8, int(ph * base_scale))
        max_sw = max(max_sw, sw)
        max_sh = max(max_sh, sh)
    if max_sw == 0:
        return base_scale
    fit = min(inner_w / max_sw, inner_h / max_sh, 1.0)
    return base_scale * fit


# ──────────────────────── Main class ──────────────────────────────────────────

class DeskRelayoutVisualizer:
    """
    Generate a reorganised desk layout image with:
    - SAM2 segmentation (pixel-accurate transparent cutouts)
    - Rotation correction (pens/markers straightened)
    - Proportionally correct sizes (laptop >> mouse)
    """

    def __init__(self, image_path: str, detections: list,
                 sam_masks: Optional[List[np.ndarray]] = None):
        """
        Parameters
        ----------
        image_path : str
            Path to the original desk photo.
        detections : list
            Detection objects with .label, .bbox, and optionally .angle_deg.
        sam_masks : list[np.ndarray] or None
            Per-detection boolean masks (full image size). If None, SAM will
            be run automatically using the detection bounding boxes.
        """
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        self.orig_h, self.orig_w = self.original.shape[:2]

        if sam_masks is None:
            sam_masks = self._run_sam(detections)

        self.items: List[_Item] = []
        for i, d in enumerate(detections):
            label = d.label.strip().lower()
            zone = ZONE_ASSIGNMENT.get(label, "temporary")
            bbox = tuple(int(v) for v in d.bbox)
            angle = getattr(d, "angle_deg", None)

            mask = sam_masks[i] if i < len(sam_masks) else None
            bgra, cw, ch = _crop_with_mask(self.original, bbox, mask)

            # For objects with a clear "up" direction (mouse, mug, bottle,
            # cup), skip angle-based rotation to avoid flipping them upside
            # down — just force vertical orientation using the original crop.
            NO_ANGLE_ROTATE = {"mouse", "mug", "cup", "coffee cup", "bottle", "phone", "cell phone"}

            if label not in NO_ANGLE_ROTATE:
                rot_angle = angle
                if rot_angle is None and bgra is not None and bgra.shape[2] == 4:
                    rot_angle = _angle_from_mask(bgra[:, :, 3])
                if rot_angle is not None:
                    bgra = _rotate_bgra(bgra, rot_angle)
                    bgra = _trim_transparent(bgra)

            # Force vertical items to be truly upright
            if label in VERTICAL_LABELS and bgra is not None and bgra.size > 0:
                ph, pw = bgra.shape[:2]
                if pw > ph:
                    bgra = cv2.rotate(bgra, cv2.ROTATE_90_CLOCKWISE)
                    bgra = _trim_transparent(bgra)

            STRAIGHTEN_LABELS = {"pen", "pencil", "marker"}
            if label in STRAIGHTEN_LABELS and bgra is not None and bgra.shape[2] == 4:
                residual = _angle_from_mask(bgra[:, :, 3])
                if residual is not None and abs(residual) > 1.5:
                    bgra = _rotate_bgra(bgra, residual)
                    bgra = _trim_transparent(bgra)
                    ph, pw = bgra.shape[:2]
                    if pw > ph:
                        bgra = cv2.rotate(bgra, cv2.ROTATE_90_CLOCKWISE)
                        bgra = _trim_transparent(bgra)

            item = _Item(
                label=label, bbox=bbox, zone=zone,
                angle_deg=angle, orig_w=cw, orig_h=ch,
                patch_bgra=bgra,
            )
            self.items.append(item)

    def _run_sam(self, detections: list) -> List[np.ndarray]:
        """Run SAM2 with detection bboxes as prompts."""
        from ultralytics import SAM

        bboxes = [list(d.bbox) for d in detections]
        if not bboxes:
            return []

        sam = SAM("sam2.1_b.pt")
        results = sam(self.image_path, bboxes=bboxes, verbose=False)
        r = results[0]

        masks = []
        if r.masks is not None:
            mask_data = r.masks.data.cpu().numpy()
            for j in range(mask_data.shape[0]):
                masks.append(mask_data[j].astype(np.uint8))
        return masks

    def _compute_global_scale(self) -> float:
        """
        Compute a single scale factor so all objects maintain their
        relative sizes while fitting within the canvas zones.
        """
        canvas_area = CANVAS_W * CANVAS_H * 0.70
        total_obj_area = 0
        for item in self.items:
            if item.patch_bgra is not None and item.zone != "remove":
                ph, pw = item.patch_bgra.shape[:2]
                total_obj_area += pw * ph

        if total_obj_area == 0:
            return 1.0

        area_scale = math.sqrt(canvas_area / total_obj_area)

        max_dim = 0
        for item in self.items:
            if item.patch_bgra is not None and item.zone != "remove":
                ph, pw = item.patch_bgra.shape[:2]
                max_dim = max(max_dim, pw, ph)

        workspace_rect = ZONE_RECTS["workspace"]
        max_zone_dim = max(workspace_rect[2], workspace_rect[3]) - 40

        dim_scale = max_zone_dim / max(max_dim, 1)

        return min(area_scale, dim_scale, 1.0)

    def generate(self, out_path: str, left_handed: bool = False) -> str:
        zone_rects = ZONE_RECTS_LEFT if left_handed else ZONE_RECTS
        mouse_rect = MOUSE_AREA_RECT_LEFT if left_handed else MOUSE_AREA_RECT

        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR[::-1], dtype=np.uint8)

        _put_text(canvas, "Reorganised Desk Layout", (50, 60),
                  color_rgb=(50, 50, 50), scale=1.6, thickness=3)

        zone_items: Dict[str, List[_Item]] = {z: [] for z in zone_rects}
        for item in self.items:
            zone_items.setdefault(item.zone, []).append(item)

        for zname, zrect in zone_rects.items():
            fill = ZONE_COLORS_RGB.get(zname, (240, 240, 240))
            border = ZONE_BORDER_RGB.get(zname, (180, 180, 180))
            _draw_rounded_rect(canvas, zrect, fill, border, radius=24, alpha=0.45)
            zx, zy, zw, zh = zrect
            display = ZONE_DISPLAY_NAMES.get(zname, zname)
            count = len(zone_items.get(zname, []))
            label_text = f"{display} ({count})" if count else display
            _put_text(canvas, label_text, (zx + 18, zy + 36),
                      color_rgb=border, scale=0.85, thickness=2)

        _draw_rounded_rect(canvas, mouse_rect,
                           (245, 245, 255), (150, 150, 200), radius=24, alpha=0.45)
        mx, my, mw, mh = mouse_rect
        ws_all = zone_items.get("workspace", [])
        mouse_n = sum(1 for i in ws_all if i.label == "mouse")
        _put_text(canvas, f"Mouse ({mouse_n})" if mouse_n else "Mouse",
                  (mx + 18, my + 36), color_rgb=(150, 150, 200), scale=0.85, thickness=2)

        global_scale = self._compute_global_scale()

        workspace_items = zone_items.get("workspace", [])
        mouse_items = [i for i in workspace_items if i.label == "mouse"]
        other_workspace = [i for i in workspace_items if i.label != "mouse"]

        for mi in mouse_items:
            if mi.patch_bgra is None or mi.patch_bgra.size == 0:
                continue
            patch = _ensure_orientation(mi.patch_bgra.copy(), "mouse", vertical=True)
            ph, pw = patch.shape[:2]
            sw = max(8, int(pw * global_scale))
            sh = max(8, int(ph * global_scale))
            scaled = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_AREA)
            cx = mx + mw // 2
            cy = my + 50 + sh // 2
            _paste_bgra(canvas, scaled, cx, cy)

        workspace_overflow: List[_Item] = []
        if other_workspace:
            ws_scale = _workspace_scale_for_fit(
                other_workspace,
                zone_rects["workspace"],
                global_scale,
                vertical_orientation=False,
                header_h=50,
                padding=24,
            )
            placed, workspace_overflow = _pack_items_no_overlap(
                other_workspace, zone_rects["workspace"], ws_scale,
                vertical_orientation=False, header_h=50, padding=24, gap=12)
            self._draw_placed(canvas, placed)

        stationery_items = zone_items.get("stationery", [])
        stationery_items.sort(key=_stationery_sort_key)
        placed_stat, overflow = _pack_items_no_overlap(
            stationery_items, zone_rects["stationery"], global_scale,
            vertical_orientation=True, header_h=50, padding=24, gap=12)
        self._draw_placed(canvas, placed_stat)

        ref_items = zone_items.get("reference", [])
        ref_combined = ref_items + overflow + workspace_overflow
        if ref_combined:
            placed_ref, _ = _pack_items_no_overlap(
                ref_combined, zone_rects["reference"], global_scale,
                vertical_orientation=False, header_h=50, padding=24, gap=12)
            self._draw_placed(canvas, placed_ref)

        temp_items = zone_items.get("temporary", [])
        if temp_items:
            placed_tmp, temp_overflow = _pack_items_no_overlap(
                temp_items, zone_rects["temporary"], global_scale,
                vertical_orientation=False, header_h=50, padding=24, gap=12)
            self._draw_placed(canvas, placed_tmp)
            if temp_overflow:
                tmp_rect = zone_rects["temporary"]
                ext_y = tmp_rect[1] + tmp_rect[3] + 10
                ext_h = max(10, CANVAS_H - ext_y - 30)
                ext_rect = (tmp_rect[0], ext_y, tmp_rect[2], ext_h)
                placed_extra, _ = _pack_items_no_overlap(
                    temp_overflow, ext_rect, global_scale,
                    vertical_orientation=False, header_h=50, padding=24, gap=12)
                self._draw_placed(canvas, placed_extra)

        removed = zone_items.get("remove", [])
        if removed:
            zrect = zone_rects["remove"]
            zx, zy, zw, zh = zrect
            _put_text(canvas, "Items removed from desk:", (zx + 20, zy + 80),
                      color_rgb=(160, 120, 120), scale=0.7)
            removed_labels = [item.label for item in removed]
            text = ", ".join(removed_labels)
            _put_text(canvas, text, (zx + 20, zy + 120),
                      color_rgb=(140, 140, 140), scale=0.65)

            pad_x = 24
            pad_bottom = 20
            text_block_h = 125
            inner_x = zx + pad_x
            inner_w = max(8, zw - 2 * pad_x)
            row_top = zy + text_block_h
            row_h = max(40, zh - text_block_h - pad_bottom)
            gap = 12
            thumbs = removed[:8]
            n_items = len(thumbs)
            slot_w = (inner_w - gap * max(0, n_items - 1)) / max(n_items, 1)
            # Fit thumbs inside the box: same max side per slot, not taller than row
            max_side = min(slot_w, row_h * 0.92)

            for idx, item in enumerate(thumbs):
                if item.patch_bgra is None or item.patch_bgra.size == 0:
                    continue
                ph, pw = item.patch_bgra.shape[:2]
                s = min(max_side / max(pw, ph), 1.0)
                tw_s = max(8, int(pw * s))
                th_s = max(8, int(ph * s))
                thumb = cv2.resize(
                    item.patch_bgra, (tw_s, th_s), interpolation=cv2.INTER_AREA
                )
                cx = int(inner_x + idx * (slot_w + gap) + slot_w / 2)
                cy = int(row_top + row_h / 2)
                _paste_bgra(canvas, thumb, cx, cy, alpha_scale=REMOVED_THUMB_ALPHA)

        total = len(self.items)
        kept = sum(1 for i in self.items if i.zone != "remove")
        removed_n = total - kept
        _put_text(canvas, f"Total: {total} objects | Placed: {kept} | Removed: {removed_n}",
                  (CANVAS_W // 2 - 250, CANVAS_H - 30), color_rgb=(160, 160, 160), scale=0.7)

        cv2.imwrite(out_path, canvas)
        print(f"Relayout image saved: {out_path}")
        return out_path

    def _draw_placed(self, canvas: np.ndarray,
                     placed: List[Tuple[_Item, int, int, np.ndarray]]) -> None:
        for item, cx, cy, scaled in placed:
            _paste_bgra(canvas, scaled, cx, cy)


# ──────────────────────── Standalone runner ────────────────────────────────────

_DEFAULT_YOLO = str(Path(__file__).resolve().parent / "runs" / "detect" / "desk_tidy_runs"
                     / "v4_yolov8m_roboflow_style" / "weights" / "best.pt")


def run_standalone(
    image_path: str,
    model_path: str = _DEFAULT_YOLO,
    conf: float = 0.4,
    left_handed: bool = False,
):
    from ultralytics import YOLO
    import importlib.util, sys

    scoring_path = Path(__file__).resolve().parent / "scoring_module" / "scripts" / "Tidy Scoring System.py"
    spec = importlib.util.spec_from_file_location("tidy_scoring_system", str(scoring_path))
    if spec and spec.loader:
        scoring = importlib.util.module_from_spec(spec)
        sys.modules["tidy_scoring_system"] = scoring
        spec.loader.exec_module(scoring)
        Detection = scoring.Detection
        estimate_angle = scoring.estimate_object_angle_deg
    else:
        raise RuntimeError("Cannot load scoring module")

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False, agnostic_nms=True)
    r = results[0]
    names = getattr(r, "names", None) or {}
    orig_img = getattr(r, "orig_img", None)

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for xyxy, c, cls_id in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            label = names.get(int(cls_id), str(int(cls_id)))
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            angle_deg = None
            if isinstance(orig_img, np.ndarray):
                ih, iw = orig_img.shape[:2]
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(iw, int(x2)), min(ih, int(y2))
                if ix2 > ix1 and iy2 > iy1:
                    roi = orig_img[iy1:iy2, ix1:ix2]
                    angle_deg = estimate_angle(label, roi)

            detections.append(Detection(
                label=label, confidence=float(c),
                bbox=(x1, y1, x2, y2), angle_deg=angle_deg,
            ))

    print(f"Detected {len(detections)} objects")
    for d in detections:
        zone = ZONE_ASSIGNMENT.get(d.label.strip().lower(), "temporary")
        angle_str = f" (angle={d.angle_deg:.1f}\u00b0)" if d.angle_deg else ""
        print(f"  {d.label:15s} -> {zone}{angle_str}")

    # SAM masks are computed automatically inside the visualizer
    viz = DeskRelayoutVisualizer(image_path, detections)
    stem = Path(image_path).stem
    out_suffix = "_relayout_left.png" if left_handed else "_relayout.png"
    viz.generate(f"{stem}{out_suffix}", left_handed=left_handed)
    print("(Used SAM2 for pixel-level segmentation)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate reorganised desk layout")
    parser.add_argument("--image", type=str, default="jpg_images/desk_065.jpg")
    parser.add_argument("--model", type=str, default=_DEFAULT_YOLO)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument(
        "--left",
        action="store_true",
        help="Left-handed layout: stationery right, temporary left, mouse bottom-left",
    )
    args = parser.parse_args()
    run_standalone(args.image, args.model, args.conf, left_handed=args.left)
