"""
DeskRelayoutVisualizer — Reorganised Layout Image Generator (Iconic Template)

Creates a clean canvas showing where each detected object should be placed
after tidying.  Each object is drawn as a dashed-outline iconic shape
(pen silhouette, mug circle, laptop rectangle, scissors icon, etc.)
sized proportionally to its real detected bounding box.

Usage (standalone):
    python desk_relayout_viz.py --image jpg_images/desk_065.jpg

Integration:
    from desk_relayout_iconic import DeskRelayoutVisualizer
    viz = DeskRelayoutVisualizer(original_image_path, detections)
    viz.generate("desk_relayout_iconic.png")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

CANVAS_W, CANVAS_H = 1400, 900
BG_COLOR = (255, 255, 255)

ZONE_RECTS: Dict[str, Tuple[int, int, int, int]] = {
    "stationery": (30,  55, 260, 600),
    "reference":  (320, 55,  760, 160),
    "workspace":  (320, 235, 760, 440),
    "temporary":  (1110, 55, 260, 600),
    "remove":     (320, 700, 380, 160),
}


# ──────────────────────── Item dataclass ──────────────────────────────────────

@dataclass
class _Item:
    label: str
    bbox: Tuple[int, int, int, int]
    zone: str
    orig_w: int = 0
    orig_h: int = 0


# ──────────────────────── Dashed drawing primitives ──────────────────────────

def _dashed_contour(
    img: np.ndarray,
    pts: list,
    color_bgr: tuple,
    thickness: int = 2,
    dash: int = 10,
    gap: int = 6,
    closed: bool = True,
) -> None:
    if not pts or len(pts) < 2:
        return
    if closed:
        pts = list(pts) + [pts[0]]
    budget = 0.0
    drawing = True
    for i in range(len(pts) - 1):
        x1, y1 = float(pts[i][0]), float(pts[i][1])
        x2, y2 = float(pts[i + 1][0]), float(pts[i + 1][1])
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len < 0.5:
            continue
        ux, uy = (x2 - x1) / seg_len, (y2 - y1) / seg_len
        pos = 0.0
        while pos < seg_len - 0.5:
            limit = dash if drawing else gap
            step = min(limit - budget, seg_len - pos)
            if drawing and step > 0.5:
                sx = int(x1 + ux * pos)
                sy = int(y1 + uy * pos)
                ex = int(x1 + ux * (pos + step))
                ey = int(y1 + uy * (pos + step))
                cv2.line(img, (sx, sy), (ex, ey), color_bgr, thickness, cv2.LINE_AA)
            pos += step
            budget += step
            if budget >= limit - 0.5:
                drawing = not drawing
                budget = 0.0


def _fill_contour(
    img: np.ndarray,
    pts: list,
    color_bgr: tuple,
    alpha: float = 0.12,
) -> None:
    if len(pts) < 3:
        return
    np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    overlay = img.copy()
    cv2.fillPoly(overlay, [np_pts], color_bgr)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ──────────────────────── Contour point generators ───────────────────────────

def _arc_pts(cx, cy, r, start_deg, end_deg, steps=12):
    pts = []
    for i in range(steps + 1):
        a = math.radians(start_deg + (end_deg - start_deg) * i / steps)
        pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
    return pts


def _ellipse_arc_pts(cx, cy, rx, ry, start_deg, end_deg, steps=24):
    pts = []
    for i in range(steps + 1):
        a = math.radians(start_deg + (end_deg - start_deg) * i / steps)
        pts.append((int(cx + rx * math.cos(a)), int(cy + ry * math.sin(a))))
    return pts


def _rounded_rect_pts(x, y, w, h, r=None):
    if r is None:
        r = max(2, min(w, h) // 5)
    r = max(2, min(r, w // 2, h // 2))
    pts = []
    pts += _arc_pts(x + r, y + r, r, 180, 270)
    pts += _arc_pts(x + w - r, y + r, r, 270, 360)
    pts += _arc_pts(x + w - r, y + h - r, r, 0, 90)
    pts += _arc_pts(x + r, y + h - r, r, 90, 180)
    return pts


# ──────────────────────── Iconic shape drawing functions ─────────────────────

def _shape_laptop(img, cx, cy, w, h, color):
    x, y = cx - w // 2, cy - h // 2
    screen_h = int(h * 0.82)
    pts = _rounded_rect_pts(x, y, w, screen_h, r=max(3, min(w, screen_h) // 10))
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=12, gap=6)
    base_y = y + screen_h + 3
    base_h = h - screen_h - 3
    bpts = [
        (x - 6, base_y), (x + w + 6, base_y),
        (x + w + 8, base_y + base_h), (x - 8, base_y + base_h),
    ]
    _fill_contour(img, bpts, color, alpha=0.07)
    _dashed_contour(img, bpts, color, thickness=2, dash=10, gap=5)


def _shape_keyboard(img, cx, cy, w, h, color):
    x, y = cx - w // 2, cy - h // 2
    pts = _rounded_rect_pts(x, y, w, h, r=max(2, min(w, h) // 6))
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=10, gap=5)
    rows = 3
    for row in range(1, rows):
        ry = y + h * row // rows
        cv2.line(img, (x + 8, ry), (x + w - 8, ry), color, 1, cv2.LINE_AA)


def _shape_mouse(img, cx, cy, w, h, color):
    pts = []
    n = 64
    for i in range(n):
        t = 2 * math.pi * i / n
        rx = w / 2
        ry = h / 2
        squeeze = 1.0 - 0.22 * max(0, -math.sin(t))
        px = cx + int(rx * squeeze * math.cos(t))
        py = cy + int(ry * math.sin(t))
        pts.append((px, py))
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)
    div_y = cy - h // 4
    cv2.line(img, (cx - w // 4, div_y), (cx + w // 4, div_y), color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, div_y + max(2, h // 12)), max(2, w // 8), color, -1, cv2.LINE_AA)


def _shape_pen(img, cx, cy, w, h, color):
    x, y = cx - w // 2, cy - h // 2
    r = max(2, min(w, h) // 4)
    pts = _rounded_rect_pts(x, y, w, h, r=r)
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)


def _shape_pencil(img, cx, cy, w, h, color):
    x, y = cx - w // 2, cy - h // 2
    r = max(2, min(w, h) // 4)
    pts = _rounded_rect_pts(x, y, w, h, r=r)
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)


def _shape_marker(img, cx, cy, w, h, color):
    cap_h = max(4, h // 6)
    x, y = cx - w // 2, cy - h // 2
    r = w // 2
    pts = [
        (cx - w // 3, y),
        (cx + w // 3, y),
        (cx + w // 2, y + cap_h),
        (cx + w // 2, y + h - r),
    ]
    pts += _arc_pts(cx, y + h - r, r, 0, 180)
    pts.append((cx - w // 2, y + cap_h))
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)


def _shape_scissors(img, cx, cy, w, h, color):
    handle_r = max(5, min(w, h) // 4)
    pivot_y = cy + h // 10
    blade_top = cy - h // 2

    blade_w = max(3, w // 5)
    _dashed_contour(img, [
        (cx - blade_w // 2, blade_top),
        (cx + w // 3, pivot_y),
        (cx - blade_w // 2 + 2, pivot_y),
    ], color, thickness=2, dash=6, gap=4, closed=True)

    _dashed_contour(img, [
        (cx + blade_w // 2, blade_top),
        (cx - w // 3, pivot_y),
        (cx + blade_w // 2 - 2, pivot_y),
    ], color, thickness=2, dash=6, gap=4, closed=True)

    lh = _ellipse_arc_pts(cx - w // 4, pivot_y + handle_r + 4,
                           handle_r, int(handle_r * 1.2), 0, 360)
    _fill_contour(img, lh, color, alpha=0.07)
    _dashed_contour(img, lh, color, thickness=2, dash=6, gap=4)

    rh = _ellipse_arc_pts(cx + w // 4, pivot_y + handle_r + 4,
                           handle_r, int(handle_r * 1.2), 0, 360)
    _fill_contour(img, rh, color, alpha=0.07)
    _dashed_contour(img, rh, color, thickness=2, dash=6, gap=4)

    cv2.circle(img, (cx, pivot_y), 3, color, -1, cv2.LINE_AA)


def _shape_phone(img, cx, cy, w, h, color):
    x, y = cx - w // 2, cy - h // 2
    r = max(3, min(w, h) // 4)
    pts = _rounded_rect_pts(x, y, w, h, r=r)
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)
    m = max(2, min(w, h) // 7)
    inner = _rounded_rect_pts(x + m, y + m * 2, w - 2 * m, h - 3 * m, r=max(2, r // 2))
    _dashed_contour(img, inner, color, thickness=1, dash=4, gap=4)
    cv2.circle(img, (cx, y + h - m), max(2, m // 2), color, 1, cv2.LINE_AA)


def _shape_mug(img, cx, cy, w, h, color):
    body_w = int(w * 0.72)
    body_rx = body_w // 2
    body_ry = h // 2
    body_cx = cx - (w - body_w) // 3

    body = _ellipse_arc_pts(body_cx, cy, body_rx, body_ry, 0, 360)
    _fill_contour(img, body, color, alpha=0.10)
    _dashed_contour(img, body, color, thickness=2, dash=8, gap=5)

    hr = max(5, (w - body_w) // 2 + 6)
    handle = _arc_pts(body_cx + body_rx, cy, hr, -55, 55, steps=16)
    _dashed_contour(img, handle, color, thickness=2, dash=6, gap=4, closed=False)


def _shape_bottle(img, cx, cy, w, h, color):
    neck_w = max(5, w // 3)
    neck_h = max(5, h // 4)
    shoulder_h = max(3, h // 8)
    cap_h = max(2, h // 12)
    cap_w = neck_w + 4
    y = cy - h // 2
    pts = [
        (cx - cap_w // 2, y),
        (cx + cap_w // 2, y),
        (cx + cap_w // 2, y + cap_h),
        (cx + neck_w // 2, y + cap_h),
        (cx + neck_w // 2, y + neck_h),
        (cx + w // 2, y + neck_h + shoulder_h),
        (cx + w // 2, y + h),
        (cx - w // 2, y + h),
        (cx - w // 2, y + neck_h + shoulder_h),
        (cx - neck_w // 2, y + neck_h),
        (cx - neck_w // 2, y + cap_h),
        (cx - cap_w // 2, y + cap_h),
    ]
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)


def _shape_tape(img, cx, cy, w, h, color):
    outer_r = min(w, h) // 2
    inner_r = max(3, outer_r * 2 // 5)
    outer = _ellipse_arc_pts(cx, cy, outer_r, outer_r, 0, 360)
    _fill_contour(img, outer, color, alpha=0.10)
    _dashed_contour(img, outer, color, thickness=2, dash=8, gap=5)
    bg = BG_COLOR[::-1]
    inner_pts = _ellipse_arc_pts(cx, cy, inner_r, inner_r, 0, 360)
    _fill_contour(img, inner_pts, bg, alpha=1.0)
    _dashed_contour(img, inner_pts, color, thickness=1, dash=5, gap=4)


def _shape_earphones(img, cx, cy, w, h, color):
    x, y = cx - w // 2, cy - h // 2
    r = max(2, min(w, h) // 5)
    pts = _rounded_rect_pts(x, y, w, h, r=r)
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)


def _shape_cable(img, cx, cy, w, h, color):
    pts = []
    for i in range(48):
        t = i / 47.0
        px = cx - w // 2 + int(t * w)
        py = cy + int(math.sin(t * 3 * 2 * math.pi) * h // 3)
        pts.append((px, py))
    _dashed_contour(img, pts, color, thickness=2, dash=6, gap=4, closed=False)


def _shape_circle(img, cx, cy, w, h, color):
    r = min(w, h) // 2
    pts = _ellipse_arc_pts(cx, cy, r, r, 0, 360)
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)


def _shape_book(img, cx, cy, w, h, color):
    x, y = cx - w // 2, cy - h // 2
    r = max(2, min(w, h) // 10)
    pts = _rounded_rect_pts(x, y, w, h, r=r)
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=10, gap=5)
    spine_x = x + max(4, w // 7)
    cv2.line(img, (spine_x, y + r), (spine_x, y + h - r), color, 1, cv2.LINE_AA)


def _shape_bowl(img, cx, cy, w, h, color):
    rx, ry = w // 2, h // 2
    pts = _ellipse_arc_pts(cx, cy, rx, ry, 10, 170, steps=30)
    rim_l = (cx - rx + int(rx * math.cos(math.radians(170))),
             cy + int(ry * math.sin(math.radians(170))))
    rim_r = (cx + int(rx * math.cos(math.radians(10))),
             cy + int(ry * math.sin(math.radians(10))))
    pts = [rim_l] + pts + [rim_r]
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=8, gap=5)


def _shape_generic_rect(img, cx, cy, w, h, color):
    x, y = cx - w // 2, cy - h // 2
    pts = _rounded_rect_pts(x, y, w, h)
    _fill_contour(img, pts, color, alpha=0.10)
    _dashed_contour(img, pts, color, thickness=2, dash=10, gap=5)


SHAPE_DISPATCH = {
    "laptop": _shape_laptop,
    "keyboard": _shape_keyboard,
    "mouse": _shape_mouse,
    "pen": _shape_pen,
    "pencil": _shape_pencil,
    "marker": _shape_marker,
    "scissors": _shape_scissors,
    "scissor": _shape_scissors,
    "phone": _shape_phone,
    "cell phone": _shape_phone,
    "mug": _shape_mug,
    "cup": _shape_mug,
    "coffee cup": _shape_mug,
    "bottle": _shape_bottle,
    "book": _shape_book,
    "notebook": _shape_book,
    "paper": _shape_generic_rect,
    "sticky note": _shape_generic_rect,
    "tape": _shape_tape,
    "earphones": _shape_earphones,
    "cable": _shape_cable,
    "charger": _shape_generic_rect,
    "eraser": _shape_generic_rect,
    "clock": _shape_circle,
    "remote": _shape_generic_rect,
    "bowl": _shape_bowl,
    "spitball": _shape_circle,
    "ring-pull can": _shape_circle,
    "food packaging": _shape_generic_rect,
    "trash": _shape_generic_rect,
    "battery": _shape_generic_rect,
}


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
    for (ccx, ccy) in [(x + radius, y + radius), (x + w - radius, y + radius),
                        (x + radius, y + h - radius), (x + w - radius, y + h - radius)]:
        cv2.circle(overlay, (ccx, ccy), radius, fill_bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for p1, p2 in [
        ((x + radius, y), (x + w - radius, y)),
        ((x + radius, y + h), (x + w - radius, y + h)),
        ((x, y + radius), (x, y + h - radius)),
        ((x + w, y + radius), (x + w, y + h - radius)),
    ]:
        cv2.line(img, p1, p2, border_bgr, 2)
    for (ccx, ccy), sa, ea in [
        ((x + radius, y + radius), 180, 270),
        ((x + w - radius, y + radius), 270, 360),
        ((x + radius, y + h - radius), 90, 180),
        ((x + w - radius, y + h - radius), 0, 90),
    ]:
        cv2.ellipse(img, (ccx, ccy), (radius, radius), 0, sa, ea, border_bgr, 2)


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

STATIONERY_PRIORITY = {"pen": 0, "pencil": 1, "marker": 2, "phone": 3, "cell phone": 3}


def _stationery_sort_key(item: _Item) -> int:
    return STATIONERY_PRIORITY.get(item.label.strip().lower(), 99)


# ──────────────────────── Layout engine ───────────────────────────────────────

def _pack_items(
    items: List[_Item],
    zone_rect: Tuple[int, int, int, int],
    global_scale: float,
    vertical_orientation: bool = True,
    header_h: int = 30,
    padding: int = 12,
    gap: int = 8,
) -> Tuple[List[Tuple[_Item, int, int, int, int]], List[_Item]]:
    """
    Pack items into a zone with no overlap, row by row.
    Returns (placed, overflow) where placed is [(item, cx, cy, draw_w, draw_h), ...]
    """
    zx, zy, zw, zh = zone_rect
    inner_x = zx + padding
    inner_y = zy + header_h + padding
    inner_w = zw - 2 * padding
    inner_h = zh - header_h - 2 * padding

    if not items:
        return [], []

    placed: List[Tuple[_Item, int, int, int, int]] = []
    overflow: List[_Item] = []

    cursor_x = inner_x
    cursor_y = inner_y
    row_h = 0

    for item in items:
        label = item.label.strip().lower()
        w, h = item.orig_w, item.orig_h
        want_vert = vertical_orientation and label in VERTICAL_LABELS
        if want_vert and w > h:
            w, h = h, w
        elif not want_vert and h > w and label in VERTICAL_LABELS:
            w, h = h, w

        sw = max(10, int(w * global_scale))
        sh = max(10, int(h * global_scale))

        if cursor_x + sw > inner_x + inner_w and cursor_x > inner_x:
            cursor_x = inner_x
            cursor_y += row_h + gap
            row_h = 0

        if cursor_y + sh > inner_y + inner_h:
            remaining_h = (inner_y + inner_h) - cursor_y
            if remaining_h > 20:
                shrink = remaining_h / sh
                sw = max(10, int(sw * shrink))
                sh = max(10, int(remaining_h))
            else:
                overflow.append(item)
                continue

        cx = cursor_x + sw // 2
        cy = cursor_y + sh // 2
        placed.append((item, cx, cy, sw, sh))
        cursor_x += sw + gap
        row_h = max(row_h, sh)

    return placed, overflow


# ──────────────────────── Main class ──────────────────────────────────────────

class DeskRelayoutVisualizer:
    """
    Generate a reorganised desk layout image with iconic dashed-outline shapes
    proportionally sized to real detected objects.
    """

    def __init__(self, image_path: str, detections: list,
                 sam_masks=None):
        self.image_path = image_path
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        self.orig_h, self.orig_w = img.shape[:2]

        self.items: List[_Item] = []
        for d in detections:
            label = d.label.strip().lower()
            zone = ZONE_ASSIGNMENT.get(label, "temporary")
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            self.items.append(_Item(
                label=label, bbox=(x1, y1, x2, y2),
                zone=zone, orig_w=bw, orig_h=bh,
            ))

    def _compute_global_scale(self) -> float:
        canvas_area = CANVAS_W * CANVAS_H * 0.50
        total_obj_area = 0
        for item in self.items:
            if item.zone != "remove":
                total_obj_area += item.orig_w * item.orig_h
        if total_obj_area == 0:
            return 1.0

        area_scale = math.sqrt(canvas_area / total_obj_area)

        max_dim = 0
        for item in self.items:
            if item.zone != "remove":
                max_dim = max(max_dim, item.orig_w, item.orig_h)

        ws = ZONE_RECTS["workspace"]
        max_zone_dim = max(ws[2], ws[3]) - 50
        dim_scale = max_zone_dim / max(max_dim, 1)

        return min(area_scale, dim_scale, 1.0)

    def generate(self, out_path: str) -> str:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR[::-1], dtype=np.uint8)

        _put_text(canvas, "Reorganised Desk Layout", (30, 38),
                  color_rgb=(50, 50, 50), scale=0.9, thickness=2)

        zone_items: Dict[str, List[_Item]] = {z: [] for z in ZONE_RECTS}
        for item in self.items:
            zone_items.setdefault(item.zone, []).append(item)

        for zname, zrect in ZONE_RECTS.items():
            fill = ZONE_COLORS_RGB.get(zname, (240, 240, 240))
            border = ZONE_BORDER_RGB.get(zname, (180, 180, 180))
            _draw_rounded_rect(canvas, zrect, fill, border, radius=14, alpha=0.45)
            zx, zy = zrect[0], zrect[1]
            display = ZONE_DISPLAY_NAMES.get(zname, zname)
            count = len(zone_items.get(zname, []))
            label_text = f"{display} ({count})" if count else display
            _put_text(canvas, label_text, (zx + 12, zy + 22),
                      color_rgb=border, scale=0.5, thickness=1)

        global_scale = self._compute_global_scale()

        # ── Mouse: workspace bottom-right ──
        workspace_items = zone_items.get("workspace", [])
        mouse_items = [i for i in workspace_items if i.label == "mouse"]
        other_workspace = [i for i in workspace_items if i.label != "mouse"]

        ws_x, ws_y, ws_w, ws_h = ZONE_RECTS["workspace"]
        ws_color = ZONE_BORDER_RGB["workspace"][::-1]

        for mi in mouse_items:
            w, h = mi.orig_h, mi.orig_w
            if w > h:
                w, h = h, w
            sw = max(10, int(w * global_scale))
            sh = max(10, int(h * global_scale))
            mcx = ws_x + ws_w - 16 - sw // 2
            mcy = ws_y + ws_h - 16 - sh // 2
            _draw_shape_at(canvas, mi.label, mcx, mcy, sw, sh, ws_color)

        if other_workspace:
            placed, _ = _pack_items(other_workspace, ZONE_RECTS["workspace"],
                                    global_scale, vertical_orientation=False)
            _draw_placed_shapes(canvas, placed, "workspace")

        # ── Stationery: vertical, pens/phones first ──
        stat_items = zone_items.get("stationery", [])
        stat_items.sort(key=_stationery_sort_key)
        placed_stat, overflow = _pack_items(stat_items, ZONE_RECTS["stationery"],
                                            global_scale, vertical_orientation=True)
        _draw_placed_shapes(canvas, placed_stat, "stationery")

        # Overflow stationery → reference zone (horizontal)
        ref_items = zone_items.get("reference", [])
        ref_combined = ref_items + overflow
        if ref_combined:
            placed_ref, _ = _pack_items(ref_combined, ZONE_RECTS["reference"],
                                        global_scale, vertical_orientation=False)
            _draw_placed_shapes(canvas, placed_ref, "reference")

        # ── Temporary zone ──
        temp_items = zone_items.get("temporary", [])
        if temp_items:
            placed_tmp, temp_overflow = _pack_items(temp_items, ZONE_RECTS["temporary"],
                                                    global_scale, vertical_orientation=False)
            _draw_placed_shapes(canvas, placed_tmp, "temporary")
            if temp_overflow:
                fb = (ZONE_RECTS["workspace"][0], ZONE_RECTS["remove"][1],
                      ZONE_RECTS["workspace"][2], ZONE_RECTS["remove"][3])
                placed_fb, _ = _pack_items(temp_overflow, fb, global_scale,
                                           vertical_orientation=False)
                _draw_placed_shapes(canvas, placed_fb, "temporary")

        # ── Removed items ──
        removed = zone_items.get("remove", [])
        if removed:
            zrect = ZONE_RECTS["remove"]
            zx, zy = zrect[0], zrect[1]
            _put_text(canvas, "Items removed from desk:", (zx + 12, zy + 50),
                      color_rgb=(160, 120, 120), scale=0.42)
            text = ", ".join(item.label for item in removed)
            _put_text(canvas, text, (zx + 12, zy + 75),
                      color_rgb=(140, 140, 140), scale=0.4)

            rm_color = ZONE_BORDER_RGB["remove"][::-1]
            rm_x = zx + 15
            for item in removed[:6]:
                s = min(40 / max(item.orig_w, 1), 40 / max(item.orig_h, 1), global_scale)
                tw = max(8, int(item.orig_w * s))
                th = max(8, int(item.orig_h * s))
                _draw_shape_at(canvas, item.label, rm_x + tw // 2, zy + 110, tw, th, rm_color)
                rm_x += tw + 10

        total = len(self.items)
        kept = sum(1 for i in self.items if i.zone != "remove")
        removed_n = total - kept
        _put_text(canvas, f"Total: {total} objects | Placed: {kept} | Removed: {removed_n}",
                  (730, CANVAS_H - 20), color_rgb=(160, 160, 160), scale=0.42)

        cv2.imwrite(out_path, canvas)
        print(f"Relayout image saved: {out_path}")
        return out_path


def _draw_shape_at(img, label, cx, cy, w, h, color_bgr):
    label_lower = label.strip().lower()
    func = SHAPE_DISPATCH.get(label_lower, _shape_generic_rect)
    func(img, cx, cy, w, h, color_bgr)
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0][0]
    _put_text(img, label, (cx - tw // 2, cy + h // 2 + 14),
              color_rgb=(110, 110, 110), scale=0.35, thickness=1)


def _draw_placed_shapes(
    img: np.ndarray,
    placed: List[Tuple[_Item, int, int, int, int]],
    zone_name: str,
) -> None:
    color_bgr = ZONE_BORDER_RGB.get(zone_name, (150, 150, 150))[::-1]
    for item, cx, cy, sw, sh in placed:
        _draw_shape_at(img, item.label, cx, cy, sw, sh, color_bgr)


# ──────────────────────── Standalone runner ────────────────────────────────────

_DEFAULT_YOLO = str(Path(__file__).resolve().parent / "runs" / "detect" / "desk_tidy_runs"
                     / "v4_yolov8m_roboflow_style" / "weights" / "best.pt")


def run_standalone(image_path: str, model_path: str = _DEFAULT_YOLO, conf: float = 0.4):
    from ultralytics import YOLO
    import importlib.util, sys

    scoring_path = Path(__file__).resolve().parent / "scoring_module" / "scripts" / "Tidy Scoring System.py"
    spec = importlib.util.spec_from_file_location("tidy_scoring_system", str(scoring_path))
    if spec and spec.loader:
        scoring = importlib.util.module_from_spec(spec)
        sys.modules["tidy_scoring_system"] = scoring
        spec.loader.exec_module(scoring)
        Detection = scoring.Detection
    else:
        raise RuntimeError("Cannot load scoring module")

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False, agnostic_nms=True)
    r = results[0]
    names = getattr(r, "names", None) or {}

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for xyxy, c, cls_id in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            label = names.get(int(cls_id), str(int(cls_id)))
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            detections.append(Detection(
                label=label, confidence=float(c),
                bbox=(x1, y1, x2, y2), angle_deg=None,
            ))

    print(f"Detected {len(detections)} objects")
    for d in detections:
        zone = ZONE_ASSIGNMENT.get(d.label.strip().lower(), "temporary")
        print(f"  {d.label:15s} -> {zone}")

    viz = DeskRelayoutVisualizer(image_path, detections)
    stem = Path(image_path).stem
    viz.generate(f"{stem}_relayout_iconic.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate reorganised desk layout (iconic template)")
    parser.add_argument("--image", type=str, default="jpg_images/desk_065.jpg")
    parser.add_argument("--model", type=str, default=_DEFAULT_YOLO)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    run_standalone(args.image, args.model, args.conf)
