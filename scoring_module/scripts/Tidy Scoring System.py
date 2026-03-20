"""
Tidy Scoring System

Implements the "Desktop Tidy Scoring Framework" (0–100 score).

GitHub: https://github.com/L1nda-L1u/ComputerVision-DeskTidier.git
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


GITHUB_REPO_URL = "https://github.com/L1nda-L1u/ComputerVision-DeskTidier.git"


# Matches the categories defined in `Scoring Framework.md`
CATEGORY_PENALTY_PER_OBJECT: Dict[str, int] = {
    "Core Work Items": 0,
    "Study Items": 0,
    "Temporary Items": 2,
    "Clutter Items": 4,
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


def bbox_area_xyxy(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


ANGLE_RECT_LABELS = {
    "laptop",
    "book",
    "phone",
    "sticky note",
    "eraser",
    "foodpacking",
    "food packaging",
    "earphones",
}

ANGLE_LINE_LABELS = {
    "pen",
    "pencil",
    "marker",
}


def normalize_orientation_deg(angle_deg: float) -> float:
    """
    Normalize orientation to [-90, 90) since 0 and 180 represent same axis.
    """

    a = ((angle_deg + 90.0) % 180.0) - 90.0
    return a


def estimate_angle_rect_deg(roi_bgr: np.ndarray) -> Optional[float]:
    if roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.02 * float(roi_bgr.shape[0] * roi_bgr.shape[1]):
        return None

    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    angle = float(rect[2])
    if w < h:
        angle += 90.0
    return normalize_orientation_deg(angle)


def estimate_angle_line_deg(roi_bgr: np.ndarray) -> Optional[float]:
    if roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=20,
        minLineLength=max(10, int(min(roi_bgr.shape[:2]) * 0.25)),
        maxLineGap=8,
    )
    if lines is None or len(lines) == 0:
        return None

    sum_cos2 = 0.0
    sum_sin2 = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 6:
            continue
        ang = math.atan2(dy, dx)
        # Double-angle trick for 180-degree periodic orientations.
        sum_cos2 += length * math.cos(2.0 * ang)
        sum_sin2 += length * math.sin(2.0 * ang)

    if abs(sum_cos2) < 1e-6 and abs(sum_sin2) < 1e-6:
        return None

    ori = 0.5 * math.atan2(sum_sin2, sum_cos2)
    return normalize_orientation_deg(math.degrees(ori))


def estimate_object_angle_deg(label: str, roi_bgr: np.ndarray) -> Optional[float]:
    l = label.strip().lower()
    if l in ANGLE_RECT_LABELS:
        return estimate_angle_rect_deg(roi_bgr) or estimate_angle_line_deg(roi_bgr)
    if l in ANGLE_LINE_LABELS:
        return estimate_angle_line_deg(roi_bgr) or estimate_angle_rect_deg(roi_bgr)
    return None


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
    # Optional foreground mask inside bbox ROI (uint8 0/255).
    roi_mask: Optional[np.ndarray] = None


def estimate_foreground_mask(roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate object foreground mask inside an ROI.
    Returns uint8 mask (0/255) with same HxW as ROI.
    """

    if roi_bgr.size == 0:
        return None

    h, w = roi_bgr.shape[:2]
    if h < 8 or w < 8:
        return None

    # Try GrabCut first (better object-vs-background separation than bbox geometry).
    try:
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (1, 1, max(1, w - 2), max(1, h - 2))
        cv2.grabCut(roi_bgr, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        fg = None

    if fg is None or int(np.count_nonzero(fg)) == 0:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Choose polarity with smaller non-zero region as likely object foreground.
        nz = int(np.count_nonzero(th))
        inv = 255 - th
        nz_inv = int(np.count_nonzero(inv))
        fg = th if 0 < nz < nz_inv else inv

    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)

    if int(np.count_nonzero(fg)) < 10:
        return None
    return fg


def mask_overlap_metrics(a: Detection, b: Detection) -> Optional[Tuple[float, float]]:
    """
    Compute overlap metrics using object masks.
    Returns (mask_iou, contain) where contain = inter/min(area_a, area_b).
    Returns None if masks unavailable/invalid.
    """

    if a.roi_mask is None or b.roi_mask is None:
        return None

    ax1, ay1, ax2, ay2 = [int(round(v)) for v in a.bbox]
    bx1, by1, bx2, by2 = [int(round(v)) for v in b.bbox]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0, 0.0

    # Slice overlap on each ROI mask.
    a_x1 = inter_x1 - ax1
    a_y1 = inter_y1 - ay1
    a_x2 = a_x1 + (inter_x2 - inter_x1)
    a_y2 = a_y1 + (inter_y2 - inter_y1)

    b_x1 = inter_x1 - bx1
    b_y1 = inter_y1 - by1
    b_x2 = b_x1 + (inter_x2 - inter_x1)
    b_y2 = b_y1 + (inter_y2 - inter_y1)

    # Guard against shape mismatches.
    a_h, a_w = a.roi_mask.shape[:2]
    b_h, b_w = b.roi_mask.shape[:2]
    if not (0 <= a_x1 < a_w and 0 <= a_y1 < a_h and 0 < a_x2 <= a_w and 0 < a_y2 <= a_h):
        return None
    if not (0 <= b_x1 < b_w and 0 <= b_y1 < b_h and 0 < b_x2 <= b_w and 0 < b_y2 <= b_h):
        return None

    a_patch = a.roi_mask[a_y1:a_y2, a_x1:a_x2] > 0
    b_patch = b.roi_mask[b_y1:b_y2, b_x1:b_x2] > 0
    if a_patch.shape != b_patch.shape or a_patch.size == 0:
        return None

    inter = int(np.logical_and(a_patch, b_patch).sum())
    area_a = int((a.roi_mask > 0).sum())
    area_b = int((b.roi_mask > 0).sum())
    union = area_a + area_b - inter
    if union <= 0:
        return None

    iou = inter / union
    contain = inter / max(1, min(area_a, area_b))
    return iou, contain


def center_in_bbox(center: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
    cx, cy = center
    x1, y1, x2, y2 = bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def elongated_overlap_ratio(line_obj: Detection, other_obj: Detection) -> float:
    """
    Fraction of elongated object that overlaps with another object.
    Prefer mask-based overlap when available; fallback to bbox intersection ratio.
    """

    # Mask-based overlap fraction on elongated object foreground.
    m = mask_overlap_metrics(line_obj, other_obj)
    if m is not None and line_obj.roi_mask is not None:
        lx1, ly1, lx2, ly2 = [int(round(v)) for v in line_obj.bbox]
        ox1, oy1, ox2, oy2 = [int(round(v)) for v in other_obj.bbox]
        inter_x1, inter_y1 = max(lx1, ox1), max(ly1, oy1)
        inter_x2, inter_y2 = min(lx2, ox2), min(ly2, oy2)
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            a_x1 = inter_x1 - lx1
            a_y1 = inter_y1 - ly1
            a_x2 = a_x1 + (inter_x2 - inter_x1)
            a_y2 = a_y1 + (inter_y2 - inter_y1)
            lmask = line_obj.roi_mask
            if (
                0 <= a_x1 < lmask.shape[1]
                and 0 <= a_y1 < lmask.shape[0]
                and 0 < a_x2 <= lmask.shape[1]
                and 0 < a_y2 <= lmask.shape[0]
            ):
                inter_line = int((lmask[a_y1:a_y2, a_x1:a_x2] > 0).sum())
                line_area = int((lmask > 0).sum())
                if line_area > 0:
                    return inter_line / line_area

    # Fallback: bbox intersection fraction on elongated bbox.
    inter = intersection_area_xyxy(line_obj.bbox, other_obj.bbox)
    line_area = bbox_area_xyxy(line_obj.bbox)
    if line_area <= 0:
        return 0.0
    return inter / line_area


def elongated_axis_cover_ratio(line_obj: Detection, other_obj: Detection) -> float:
    """
    Estimate how much of elongated object's principal axis lies over other object's bbox.
    Useful when contact area is small but placement is clearly "on top of" (e.g., pen on book).
    """

    if line_obj.roi_mask is None:
        return 0.0

    ys, xs = np.where(line_obj.roi_mask > 0)
    if len(xs) < 12:
        return 0.0

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    try:
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    except Exception:
        return 0.0

    x1, y1, x2, y2 = line_obj.bbox
    roi_w = max(1.0, x2 - x1)
    roi_h = max(1.0, y2 - y1)
    axis_len = max(roi_w, roi_h) * 1.2

    p1 = (x0 - vx * axis_len * 0.5, y0 - vy * axis_len * 0.5)
    p2 = (x0 + vx * axis_len * 0.5, y0 + vy * axis_len * 0.5)

    # Sample along the axis and count points covered by other bbox.
    ox1, oy1, ox2, oy2 = other_obj.bbox
    total = 40
    covered = 0
    for t in np.linspace(0.0, 1.0, total):
        sx = float(p1[0] * (1 - t) + p2[0] * t) + x1
        sy = float(p1[1] * (1 - t) + p2[1] * t) + y1
        if ox1 <= sx <= ox2 and oy1 <= sy <= oy2:
            covered += 1
    return covered / total


def draw_overlap_visualization(
    image_bgr: np.ndarray,
    detections: List[Detection],
    overlap_pairs_idx: List[Tuple[int, int]],
    out_path: Path,
) -> None:
    """
    Save an image visualizing overlap detections.
    - Overlap objects: red boxes
    - Non-overlap objects: gray boxes
    - Overlap pair center links: yellow lines
    - If mask exists, overlay translucent foreground masks
    """

    vis = image_bgr.copy()
    overlap_ids = set()
    for i, j in overlap_pairs_idx:
        overlap_ids.add(i)
        overlap_ids.add(j)

    # Draw all boxes first
    for idx, d in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
        color = (30, 30, 220) if idx in overlap_ids else (130, 130, 130)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{idx}:{d.label}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        if d.roi_mask is not None:
            h, w = vis.shape[:2]
            ix1 = max(0, min(w, x1))
            iy1 = max(0, min(h, y1))
            ix2 = max(0, min(w, x2))
            iy2 = max(0, min(h, y2))
            if ix2 > ix1 and iy2 > iy1:
                roi_h = iy2 - iy1
                roi_w = ix2 - ix1
                m = d.roi_mask
                if m.shape[:2] != (roi_h, roi_w):
                    m = cv2.resize(m, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                color_fill = np.array((0, 0, 255), dtype=np.uint8) if idx in overlap_ids else np.array((180, 180, 180), dtype=np.uint8)
                patch = vis[iy1:iy2, ix1:ix2]
                alpha = 0.28
                mask_bool = m > 0
                patch[mask_bool] = (patch[mask_bool] * (1 - alpha) + color_fill * alpha).astype(np.uint8)
                vis[iy1:iy2, ix1:ix2] = patch

    # Draw overlap links
    for k, (i, j) in enumerate(overlap_pairs_idx, start=1):
        ax1, ay1, ax2, ay2 = detections[i].bbox
        bx1, by1, bx2, by2 = detections[j].bbox
        ca = (int((ax1 + ax2) / 2), int((ay1 + ay2) / 2))
        cb = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
        cv2.line(vis, ca, cb, (0, 215, 255), 2, cv2.LINE_AA)
        mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
        cv2.putText(vis, f"O{k}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 2, cv2.LINE_AA)

    cv2.putText(
        vis,
        f"Overlap pairs: {len(overlap_pairs_idx)}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 220),
        2,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def object_load_penalty(num_objects: int) -> int:
    """
    Table (Scoring Framework section 2).
    """

    if num_objects <= 8:
        return 0
    if 9 <= num_objects <= 12:
        return 3
    if 13 <= num_objects <= 15:
        return 5
    if 16 <= num_objects <= 18:
        return 8
    return 10


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
    overlap_mode: str = "bbox",
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
    workspace_blocked_categories = set()
    for d in detections:
        cx, cy = bbox_center_xyxy(d.bbox)
        if left <= cx <= right and top <= cy <= bottom:
            workspace_blocked_objects += 1
            cat = infer_category(d.label)
            workspace_blocked_categories.add(cat)
    # Apply workspace penalty once per category (not once per object).
    for cat in workspace_blocked_categories:
        workspace_penalty += WORKSPACE_PENALTY_IF_IN_WORKSPACE[cat]

    # 5.1) Object Overlap Penalty
    # Sensitive occlusion-oriented overlap rule:
    # - Overlap is counted if IoU > 0.20 OR contain >= 0.35
    #   where contain = intersection / min(area_a, area_b).
    # Also skip likely duplicate detections of the same object (same label + near containment).
    overlap_pairs = 0
    overlap_penalty = 0
    overlap_pairs_idx: List[Tuple[int, int]] = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a = detections[i]
            b = detections[j]
        if overlap_mode == "mask":
            m = mask_overlap_metrics(a, b)
            if m is None:
                iou = iou_xyxy(a.bbox, b.bbox)
                inter = intersection_area_xyxy(a.bbox, b.bbox)
                smaller = min(bbox_area_xyxy(a.bbox), bbox_area_xyxy(b.bbox))
                contain = inter / smaller if smaller > 0 else 0.0
            else:
                iou, contain = m
        else:
            iou = iou_xyxy(a.bbox, b.bbox)
            inter = intersection_area_xyxy(a.bbox, b.bbox)
            smaller = min(bbox_area_xyxy(a.bbox), bbox_area_xyxy(b.bbox))
            contain = inter / smaller if smaller > 0 else 0.0

        # Same-label near-contained boxes are often duplicate predictions.
        # Keep this very strict to avoid dropping true overlaps among similar items.
        if a.label == b.label and iou >= 0.85 and contain >= 0.95:
            continue

        # Extra occlusion rule: elongated object partially lying on a larger surface
        # (e.g. pen on book), even if center is outside.
        a_area = bbox_area_xyxy(a.bbox)
        b_area = bbox_area_xyxy(b.bbox)
        elongated_on_surface = False
        if a.label.strip().lower() in ANGLE_LINE_LABELS and a_area < b_area:
            elongated_on_surface = (
                elongated_overlap_ratio(a, b) >= 0.18
                or elongated_axis_cover_ratio(a, b) >= 0.25
            )
        elif b.label.strip().lower() in ANGLE_LINE_LABELS and b_area < a_area:
            elongated_on_surface = (
                elongated_overlap_ratio(b, a) >= 0.18
                or elongated_axis_cover_ratio(b, a) >= 0.25
            )

        a_is_line = a.label.strip().lower() in ANGLE_LINE_LABELS
        b_is_line = b.label.strip().lower() in ANGLE_LINE_LABELS

        # Parallel/adjacent pens often have tiny edge contact that should NOT count as overlap.
        # For line-vs-line pairs, use stricter criteria.
        if a_is_line and b_is_line:
            is_overlap = (iou > 0.35) or (contain >= 0.55)
        else:
            is_overlap = (iou > 0.2) or (contain >= 0.35) or elongated_on_surface

        if is_overlap:
            overlap_pairs += 1
            overlap_pairs_idx.append((i, j))
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
    reasons.append(
        f"{workspace_blocked_objects} objects in central workspace; "
        f"blocked categories={len(workspace_blocked_categories)} -> workspace penalty {workspace_penalty}"
    )
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
            "overlap_pairs_idx": overlap_pairs_idx,
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
    parser.add_argument(
        "--estimate-object-angle",
        action="store_true",
        help="Estimate object angles from each ROI for alignment penalty.",
    )
    parser.add_argument(
        "--overlap-mode",
        type=str,
        default="bbox",
        choices=["bbox", "mask"],
        help="Overlap estimation mode: bbox IoU or object mask overlap.",
    )
    parser.add_argument(
        "--overlap-visualize",
        action="store_true",
        help="Save an image with overlap pairs highlighted.",
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
        orig_img = getattr(r, "orig_img", None)
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            # ultralytics boxes: xyxy, conf, cls
            for xyxy, conf, cls_id in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                cls_id_int = int(cls_id)
                label = names.get(cls_id_int, str(cls_id_int))
                x1, y1, x2, y2 = (
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3]),
                )
                angle_deg = None
                roi_mask = None
                if args.estimate_object_angle and isinstance(orig_img, np.ndarray):
                    h_img, w_img = orig_img.shape[:2]
                    ix1 = max(0, min(w_img, int(round(x1))))
                    iy1 = max(0, min(h_img, int(round(y1))))
                    ix2 = max(0, min(w_img, int(round(x2))))
                    iy2 = max(0, min(h_img, int(round(y2))))
                    if ix2 > ix1 and iy2 > iy1:
                        roi = orig_img[iy1:iy2, ix1:ix2]
                        angle_deg = estimate_object_angle_deg(label, roi)
                if args.overlap_mode == "mask" and isinstance(orig_img, np.ndarray):
                    h_img, w_img = orig_img.shape[:2]
                    ix1 = max(0, min(w_img, int(round(x1))))
                    iy1 = max(0, min(h_img, int(round(y1))))
                    ix2 = max(0, min(w_img, int(round(x2))))
                    iy2 = max(0, min(h_img, int(round(y2))))
                    if ix2 > ix1 and iy2 > iy1:
                        roi = orig_img[iy1:iy2, ix1:ix2]
                        roi_mask = estimate_foreground_mask(roi)
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2),
                        angle_deg=angle_deg,
                        roi_mask=roi_mask,
                    )
                )

        # ultralytics gives orig_shape as (h, w)
        h, w = r.orig_shape
        res = tidy_score(
            detections,
            image_size=(w, h),
            desk_orientation_deg=float(args.desk_orientation),
            alignment_misalignment_threshold_deg=float(args.alignment_threshold),
            overlap_mode=args.overlap_mode,
        )
        scores.append(res["tidy_score"])

        if args.overlap_visualize and isinstance(orig_img, np.ndarray):
            if args.save:
                vis_dir = Path(str(args.save_project)) / str(args.save_name)
            else:
                vis_dir = Path.cwd() / "overlap_visualizations"
            vis_name = f"{img_path.stem}_overlap_debug.jpg"
            draw_overlap_visualization(
                image_bgr=orig_img,
                detections=detections,
                overlap_pairs_idx=res["explanation"].get("overlap_pairs_idx", []),
                out_path=vis_dir / vis_name,
            )

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