#!/usr/bin/env python3
"""
Quick demo: detection image, scores, text plan, before/after strip.
Run from repo root: python scripts/teacher_demo.py data/images/desk_065.jpg
Outputs default to teacher_demo/ under the repo root.
Requires: runs/.../best.pt (see README).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_pipeline as rp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Desk tidier — quick teacher demo")
    ap.add_argument("image", help="Path to desk photo, e.g. data/images/desk_065.jpg")
    ap.add_argument(
        "--output-dir",
        default="teacher_demo",
        help="Output folder relative to repo root",
    )
    ap.add_argument(
        "--use-classifier",
        action="store_true",
        help="Enable tidy/untidy classifier gate (needs classifier/desk_classifier.pth)",
    )
    ap.add_argument("--left-handed", action="store_true", help="Left-handed relayout zones")
    args = ap.parse_args()

    img = Path(args.image)
    if not img.is_file():
        print(f"ERROR: Image not found: {img.resolve()}")
        sys.exit(1)

    if not rp.YOLO_MODEL.exists():
        print(f"ERROR: YOLO weights not found: {rp.YOLO_MODEL}")
        sys.exit(1)

    out = REPO_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(rp.YOLO_MODEL))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = None
    if args.use_classifier and rp.CLASSIFIER_MODEL.exists():
        clf = rp.load_classifier(rp.CLASSIFIER_MODEL, dev)
        print("Classifier: enabled")
    elif args.use_classifier:
        print("WARNING: Classifier not found; gating skipped")

    rp.process_image(
        image_path=img,
        yolo_model=yolo,
        classifier=clf,
        cls_device=dev,
        output_dir=out,
        overlap_mode="mask",
        left_handed_relayout=args.left_handed,
    )
    print(f"\n>>> Outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
