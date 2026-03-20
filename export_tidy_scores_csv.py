from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    tidy_script = repo_dir / "Tidy Scoring System.py"
    tidy = load_tidy_module(tidy_script)

    model_path = repo_dir / "runs" / "detect" / "desk_tidy_runs" / "v4_yolov8m_roboflow_style" / "weights" / "best.pt"
    source_dir = Path(r"C:\Users\caoyi\Desktop\DwBS\DVS\final Project\DeskTidier.v3i.yolov8")
    output_csv = repo_dir / "tidy_scores_conf0.4_v4.csv"

    conf = 0.4
    iou = 0.5
    imgsz = 640
    device = "cpu"

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in exts])
    if not images:
        raise FileNotFoundError(f"No images found under: {source_dir}")

    model = YOLO(str(model_path))
    rows: List[Dict[str, Any]] = []

    for img_path in images:
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

    avg_score = sum(r["score"] for r in rows) / len(rows)
    print(f"Exported {len(rows)} rows to: {output_csv}")
    print(f"Average score: {avg_score:.2f}")


if __name__ == "__main__":
    main()

