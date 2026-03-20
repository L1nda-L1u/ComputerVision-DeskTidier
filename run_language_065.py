from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ultralytics import YOLO

from desk_language_recommend import generate_language_recommendations


def load_scoring_module():
    root = Path(__file__).resolve().parent
    scoring_path = root / "scoring_module" / "scripts" / "Tidy Scoring System.py"
    spec = importlib.util.spec_from_file_location("tidy_scoring_system", str(scoring_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scoring module: {scoring_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tidy_scoring_system"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = Path(__file__).resolve().parent
    scoring = load_scoring_module()

    model_path = root / "runs" / "detect" / "desk_tidy_runs" / "v4_yolov8m_roboflow_style" / "weights" / "best.pt"
    image_path = Path(
        r"C:\Users\caoyi\Desktop\DwBS\DVS\final Project\DeskTidier.v3i.yolov8\test\images\desk_065_jpg.rf.4fb2734a863b527113972e146b70b277.jpg"
    )

    model = YOLO(str(model_path))
    pred = model.predict(source=str(image_path), conf=0.4, iou=0.5, imgsz=640, device="cpu", verbose=False)[0]

    detections = []
    names = pred.names
    if pred.boxes is not None and len(pred.boxes) > 0:
        for xyxy, score, cls_id in zip(pred.boxes.xyxy.tolist(), pred.boxes.conf.tolist(), pred.boxes.cls.tolist()):
            label = names.get(int(cls_id), str(int(cls_id)))
            detections.append(
                scoring.Detection(
                    label=label,
                    confidence=float(score),
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    angle_deg=None,
                )
            )

    h, w = pred.orig_shape
    tidy_result = scoring.tidy_score(detections, image_size=(w, h))
    rec = generate_language_recommendations(detections, tidy_result)

    print(f"Tidy Score: {rec['tidy_score']} ({rec['tidy_level']})")
    print(f"Decision: {rec['decision']}")
    print(f"Detected objects: {len(detections)}")
    print("\nReasons:")
    for r in rec["reasons"]:
        print(f"- {r}")
    print("\nSuggestions:")
    for s in rec["suggestions"]:
        print(f"- {s}")

    grouped = rec.get("grouped_actions", [])
    if grouped:
        print("\nAction Table:")
        print("| Action | Zone | Items |")
        print("|---|---|---|")
        for row in grouped:
            print(f"| {row['action']} | {row['zone']} | {row['items']} |")


if __name__ == "__main__":
    main()

