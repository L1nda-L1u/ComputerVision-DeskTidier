from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    score_script = repo_dir / "Tidy Scoring System.py"

    model_path = r"C:\Users\caoyi\Documents\GitHub\ComputerVision\runs\detect\desk_tidy_runs\v4_yolov8m_roboflow_style\weights\best.pt"
    image_path = r"C:\Users\caoyi\Desktop\DwBS\DVS\final Project\DeskTidier.v3i.yolov8\test\images\desk_065_jpg.rf.4fb2734a863b527113972e146b70b277.jpg"

    save_project = r"C:\Users\caoyi\runs\desk_tidy_score_single"
    save_name = "desk_065"
    conf = 0.4
    iou = 0.5
    imgsz = 640

    # Quick detection preview: same style as model("test.jpg", conf=..., iou=..., imgsz=...)
    model = YOLO(model_path)
    results = model(image_path, conf=conf, iou=iou, imgsz=imgsz, device="cpu")
    for idx, r in enumerate(results, start=1):
        print(f"Result {idx}: boxes={len(r.boxes)}")
        if r.boxes is not None and len(r.boxes) > 0:
            cls_counter = Counter(int(c) for c in r.boxes.cls.tolist())
            name_map = r.names
            pretty = {name_map[k]: v for k, v in sorted(cls_counter.items())}
            print(f"Class counts: {pretty}")

    # Run the scoring script via Python so you don't need to type anything in cmd.
    cmd = [
        sys.executable,
        str(score_script),
        "--model",
        model_path,
        "--source",
        image_path,
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--imgsz",
        str(imgsz),
        "--device",
        "cpu",
        "--estimate-object-angle",
        "--save",
        "--save-project",
        save_project,
        "--save-name",
        save_name,
    ]

    subprocess.run(cmd, check=True)
    annotated_path = Path(save_project) / save_name
    print(f"\nAnnotated image folder: {annotated_path}")


if __name__ == "__main__":
    main()

