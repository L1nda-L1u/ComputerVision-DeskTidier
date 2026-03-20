from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    score_script = repo_dir / "Tidy Scoring System.py"

    model_path = r"C:\Users\caoyi\runs\detect\desk_tidy_runs\v3\weights\best.pt"
    image_path = r"C:\Users\caoyi\Desktop\DwBS\DVS\final Project\DeskTidier.v3i.yolov8\test\images\desk_065_jpg.rf.4fb2734a863b527113972e146b70b277.jpg"

    save_project = r"C:\Users\caoyi\runs\desk_tidy_score_single"
    save_name = "desk_065"

    # Run the scoring script via Python so you don't need to type anything in cmd.
    cmd = [
        sys.executable,
        str(score_script),
        "--model",
        model_path,
        "--source",
        image_path,
        "--conf",
        "0.25",
        "--iou",
        "0.5",
        "--imgsz",
        "640",
        "--device",
        "cpu",
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

