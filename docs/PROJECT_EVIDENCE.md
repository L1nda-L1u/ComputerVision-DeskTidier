The system follows a linear pipeline with optional branches. First, a **YOLOv8** detector trained on our desk dataset predicts bounding boxes and class labels. Second, a **rule-based tidy scoring module** maps detections and geometry to penalties and a 0–100 score. Third, a **language module** turns penalties and labels into short reasons and zone-based suggestions. Fourth, **visual modules** draw movement plans, schematic “after” layouts, and a relayout image; our `run_pipeline` / `teacher_demo` scripts also save a **detection plot**, a **before/after strip**, and related PNGs. Optionally, a **binary classifier** (ResNet18, tidy vs untidy) can gate batch jobs so that already-tidy images are skipped—this is a workflow optimisation, not the definition of tidiness.

System architecture and data flow：Dataset → Train YOLO → Infer → TidyScore → Language → Visuals / Demo

We used a YOLO-format desk dataset (imported from Roboflow-style exports), with train/validation splits and class names aligned to items we care about on a desk: laptops, books, pens, mugs, cables, and so on. These raw labels are later **mapped** in software to higher-level **categories** used only for scoring—Core work items, study items, temporary items, and clutter—so that the same detector can support both “what object is this?” and “how much does this kind of object contribute to mess?”.



We trained a **YOLOv8m** detector; the best weights are stored under `runs/detect/desk_tidy_runs/v4_yolov8m_roboflow_style/weights/best.pt`. On the **validation split at the end of training** (final epoch in `results.csv`), indicative box metrics are approximately:


| Metric (val, last epoch)       | Value  |
| ------------------------------ | ------ |
| Precision (B)                  | ~0.783 |
| Recall (B)                     | ~0.816 |
| mAP@0.5                        | ~0.851 |
| mAP@0.5:0.95                   | ~0.803 |


These numbers describe **localisation and classification quality on the validation set**, not the tidiness score itself. They should be cited when discussing detector reliability; for “does the desk *look* tidy to a human,” separate subjective discussion is needed (Section 7).

**[FIGURE 4 — Placeholder]**  
*Suggested content:* **PR / F1 / mAP curves** from `runs/detect/desk_tidy_runs_eval/v4_test/` (e.g. `BoxF1_curve.png`, `BoxPR_curve.png`) or screenshot from Ultralytics results.  
*Caption example:* “Detection metrics on the evaluation run (v4 test).”

Before trusting any tidy score, we tuned **post-processing** of the detector. Using default thresholds often yielded too few boxes (missed objects) or unstable duplicates. We converged toward settings such as **confidence ≈ 0.4**, **NMS IoU ≈ 0.5**, and **inference size 640**, and checked behaviour on named examples (e.g. images where the human count of objects was known). This iteration is **evidence-driven**: we compared class histograms and box counts against expectation, not only aggregate mAP.

**[FIGURE 5 — Placeholder]**  
*Suggested content:* Side-by-side **same image** with two settings (e.g. low vs tuned `conf`) or table of *detected count vs expected count* for 2–3 filenames.  
*Caption example:* “Effect of confidence threshold on detection count (illustrative).”


**Object load** penalises simply having **many** objects on the desk, using stepped brackets (e.g. more objects → higher penalty up to a cap). **Category penalty** assigns each detected label to a small set of semantic categories (core, study, temporary, clutter) and applies different per-object weights, reflecting that clutter-like items are penalised more heavily than core tools. **Workspace obstruction** uses a **central rectangle** covering the middle **60%** of the image area; objects whose box **centre** falls inside are considered to obstruct the “workspace.” We iterated this from a stricter central region and later refined **how** workspace penalty accumulates—moving from “charge per object” toward **charging per category present in the zone**, so one category does not multiply unfairly with many instances.

**Spatial disorder** has two strands. **Dispersion** measures how spread out objects are across the image and maps to low/medium/high penalty bands. **Overlap** started from bounding-box **IoU** and **containment** rules; we then added **mask-assisted overlap** to better approximate real occlusion than axis-aligned rectangles alone, and **elongated-object heuristics** (e.g. pens on books) using overlap ratios and principal-axis coverage so that small items on large surfaces are not missed, while adjacent similar objects (e.g. two pens side by side) are less often falsely flagged.

**Alignment** adds penalty when an object’s estimated orientation deviates from the assumed desk direction (0°). **Rectangular** items use **OpenCV `minAreaRect`** on the crop; **elongated** items use `**HoughLinesP`**; other categories may skip angle estimation. This was a later iteration: early versions had no alignment term.

**[FIGURE 6 — Placeholder]**  
*Suggested content:* Screenshot of **terminal or log** showing `Tidy Score`, `Total Penalty`, and **penalty breakdown** lines for one image (e.g. from `teacher_demo` output).  
*Caption example:* “Example scored output with interpretable penalty breakdown.”

We treated concrete failures as requirements. For **workspace**, tightening or loosening the central zone and switching to **per-category** obstruction addressed cases where the score felt too harsh or too blind to “how” the desk was blocked. For **overlap**, cases like **many apparent overlaps but IoU-only score zero** led to mask-based metrics; cases like **adjacent pens** led to stricter rules for line-like vs line-like pairs; cases like **a pen on a book** led to elongated-on-surface rules. Each change was motivated by a **named image** or pattern and checked again on a small set of desks.

**[FIGURE 7 — Placeholder]**  
*Suggested content:* **Before/after rule behaviour** on one problematic image (e.g. overlap count or penalty line before vs after a code change)—can be a small table or two cropped detection images.  
*Caption example:* “Iterative refinement driven by failure cases (illustrative).”

We added a **ResNet18** binary classifier (tidy vs untidy) trained on labelled desk crops, optional for **gating** in batch export so “already tidy” images need not be fully rescored. The **language module** groups suggestions by action and target zone (e.g. move stationery items together) rather than one sentence per box. **Visual outputs** include plan/after diagrams and a **relayout** image; the pipeline composes a **before/after** strip comparing the original photo to the suggested layout. Separately, a **web demo** was built so users can choose handedness, upload a photo, and walk through the same story interactively—the implementation lives outside the minimal GitHub core but is part of the full project story.

**[FIGURE 8 — Placeholder]**  
*Suggested content:* `*_detection.png` and `*_before_after.png` from `teacher_demo/` or `pipeline_output/`.  
*Caption example:* “Detection overlay and before/after comparison strip.”

**[FIGURE 9 — Placeholder]**  
*Optional:* Screenshot of the **web demo** (handedness → upload → score → suggestions).  
*Caption example:* “Web demo workflow (if submitted separately).”

## Deliverables checklist


| Item                           | Location / note                                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Scoring specification          | `scoring_module/docs/Scoring Framework.md`                                                                        |
| Main pipeline                  | `scripts/run_pipeline.py`                                                                                         |
| One-command demo               | `README.md` → `python scripts/teacher_demo.py …`                                                                  |
| Trained detector weights       | `runs/detect/desk_tidy_runs/v4_yolov8m_roboflow_style/weights/best.pt` *(may be gitignored; provide if required)* |
| Example CSV / language exports | e.g. `scoring_module/outputs/`, `language_score_results.md`                                                       |
| This evidence narrative        | `docs/PROJECT_EVIDENCE.md`                                                                                        |