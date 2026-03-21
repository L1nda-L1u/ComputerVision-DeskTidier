# ComputerVision-DeskTidier

Desktop organization project using computer vision.

---

## Description

We built a vision-based desktop tidiness decision system that combines object detection, scoring, and actionable cleanup guidance. In this repository, we implemented an end-to-end pipeline with a YOLO-based desk object detector, a rule-based tidy scoring framework (object load, category mix, central workspace obstruction, overlap, and alignment), batch CSV export for evaluation, and recommendation modules that translate detections into clear, grouped cleanup actions.

---

## What We Achieved

Beyond the GitHub codebase, we also developed a demo web page to showcase real user interaction: users first choose their dominant hand (left or right), then upload a desk photo, after which the system detects desk objects, decides whether tidying is needed and how messy the desk is (with a scoring table), and finally generates both visual and text-based organizing plans (image + language guidance). Together, these components form a practical decision-support tool for personalized desk organization.

---

## Quick test (demo)

Run from the **repository root** (requires trained weights at `runs/detect/desk_tidy_runs/v4_yolov8m_roboflow_style/weights/best.pt`):

```bash
python scripts/teacher_demo.py data/images/desk_066.jpg
```

**You get:** `teacher_demo/*_detection.png` (YOLO boxes), terminal **tidy level + score + penalty breakdown + text tidy plan**, and `teacher_demo/*_before_after.png` (original vs suggested layout). Use any image path instead of `data/images/desk_065.jpg` if needed. Add `--left-handed` for a left-handed layout.

### Output Example：

#### Image: desk_066.jpg

#### Classifier: skipped

#### Detected 22 objects:

   phone              conf=0.99  [Study Items], angle=-90.0°
   ring-pull can      conf=0.98  [Clutter Items]
   ring-pull can      conf=0.97  [Clutter Items]
   marker             conf=0.97  [Study Items], angle=89.5°
   pencil             conf=0.97  [Study Items], angle=-29.0°
   scissor            conf=0.96  [Temporary Items]
   bottle             conf=0.96  [Temporary Items]
   pen                conf=0.96  [Study Items], angle=6.3°
   pen                conf=0.95  [Study Items], angle=-73.5°
   spitball           conf=0.95  [Clutter Items]
   mouse              conf=0.94  [Core Work Items]
   marker             conf=0.94  [Study Items], angle=87.1°
   pen                conf=0.93  [Study Items], angle=-55.5°
   marker             conf=0.93  [Study Items], angle=-82.4°
   pen                conf=0.92  [Study Items], angle=-4.6°
   marker             conf=0.91  [Study Items], angle=-80.4°
   marker             conf=0.90  [Study Items], angle=88.6°
   earphones          conf=0.89  [Study Items], angle=-67.2°
   pencil             conf=0.88  [Study Items], angle=-5.8°
   mug                conf=0.87  [Temporary Items]
   tape               conf=0.83  [Temporary Items]
   pencil             conf=0.75  [Study Items], angle=88.8°

#### Tidy Score: 0 (Very Messy)

#### Total Penalty: 108

#### Penalty breakdown (by type):

   · alignment_penalty: 55
   · category_penalty: 20
   · object_load_penalty: 10
   · spatial_dispersion_penalty: 5
   · spatial_overlap_penalty: 0
  workspace_obstruction_penalty: 18
  22 objects detected -> object load penalty 10
  Category counts: Clutter=3, Core=1, Study=14, Temporary=4; category penalty 20
  15 objects in central workspace; blocked categories=3 -> workspace penalty 18
  Overlap pairs (IoU>0.30): 0 -> overlap penalty 0
  Dispersion: Medium -> dispersion penalty 5
  Misaligned objects: 11 -> alignment penalty 55

#### Decision: Tidying needed.

#### Reasons:

   · The central workspace is obstructed by non-essential items.
   · Too many objects are on the desk.
   · Several temporary or clutter-related items are present.
   · Objects are scattered across the desk.
   · Some objects are misaligned with the desk orientation.

#### Suggestions:

   · Clear the central workspace first.
   · Relocate temporary items to the temporary item zone.
   · Regroup scattered items into functional zones.
   · Align objects parallel to the desk edges.
   · Keep in workspace: mouse.
   · Move to stationery zone: earphones, marker, pen, pencil, phone, scissor, tape.
   · Move to temporary item zone: bottle, mug.
   · Remove from desk: ring-pull can, spitball.
Plan image saved: C:\Users\Documents\GitHub\ComputerVision\teacher_demo\desk_066_plan.png
After image saved: C:\Users\Documents\GitHub\ComputerVision\teacher_demo\desk_066_after.png

---

## Evidence

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

### Deliverables checklist

| Item                           | Location / note                                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Scoring specification          | `scoring_module/docs/Scoring Framework.md`                                                                        |
| Main pipeline                  | `scripts/run_pipeline.py`                                                                                         |
| One-command demo               | `README.md` → `python scripts/teacher_demo.py …`                                                                  |
| Trained detector weights       | `runs/detect/desk_tidy_runs/v4_yolov8m_roboflow_style/weights/best.pt` *(may be gitignored; provide if required)* |
| Example CSV / language exports | e.g. `scoring_module/outputs/`, `language_score_results.md`                                                       |
| This evidence narrative        | `docs/PROJECT_EVIDENCE.md`                                                                                        |

---

## Evaluation

The application is designed as a vision-based desk tidiness assistant that generates a structured evaluation from a single image. It uses a custom-trained YOLO detector to identify common desk objects and applies a transparent rule-based scoring model to convert detections into a 0–100 tidiness score, a qualitative label, and actionable suggestions. In addition, a lightweight binary classifier is used to determine whether the scene is already tidy, allowing the system to skip deeper analysis when appropriate. This end-to-end pipeline—detection, scoring, explanation, and recommendation—makes the system function as a simple decision-support tool rather than just a detector.

A key strength of the system is that it operationalises the abstract concept of “clutter” in a structured way. The score combines several dimensions, including object count, semantic categories (core items vs. temporary items), spatial risk (whether objects occupy the central work area), geometric clutter (overlap and dispersion), and approximate alignment of elongated objects. The system also produces rich and interpretable outputs, such as annotated detection images, penalty breakdowns explaining score reductions, grouped action suggestions, and visualised “after” layouts. These outputs help users understand why a score is assigned and how to improve their workspace.

However, the system has several limitations. First, its performance depends heavily on the accuracy of the object detector. Missed detections or incorrect labels can directly affect the score. Second, the scoring system is based on predefined rules, which reflect design assumptions and may not generalise across different cultures, desk sizes, or personal work styles. Third, object overlap and segmentation are only approximated, making it difficult to handle cases such as transparent objects, heavy occlusion, or unseen categories. Fourth, orientation estimation is relatively coarse and may be unreliable for irregular shapes or complex backgrounds. Fifth, the generated layouts are only illustrative suggestions and do not consider physical constraints such as gravity, cable length, ergonomics, or user habits. Finally, the binary “tidy vs. cluttered” judgement is based on a simple classifier and should not be interpreted as an objective measure of productivity or organisation.

Overall, this system is best understood as an interpretable prototype rather than a complete solution. It demonstrates how visual detection can be combined with rule-based reasoning to provide structured feedback and actionable suggestions. At the same time, it highlights the trade-offs involved in automating subjective concepts such as tidiness, and the continued importance of human judgement in understanding context and personal needs.

---

## Personal Statement -- Yiwen Cao

I collaborated with Linda to develop the initial concept of this project. She was designing a robot called MicroTidy, and we positioned our system as its “eyes and brain”, responsible for perception and decision-making. 

As my first attempt at building a computer vision pipeline from scratch, I chose to construct the dataset myself rather than rely on existing ones, as they did not fully match our context. We collected around 150 desk images and jointly annotated them, with me defining 20 object categories and annotating 26 images. I then trained a YOLOv8 model to detect common desk objects such as pens, books, cups, and cables. After four rounds of iteration, the model achieved Precision 0.783, Recall 0.816, and [mAP@0.5](mailto:mAP@0.5) of 0.851.

After the detection stage, I developed the Tidy Score module, which evaluated desk organisation across Object Load, Category, Workspace Obstruction, Spatial Disorder, and Alignment. During this process, I realised that some aspects of tidiness are difficult to capture through rule-based methods alone. In particular, Spatial Disorder and Alignment were the most challenging. For overlap detection in Spatial Disorder, I refined the method from basic bounding box IoU and containment rules to a mask-assisted approach, which reduced cases where large bounding boxes were mistaken for real overlap. I also introduced additional rules for elongated objects on flat surfaces, such as pens placed on books, using axis coverage and area proportion to reduce false positives and improve detection accuracy. For Alignment, I used different visual methods for different object categories: rectangular objects such as laptops, books, notebooks, phones, and sticky notes were analysed using OpenCV’s minAreaRect, while elongated objects such as pens, pencils, markers, and scissors were analysed using HoughLinesP. Irregular objects such as mugs and cables were excluded from angle estimation. Even with these refinements, I believe both modules still have room for improvement.

One important limitation I identified was that some implicit human judgements of tidiness, such as subtle visual alignment between a cup and the edge of a laptop, are difficult to detect reliably with hand-crafted rules and may be better addressed through machine learning. To improve the overall system, we adopted a modular two-stage strategy: a binary classifier developed by Linda first determines whether the desk needs tidying, and only if tidying is needed does the system proceed to detailed scoring. This made the system more robust and interpretable. 

I also worked on the text-based recommendation component, producing concise and practical tidying suggestions, and collaborated with Linda on a simple demo website, where I built the overall framework and she focused on visual refinement and interface details.

Through this project, I not only developed technical skills in computer vision and system building, but also gained a stronger understanding of how to design an interpretable and extensible system. One key lesson was that modularising detection, classification, scoring, and recommendation made the system easier to explain, test, and improve. At the same time, the project showed me the limitations of purely rule-based approaches in capturing human perceptions of order. 

The current system still has limitations, including relatively simple layout rules that only consider left- and right-handed preferences in a basic way, and the lack of safety-aware logic, such as detecting risky placements like a cup positioned too close to a laptop. If given more time, I would like to develop a more personalised strategy system and explore the use of large language models to generate more natural, context-sensitive recommendations.

---

## Personal Statement - Linda

I worked closely with Yiwen to design the initial idea about the project. As I'm thinking design a MicroTidy robot, we decided to position this system as the “eyes and brain” of the robot, responsible for perception and decision making.

This project was my first time building a computer vision pipeline from zero. One of the first challenges we faced was the lack of a suitable dataset. We collected our own desk data by photographing and constructed a dataset of around 150 images. Initially, I experimented with pretrained ImageNet models and COCO-based YOLO detection, but the performance was poor because those models were not tailored to desk-level object understanding. To address this, we worked on dataset annotation and I labeled 27 images myself, defining 20 object categories. This step was important because it allowed the model to learn meaningful desk-related semantics rather than generic object classes. After training, the detection performance improved significantly from 4% to 93%.

I then focused on building a binary classifier to determine whether a desk requires tidying. The classifier achieved very high accuracy, correctly classifying 138 out of 139 images. The only misclassified case was also ambiguous even for human judgement, which suggests that the model has learned a reasonable decision boundary. In our system, if the classifier outputs “tidy”, no further action is taken. If it outputs “untidy”, the image is passed to Yiwen’s scoring model for detailed analysis.

After that, I developed the tidy suggestion system. This includes both textual and visual outputs. The textual suggestions are generated based on interpretable factors such as object count, spatial distribution, overlap, and alignment. The visual output reconstructs a reorganised desk layout by grouping objects into functional zones, providing an intuitive “after” view. This combination allows the system to be both explainable and user-friendly. We also built a website to show our demo.

Through this project, I learned not only technical skills but also how to design a system. One key insight is that real-world data is messy and often incomplete. For example, objects may be partially occluded or located at the edge of the image, which makes perfect segmentation impossible. This forced me to think beyond ideal pipelines and design fallback strategies, such as simplifying the visual output when object extraction is unreliable.

In terms of design decisions, we chose a modular pipeline consisting of detection, classification, scoring, and suggestion generation. This makes the system interpretable and easy to extend. We also deliberately separated binary decision making from fine-grained scoring, which improved robustness and clarity.

There are also several limitations in our current system. Due to time constraints, our layout rules are relatively simple and only consider left- and right-handed preferences in a basic way. With more time, I would like to make the system more personalised by allowing users to define their own desk zones and preferences. I would also integrate a large language model to generate more natural and context-aware suggestions.

Additionally, the visual reconstruction could be improved. Currently, some small or thin objects such as pens are difficult to segment cleanly, especially when partially occluded. In future work, I would improve this by using more advanced segmentation models or by designing a hybrid representation that combines icons and extracted object patches.