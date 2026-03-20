# ComputerVision-DeskTidier

Desktop organization project using computer vision.

## Description

We built a vision-based desktop tidiness decision system that combines object detection, scoring, and actionable cleanup guidance. In this repository, we implemented an end-to-end pipeline with a YOLO-based desk object detector, a rule-based tidy scoring framework (object load, category mix, central workspace obstruction, overlap, and alignment), batch CSV export for evaluation, and recommendation modules that translate detections into clear, grouped cleanup actions.

## What We Achieved

Beyond the GitHub codebase, we also developed a demo web page to showcase real user interaction: users first choose their dominant hand (left or right), then upload a desk photo, after which the system detects desk objects, decides whether tidying is needed and how messy the desk is (with a scoring table), and finally generates both visual and text-based organizing plans (image + language guidance). Together, these components form a practical decision-support tool for personalized desk organization.

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

## Evidence

一、项目背景与目标（1 段）
说明课程/作业要求是什么，你们要解决的真实问题是什么（例如：从单张桌面照片判断乱不乱、给出可执行整理建议）。写清输入（照片）、输出（分数、等级、文字建议、可选可视化），以及评价标准在你们心里是什么（可复现、可解释、和人的直觉大致一致等）。

二、总体技术路线（1 小段 + 可配一张架构图）
用一段话概括流水线：数据 → YOLO 检测 → 规则化 Tidy Score →（可选）分类门控 → 语言建议 → 可视化 / Demo。说明各模块谁依赖谁（例如：分数完全依赖检测结果；建议依赖分数与标签）。若交电子版，可画一张方框图放在这里。

三、数据与检测基线：YOLOv8 训练
3.1 数据集
写数据集来源（如 Roboflow 导出、类别列表、训练/验证划分）、标注格式（YOLO）、与桌面整洁任务相关的类别设计（Core / Study / Temporary / Clutter 等与标签的对应关系）。

3.2 模型与训练设置
模型选型（如 YOLOv8m）、训练产物路径（如 runs/detect/.../weights/best.pt）、验证集上关键指标（mAP、P、R 等，从 results.csv 或验证报告摘一行最终值）。

3.3 推理参数作为「第一轮迭代」
说明你们发现仅靠默认 conf/iou/imgsz 会出现漏检、重复框等问题，因此做了多轮试错，记录例如：conf=0.4、iou=0.5、imgsz=640 等，并用少量样例图（如 desk_017、desk_038）说明「调整前 vs 调整后」检测数量或观感的变化。这一节的要点是：检测质量直接约束后面所有分数，所以迭代从这里开始。

四、整洁评分：从框架文档到代码实现
4.1 评分哲学
用一段话写：Tidy Score = 100 − Total Penalty，Penalty 由若干维度相加，目标是可解释而不是黑盒。

4.2 各维度对应「规则」与「用到的信息」  
按模块写（不必很长，每段 3～5 句即可）：

物体数量（Object Load）：只依赖检测个数与分档表。
类别（Category）：标签映射到 Core/Study/Temporary/Clutter，每类单位惩罚不同。
中央工作区（Workspace）：定义中央 60% 区域；用框的中心点是否落入该区域；强调你们从「每物扣一次」迭代为「按类别在区内计罚」的原因（更合理、减少重复惩罚）。
空间杂乱（Spatial Disorder）：包含分散度与重叠；说明重叠最初用框 IoU，后来迭代为 mask/前景辅助 + 细长物体在平面上的补充规则（减少并排笔误报、提高笔在书上等情形）。
对齐（Alignment）：按类别选用 minAreaRect 或 HoughLinesP，与桌面参考角比较；说明哪些类别不测角度。
4.3 等级划分
说明 Tidy / Slightly Messy / Messy / Very Messy 与分数区间的对应，作为面向用户的解释层。

五、工程迭代过程（建议单独成节，这是老师最想看的「证据链」）
用时间线或版本线写，每一段包含：问题 → 你们怎么做 → 结果/仍存在的问题。

可按下面顺序组织（与你们对话历史一致，可按实际删改）：

Baseline：只有检测 + 简单打分或粗糙规则。
检测参数调优：针对具体图片调 conf/iou/imgsz，改善漏检与重复框。
工作区与遮挡逻辑：中央区域比例调整；工作区惩罚改为按类。
重叠逻辑迭代：IoU + 包含度 → 更严/更松的阈值 → mask 重叠 → 细长物体规则；可举 1～2 张典型图（如 desk_065 并排笔、desk_051 笔在书上）。
对齐：从「无」到按类别估角度并计入 penalty。
分类器门控（可选）：ResNet 二分类用于批量流程或跳过「已整洁」图；说明不能替代主观整洁评价。
语言建议与表格化输出：从逐条罗列到按区域/动作分组。
可视化与 Demo：检测图、plan/relayout、before/after 条带、一键 teacher_demo / 网页 Demo 等。
仓库与文档整理：scoring_module、README 一键命令、文档索引等。
这一节里，每个迭代点最好有一句「证据」：例如某张图 overlap 从 0 变为 1、或 CSV 中一批图的分数分布变化——不必多，2～3 个 concrete 例子就够。

六、评估与局限性（诚实收尾）
6.1 检测侧评估
YOLO 在验证集上的 mAP/P/R（已给数字可引用）；若做了 v3 vs v4 或不同 conf 的对比，可简述。

6.2 整洁分与主观感受
说明规则分是设计产物，与「每个人心中的整洁」不完全一致；举 1 个可能分数与直觉不一致的情况（检测错、类别映射争议、工作区定义等）。

6.3 系统不能做什么
一段话写清：不理解意图、不保证物理可行、对遮挡/反光/训练外类别脆弱等——与「evaluation」段落一致即可，显得严谨。

七、最终交付物（给老师一个 checklist）
列出可提交内容：代码仓库结构、主文档（如 Scoring Framework.md）、训练权重位置说明、一键运行命令（README 里那行）、样例输出（检测图、分数打印、before/after 图）、若有 网页 Demo 则写访问方式或截图位置。

八、结论（1 段）
总结：项目完成了「检测 + 可解释评分 + 建议 + 可视化」的闭环；中间通过多轮参数与规则迭代把失败案例变成可讲清楚的设计决策；未来可在数据、模型、与主观 user study 上继续改进。

## Evaluation

The application is designed as a vision-based desk tidiness assistant that generates a structured evaluation from a single image. It uses a custom-trained YOLO detector to identify common desk objects and applies a transparent rule-based scoring model to convert detections into a 0–100 tidiness score, a qualitative label, and actionable suggestions. In addition, a lightweight binary classifier is used to determine whether the scene is already tidy, allowing the system to skip deeper analysis when appropriate. This end-to-end pipeline—detection, scoring, explanation, and recommendation—makes the system function as a simple decision-support tool rather than just a detector.

A key strength of the system is that it operationalises the abstract concept of “clutter” in a structured way. The score combines several dimensions, including object count, semantic categories (core items vs. temporary items), spatial risk (whether objects occupy the central work area), geometric clutter (overlap and dispersion), and approximate alignment of elongated objects. The system also produces rich and interpretable outputs, such as annotated detection images, penalty breakdowns explaining score reductions, grouped action suggestions, and visualised “after” layouts. These outputs help users understand why a score is assigned and how to improve their workspace.

However, the system has several limitations. First, its performance depends heavily on the accuracy of the object detector. Missed detections or incorrect labels can directly affect the score. Second, the scoring system is based on predefined rules, which reflect design assumptions and may not generalise across different cultures, desk sizes, or personal work styles. Third, object overlap and segmentation are only approximated, making it difficult to handle cases such as transparent objects, heavy occlusion, or unseen categories. Fourth, orientation estimation is relatively coarse and may be unreliable for irregular shapes or complex backgrounds. Fifth, the generated layouts are only illustrative suggestions and do not consider physical constraints such as gravity, cable length, ergonomics, or user habits. Finally, the binary “tidy vs. cluttered” judgement is based on a simple classifier and should not be interpreted as an objective measure of productivity or organisation.

Overall, this system is best understood as an interpretable prototype rather than a complete solution. It demonstrates how visual detection can be combined with rule-based reasoning to provide structured feedback and actionable suggestions. At the same time, it highlights the trade-offs involved in automating subjective concepts such as tidiness, and the continued importance of human judgement in understanding context and personal needs.

## Personal Statement -- Yiwen Cao

I collaborated with Linda to develop the initial concept of this project. She was designing a robot called MicroTidy, and we positioned our system as its “eyes and brain”, responsible for perception and decision-making. 

As my first attempt at building a computer vision pipeline from scratch, I chose to construct the dataset myself rather than rely on existing ones, as they did not fully match our context. We collected around 150 desk images and jointly annotated them, with me defining 20 object categories and annotating 26 images. I then trained a YOLOv8 model to detect common desk objects such as pens, books, cups, and cables. After four rounds of iteration, the model achieved Precision 0.783, Recall 0.816, and [mAP@0.5](mailto:mAP@0.5) of 0.851.

After the detection stage, I developed the Tidy Score module, which evaluated desk organisation across Object Load, Category, Workspace Obstruction, Spatial Disorder, and Alignment. During this process, I realised that some aspects of tidiness are difficult to capture through rule-based methods alone. In particular, Spatial Disorder and Alignment were the most challenging. For overlap detection in Spatial Disorder, I refined the method from basic bounding box IoU and containment rules to a mask-assisted approach, which reduced cases where large bounding boxes were mistaken for real overlap. I also introduced additional rules for elongated objects on flat surfaces, such as pens placed on books, using axis coverage and area proportion to reduce false positives and improve detection accuracy. For Alignment, I used different visual methods for different object categories: rectangular objects such as laptops, books, notebooks, phones, and sticky notes were analysed using OpenCV’s minAreaRect, while elongated objects such as pens, pencils, markers, and scissors were analysed using HoughLinesP. Irregular objects such as mugs and cables were excluded from angle estimation. Even with these refinements, I believe both modules still have room for improvement.

One important limitation I identified was that some implicit human judgements of tidiness, such as subtle visual alignment between a cup and the edge of a laptop, are difficult to detect reliably with hand-crafted rules and may be better addressed through machine learning. To improve the overall system, we adopted a modular two-stage strategy: a binary classifier developed by Linda first determines whether the desk needs tidying, and only if tidying is needed does the system proceed to detailed scoring. This made the system more robust and interpretable. 

I also worked on the text-based recommendation component, producing concise and practical tidying suggestions, and collaborated with Linda on a simple demo website, where I built the overall framework and she focused on visual refinement and interface details.

Through this project, I not only developed technical skills in computer vision and system building, but also gained a stronger understanding of how to design an interpretable and extensible system. One key lesson was that modularising detection, classification, scoring, and recommendation made the system easier to explain, test, and improve. At the same time, the project showed me the limitations of purely rule-based approaches in capturing human perceptions of order. 

The current system still has limitations, including relatively simple layout rules that only consider left- and right-handed preferences in a basic way, and the lack of safety-aware logic, such as detecting risky placements like a cup positioned too close to a laptop. If given more time, I would like to develop a more personalised strategy system and explore the use of large language models to generate more natural, context-sensitive recommendations.

## Personal Statement - Linda

I worked closely with Yiwen to design the initial idea about the project. As I'm thinking design a MicroTidy robot, we decided to position this system as the “eyes and brain” of the robot, responsible for perception and decision making.

This project was my first time building a computer vision pipeline from zero. One of the first challenges we faced was the lack of a suitable dataset. We collected our own desk data by photographing and constructed a dataset of around 150 images. Initially, I experimented with pretrained ImageNet models and COCO-based YOLO detection, but the performance was poor because those models were not tailored to desk-level object understanding. To address this, we worked on dataset annotation and I labeled 27 images myself, defining 20 object categories. This step was important because it allowed the model to learn meaningful desk-related semantics rather than generic object classes. After training, the detection performance improved significantly from 4% to 93%.

I then focused on building a binary classifier to determine whether a desk requires tidying. The classifier achieved very high accuracy, correctly classifying 138 out of 139 images. The only misclassified case was also ambiguous even for human judgement, which suggests that the model has learned a reasonable decision boundary. In our system, if the classifier outputs “tidy”, no further action is taken. If it outputs “untidy”, the image is passed to Yiwen’s scoring model for detailed analysis.

After that, I developed the tidy suggestion system. This includes both textual and visual outputs. The textual suggestions are generated based on interpretable factors such as object count, spatial distribution, overlap, and alignment. The visual output reconstructs a reorganised desk layout by grouping objects into functional zones, providing an intuitive “after” view. This combination allows the system to be both explainable and user-friendly. We also built a website to show our demo.

Through this project, I learned not only technical skills but also how to design a system. One key insight is that real-world data is messy and often incomplete. For example, objects may be partially occluded or located at the edge of the image, which makes perfect segmentation impossible. This forced me to think beyond ideal pipelines and design fallback strategies, such as simplifying the visual output when object extraction is unreliable.

In terms of design decisions, we chose a modular pipeline consisting of detection, classification, scoring, and suggestion generation. This makes the system interpretable and easy to extend. We also deliberately separated binary decision making from fine-grained scoring, which improved robustness and clarity.

There are also several limitations in our current system. Due to time constraints, our layout rules are relatively simple and only consider left- and right-handed preferences in a basic way. With more time, I would like to make the system more personalised by allowing users to define their own desk zones and preferences. I would also integrate a large language model to generate more natural and context-aware suggestions.

Additionally, the visual reconstruction could be improved. Currently, some small or thin objects such as pens are difficult to segment cleanly, especially when partially occluded. In future work, I would improve this by using more advanced segmentation models or by designing a hybrid representation that combines icons and extracted object patches.