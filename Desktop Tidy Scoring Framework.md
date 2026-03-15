# Desktop Tidy Scoring Framework

## 1. Overall Scoring Method

The system evaluates desk tidiness using a **0–100 score**.

```
Tidy Score = 100 − Total Penalty
```
Where:
```
Total Penalty =
Object Load Penalty + Category Penalty + Workspace Obstruction Penalty + Spatial Disorder Penalty
```
---
# 2. Object Load Penalty (Number of Objects)
Too many objects on the desk usually indicate clutter.
| Number of Detected Objects | Penalty |
| -------------------------- | ------- |
| ≤ 7                        | 0       |
| 8 – 10                     | 5       |
| 10 – 13                    | 10      |
| 14 – 17                    | 15      |
| > 17                       | 20      |
---
# 3. Category Penalty (Object Type)
Different categories contribute differently to clutter.
| Category        | Penalty per Object | Description                                 |
| --------------- | ------------------ | ------------------------------------------- |
| Core Work Items | 0                  | Essential working tools                     |
| Study Items     | 1                  | Normal study-related objects                |
| Temporary Items | 3                  | Short-term items that may cause distraction |
| Clutter Items   | 6                  | Objects highly associated with mess         |
### Category Definition
| Category        | Objects                                          |
| --------------- | ------------------------------------------------ |
| Core Work Items | laptop, mouse, book, notebook                    |
| Study Items     | pen, pencil, phone, eraser, scissor, sticky note |
| Temporary Items | mug, bowl, bottle, earphones                     |
| Clutter Items   | cable, spitball, ring-pull can                   |
---
# 4. Workspace Obstruction Penalty
The **central workspace area** should remain clear.
Workspace Zone Definition:
```
Central 40% area of the desk image
```
Penalty rules:
| Object Category | Penalty if Located in Workspace |
| --------------- | ------------------------------- |
| Core Work Items | 0                               |
| Study Items     | 3                               |
| Temporary Items | 6                               |
| Clutter Items   | 10                              |
Example:
* Mug in front of laptop → spill risk
* Cable in central workspace → obstruction
---
# 5. Spatial Disorder Penalty
Messy spatial arrangements also affect perceived tidiness.
## 5.1 Object Overlap
If bounding boxes overlap significantly:
| Condition       | Penalty       |
| --------------- | ------------- |
| Overlap > 30%   | 2 per overlap |
| Maximum penalty | 10            |
---
## 5.2 Object Dispersion
Measure how scattered objects are across the desk.
| Dispersion Level | Penalty |
| ---------------- | ------- |
| Low              | 0       |
| Medium           | 5       |
| High             | 10      |
---
# 6. Tidy Level Interpretation
| Score    | Level          | Interpretation                 |
| -------- | -------------- | ------------------------------ |
| 85 – 100 | Tidy           | No cleaning needed             |
| 70 – 84  | Slightly Messy | Minor organisation recommended |
| 50 – 69  | Messy          | Cleaning recommended           |
| < 50     | Very Messy     | Immediate tidying required     |
---
# 7. Explanation Output
The system should provide interpretable explanations.
Example:
```
Tidy Score: 63
Reasons:
• 3 temporary items detected (mug, bottle)
• Cable clutter detected
• 2 objects blocking the central workspace
```
---

**System Pipeline**
```
Image Input
   ↓
Object Detection
   ↓
Object Categorisation
   ↓
Tidy Score Calculation
   ↓
Explanation Generation
   ↓
Tidying Suggestions
```
