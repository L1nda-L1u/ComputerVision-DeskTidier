# Desktop Tidy Scoring Framework

## 1. Overall Scoring Method

The system evaluates desk tidiness using a **0–100 score**.

```
Tidy Score = 100 − Total Penalty
```

Where:

```
Total Penalty = Object Load Penalty + Category Penalty + Workspace Obstruction Penalty + Spatial Disorder Penalty + Alignment Penalty
```

---

# 2. Object Load Penalty (Number of Objects)

Too many objects on the desk usually indicate clutter.

| Number of Detected Objects | Penalty |
| -------------------------- | ------- |
| ≤ 8                        | 0       |
| 9 – 12                     | 5       |
| 13 – 15                    | 10      |
| 16 – 18                    | 15      |
| > 18                       | 20      |

---

# 3. Category Penalty (Object Type)

Different categories contribute differently to clutter.

| Category        | Penalty per Object | Objects                                            | Description                                 |
| --------------- | ------------------ | -------------------------------------------------- | ------------------------------------------- |
| Core Work Items | 0                  | laptop, mouse, book, notebook                      | Essential working tools                     |
| Study Items     | 0                  | pen, pencil, phone, eraser, earphones, sticky note | Normal study-related objects                |
| Temporary Items | 2                  | mug, bowl, bottle, scissor                         | Short-term items that may cause distraction |
| Clutter Items   | 6                  | cable, spitball, ring-pull can                     | Objects highly associated with mess         |
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
| Study Items     | 2                               |
| Temporary Items | 6                               |
| Clutter Items   | 10                              |

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

# 6. Alignment Penalty (Object Orientation)

## Alignment Detection

Orientation can be estimated using:

* bounding box angle
* edge detection
* minimum area rectangle (OpenCV)

The **desk orientation** is assumed to be horizontal.

---

## Penalty Rules
| Category        | Penalty per Object |
| --------------- | ------------------ | 
| Core Work Items | 8                  |
| Study Items     | 5                  |
| Temporary Items | 3                  |
| Clutter Items   | 1                  |

---

# 7. Tidy Level Interpretation

| Score    | Level          | Interpretation                 |
| -------- | -------------- | ------------------------------ |
| 85 – 100 | Tidy           | No cleaning needed             |
| 70 – 84  | Slightly Messy | Minor organisation recommended |
| 50 – 69  | Messy          | Cleaning recommended           |
| < 50     | Very Messy     | Immediate tidying required     |

---

# 8. Explanation Output

The system should provide interpretable explanations.

Example:

```
Tidy Score: 61

Reasons:
• 4 temporary items detected
• Cable clutter detected
• 2 objects blocking the workspace
• 3 items misaligned with desk orientation
```
