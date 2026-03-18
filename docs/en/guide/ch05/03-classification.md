---
title: 5.3 Classification and Detection
description: Understand medical image classification and detection with one consistent template: when full contours are unnecessary and how models first decide, then localize
---
# 5.3 Classification and Detection

## Opening question
This section answers: **when we do not need pixel-level contours, how do we answer “is there disease,” “what disease is it,” and “roughly where is it.”**

Readers often face pain points like these:

- chest X-ray screening only needs a first abnormal/normal decision;
- emergency triage cares more about high recall than fine delineation;
- large-scale screening needs fast routing before detailed review.

So not every problem should jump straight to segmentation. In many medical AI workflows, classification and detection are the first gate.

---

## Intuitive explanation
A simple way to frame this section is as three levels of questions:

- **classification** answers “whether / what”;
- **detection** answers “where”;
- **segmentation** answers “where exactly is the boundary.”

Classification models focus on image-wide patterns related to diagnosis. Detection builds on classification and learns to provide rough location information as well.

The hard part in medicine is not only model architecture. It is also that:

- positive cases are much rarer than negative ones;
- tiny lesions may occupy only a tiny fraction of the image;
- clinicians want more than a score—they want something they can inspect and question.

![Chest X-ray classification](/images/ch05/CheXNet.png)
*Figure: classification emphasizes image-level diagnostic patterns, while detection adds explicit localization of suspicious regions.*

---

## Core method
This section keeps only 4 key ideas.

### 1. Decide whether you need a global label or candidate locations
If the goal is screening, triage, or first-pass warning, classification may be enough. If clinicians need quick review of suspicious regions, detection is often a better fit.

### 2. Put recall first when the task demands it
In medical screening especially, it is often better to flag extra suspicious cases than to miss a serious lesion.

### 3. Handle class imbalance explicitly
Rare positives are the norm. Resampling, weighted losses, and threshold tuning often matter earlier than swapping the backbone.

### 4. Make outputs reviewable
Probabilities, heatmaps, confusion matrices, ROC/AUC curves, and error analysis all help clinicians judge whether the model is trustworthy.

---

## Typical case
### Case 1: Binary or multi-label chest X-ray classification
- **Goal**: predict normal/abnormal, pneumonia, effusion, nodules, and similar labels.
- **Difficulty**: positive cases are sparse and many abnormalities occupy only a small region.
- **Local code**: `src/ch05/medical_image_classification/main.py`.

### Case 2: Lesion detection as a triage entry point
- **Goal**: provide candidate boxes for a clinician or a later segmentation model to review.
- **Suitable for**: lung nodules, breast calcifications, suspicious fractures, and similar findings.
- **Section focus**: build the intuition for classification first, then see why detection must additionally learn location.

### Case 3: Model interpretation and error analysis
- **Goal**: understand not just the score, but where the model is looking.
- **Suggested outputs**: prediction probabilities, confusion matrix, ROC/AUC, heatmaps, or attention maps.
- **Local result file**: `src/ch05/medical_image_classification/output/medical_classification_report.json`.

---

## Practice tips
The text only keeps short fragments for intuition; the full network, training loop, and visualizations are in the local scripts.

### 1. Minimal classification head
```python
import torch.nn as nn


def classification_head(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
```

### 2. Weighted cross-entropy for imbalance
```python
import torch
import torch.nn.functional as F


def weighted_ce(logits, targets, class_weights):
    return F.cross_entropy(logits, targets, weight=torch.tensor(class_weights))
```

### 3. Convert logits into readable probabilities
```python
import torch


def to_probabilities(logits):
    return torch.softmax(logits, dim=1)
```

### 4. Checks worth doing first
- do not rely on accuracy alone; also inspect recall, precision, F1, and AUC;
- analyze false positives and false negatives separately;
- verify that heatmaps really land near pathology;
- define clearly how classification and detection divide labor in the workflow.

---

## Summary
In this section you learned that **classification and detection handle screening, routing, and coarse localization when fine contours are unnecessary.**

The next section moves to augmentation and restoration because the first three sections quietly assume the input is already “usable.” In reality, we still need to ask: **what do we do when data are scarce, image quality is poor, or contrast is not strong enough?**
