---
title: 5.4 Image Augmentation and Restoration
description: Understand medical image augmentation and restoration with one consistent template: not as demos, but as responses to small data and degraded images
---

# 5.4 Image Augmentation and Restoration

## Opening question
This section answers: **when data are scarce, images are degraded, contrast is weak, or quality is unstable, how can we make models more robust and images more usable without damaging medical meaning.**

The real pain point is usually not “how do I produce an impressive demo image,” but rather:

- the training set is small and the model overfits quickly;
- chest X-rays or CT slices have weak local contrast and subtle lesions are hard to see;
- MRI or CT images include noise, bias field, or local degradation that hurts both reading and training.

So the goal here is not visual showmanship. It is to return to the underlying problem: **which changes help the task, and which ones actually harm it.**

---

## Intuitive explanation
The easiest confusion in this topic is that augmentation and restoration are not the same thing.

- **Data augmentation** mainly serves training and tries to expose the model to more reasonable variation.
- **Image enhancement / restoration** mainly serves the image itself and tries to make important structures easier to see or degradation less severe.

A simple memory trick is:

- augmentation asks: **how do we stop the model from being too fragile?**
- restoration asks: **how do we stop the image from being too hard to read?**

Their shared constraint is the same: **never improve visibility at the cost of medical realism.**

---

## Core method
This section only keeps 4 key ideas.

### 1. Separate “expand the training distribution” from “improve the original image”
If the problem is small data and weak generalization, think augmentation first. If the problem is poor visibility or strong degradation in the image itself, think enhancement or restoration first.

### 2. Changes must stay close to realistic acquisition variation
Small rotations, mild noise, and limited contrast changes are often reasonable. Large geometric distortions or dramatic brightness shifts may break anatomy or lesion shape.

### 3. “Looks better” is not the same as “works better”
The purpose of CLAHE, denoising, or bias correction is not to make the most dramatic-looking image. It is to make task-relevant structures easier to interpret.

### 4. Keep the operation set small and explainable
Instead of stacking many strong transforms, it is usually better to keep a few interpretable ones first and validate whether they really help on held-out data or downstream tasks.

---

## Typical case
### Case 1: Data augmentation during training
- **Pain point**: small sample size, center differences, and models memorizing training details.
- **Typical operations**: small rotations, translations, scaling, mild noise, contrast perturbation, elastic deformation.
- **Local code**: `src/ch05/medical_image_augmentation/main.py`, `src/ch05/medical_image_augmentation/simple_augmentation.py`.

### Case 2: Visibility enhancement for X-ray or low-contrast images
- **Pain point**: boundaries and textures are hard to see, so both humans and models miss detail.
- **Typical method**: CLAHE.
- **Local code**: `src/ch05/clahe_enhancement/main.py`.

### Case 3: Restoration for degraded image quality
- **Pain point**: bias field, uneven illumination, or local degradation makes the same tissue look inconsistent.
- **Related code**: `src/ch05/visualize_bias_field/main.py`, `src/ch05/n4itk_bias_correction/main.py`.

---

## Practice tips
The body keeps only short intuition-building snippets; full implementations and parameter sweeps are under `src/ch05/`.

### 1. A minimal geometric augmentation snippet
```python
from skimage.transform import rotate


def small_rotation(image, angle=5):
    return rotate(image, angle, preserve_range=True)
```

### 2. A minimal intensity perturbation snippet
```python
import numpy as np


def adjust_contrast(image, factor=1.1):
    mean = np.mean(image)
    return (image - mean) * factor + mean
```

### 3. The key CLAHE call
```python
import cv2


def run_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)
```

### 4. Minimal checks to do first
- after augmentation, do anatomy and lesion shape still make sense;
- if there is a mask, is it still synchronized with the image;
- does augmentation help validation performance rather than just training performance;
- does restoration make the image easier to read, not just brighter or sharper.

---

## Summary
In this section you learned that **augmentation and restoration are not about making pretty demos; they are about addressing real pain points such as small datasets, weak contrast, and degraded image quality with a small number of medically reasonable operations.**

The next section moves to new paradigms because once we understand preprocessing, segmentation, classification, and augmentation, we can better judge what SAM and generative AI are really changing—and what they are not.
