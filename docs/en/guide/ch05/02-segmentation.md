---
title: 5.2 Segmentation: U-Net and its variants
description: Understand medical image segmentation with one consistent template: why pixel-level answers matter and why U-Net works
---
# 5.2 Segmentation: U-Net and its variants

## Opening question
This section answers: **how do we make a model know not only whether disease exists, but also where the lesion or organ is and where its boundary ends.**

Readers often feel classification is not enough in cases such as:

- measuring tumor volume;
- outlining the lung field precisely;
- defining where a radiotherapy target should stop slice by slice;
- describing the spatial relation between lesions and nearby anatomy.

All of these show that clinical work often needs more than a global label. It needs **pixel-level or voxel-level localization**.

---

## Intuitive explanation
Before thinking of segmentation as a complicated network, it is better to think of it as: **turning image reading into measurable regions.**

Classification answers whether something is present. Segmentation goes further and answers:

- where the region is;
- how large it is;
- what the volume is;
- how its boundary relates to nearby structures.

U-Net became classic not because it is simply deep, but because it captures one core tension:

- deep features are better at saying **what** something is;
- shallow features are better at preserving **where** it is.

So the network extracts semantics on the way down and reconnects spatial detail on the way back up.

![U-Net architecture evolution](/images/ch05/03-unet-architecture-en.png)
*Figure: the key to U-Net is not just downsampling and upsampling, but the skip connections that bring boundary detail back.*

---

## Core method
This section keeps only 4 key ideas.

### 1. Be clear about the output
Segmentation usually outputs not a single score, but a probability map or mask at the same scale as the input, or one that can be mapped back to the original image.

### 2. Preserve semantics and boundaries together
The encoder extracts high-level semantics; the decoder restores spatial resolution; skip connections prevent small structures and boundaries from disappearing completely during downsampling.

### 3. Keep labels perfectly aligned with images
In segmentation, one of the biggest failure modes is not weak augmentation but desynchronized transforms between image and mask. Once they drift apart, the model is no longer learning true boundaries.

### 4. Do not judge quality by loss alone
Dice, IoU, sensitivity, connected-component behavior, and post-processing results often say more about practical segmentation quality than training loss by itself.

---

## Typical case
### Case 1: Lung field segmentation
- **Goal**: obtain a lung mask from chest CT.
- **Value**: provides an ROI for later nodule detection, infection analysis, and quantitative measurement.
- **Local code**: `src/ch05/lung_segmentation_network/main.py`.

### Case 2: Organ or tumor segmentation
- **Goal**: precise contours for liver, pancreas, brain tumor, prostate, and similar targets.
- **Value**: volume estimation, disease burden tracking, and radiotherapy target delineation.
- **Extensions**: 2D U-Net, 3D U-Net, Attention U-Net, nnU-Net.

### Case 3: Augmentation in segmentation tasks
- **Goal**: make image and mask undergo the same geometric transform.
- **Local code**: `src/ch05/medical_segmentation_augmentation/main.py`.
- **Reminder**: when the image changes, the label must change in sync.

---

## Practice tips
The main text only keeps short code fragments for intuition; full training, evaluation, and visualization live in `src/ch05/`.

### 1. A minimal convolution block
```python
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )
```

### 2. The key skip-connection action: concatenation
```python
import torch


def fuse_skip(upsampled, shallow_feature):
    return torch.cat([upsampled, shallow_feature], dim=1)
```

### 3. A minimal Dice implementation
```python
import torch


def dice_score(pred_mask, true_mask, eps=1e-6):
    inter = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    return (2 * inter + eps) / (union + eps)
```

### 4. Checks that are easy to forget
- whether every image really matches the right mask;
- whether preprocessing kept the label aligned;
- whether small lesions vanish during resizing;
- whether post-processing removes true lesions by mistake.

---

## Summary
In this section you learned that **segmentation turns “seeing a lesion” into “outlining a measurable region,” and that the core intuition of U-Net is to preserve semantics and boundaries at the same time.**

The next section moves to classification and detection because not every workflow needs a fine contour first. Many clinical pipelines start by asking: **is there disease, what is it, and roughly where should we look?**
