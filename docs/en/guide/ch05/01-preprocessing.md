---
title: 5.1 Preprocessing (with modality differences in mind)
description: Understand medical image preprocessing with one consistent template: why it exists, what intuition matters, and how it supports downstream tasks
---
# 5.1 Preprocessing (with modality differences in mind)

## Opening question
This section answers: **why raw medical images cannot be fed directly into a model, and why different modalities need different preprocessing routes.**

Readers usually run into three pain points first:

- even CT scans from different hospitals can have different spacing, orientation, and slice thickness;
- the same tissue can look very different across MRI scanners or protocols;
- a model may appear to train well while actually learning scanner differences instead of disease patterns.

So preprocessing is not just about “cleaning up images.” It is about turning raw data into inputs that are **comparable, learnable, and reproducible**.

---

## Intuitive explanation
A simple way to think about preprocessing is:

1. **put every scan on the same ruler first;**
2. **then suppress variations that are not relevant to the task.**

If we skip that first step, the model will see many differences that have nothing to do with disease:

- voxel size differences;
- orientation differences;
- intensity scales that are not comparable at all;
- artifacts, bias field, or noise hiding the real signal.

That is why preprocessing is not meant to make images look prettier. Its job is to help later segmentation, classification, and detection models learn the **stable medical signal** instead of nuisance variation.

![Medical image preprocessing hierarchy](/images/ch05/01-preprocessing-hierarchy-en.png)
*Figure: preprocessing usually starts with spatial normalization, then applies modality-specific intensity handling, and only then connects to downstream tasks.*

---

## Core method
This section only keeps the 4 most important moves.

### 1. Normalize space
Different scans can have very different spacing, orientation, and size. Common steps include:

- resampling to a target voxel spacing;
- standardizing orientation;
- cropping an ROI or padding to a fixed size.

### 2. Normalize intensity
Different modalities follow different rules, so one recipe should not be forced onto all images:

- **CT** usually preserves HU meaning and uses clipping, windowing, or normalization;
- **MRI** often needs bias correction and intensity standardization;
- **X-ray** often depends more on local contrast and dynamic range adjustment.

### 3. Reduce obvious interference
Noise, bias field, metal artifacts, and local degradation can easily mislead a model. Preprocessing often does not fully “solve” them; it lowers the strongest nuisance factors first.

### 4. Keep training and deployment consistent
Many pipelines fail not because of the model architecture, but because preprocessing differs between training and inference. Reliable workflows usually:

- put preprocessing parameters into scripts;
- save key metadata before and after processing;
- reuse the same rules across training, validation, and deployment.

---

## Typical case
### Case 1: CT lung nodule or organ analysis
- **Pain point**: the HU range is very wide, mixing air, soft tissue, bone, and metal.
- **Approach**: HU clipping → windowing/normalization → resampling to a common spacing.
- **Local code**:
  - `src/ch05/clip_hu_values/main.py`
  - `src/ch05/medical_image_resampling/main.py`
  - `src/ch05/detect_metal_artifacts/main.py`

### Case 2: MRI brain modeling
- **Pain point**: the same tissue may have different brightness across locations and scanners.
- **Approach**: bias correction → intensity normalization → unified size and resolution.
- **Local code**:
  - `src/ch05/n4itk_bias_correction/main.py`
  - `src/ch05/white_stripe_normalization/main.py`
  - `src/ch05/visualize_bias_field/main.py`

### Case 3: Multi-center training
- **Pain point**: the model may treat hospital or scanner identity as a label shortcut.
- **Approach**: fix spatial rules, fix intensity rules, and save preprocessing logs so training and inference stay aligned.

---

## Practice tips
The main text only keeps short snippets for intuition; the full implementations live in `src/ch05/`.

### 1. CT: clip extreme HU values back into the task range
```python
import numpy as np


def clip_hu(image, hu_min=-1000, hu_max=1000):
    image = np.asarray(image, dtype=np.float32)
    return np.clip(image, hu_min, hu_max)
```

### 2. Resampling: compute scale factors first
```python
import numpy as np


def scale_factors(original_spacing, target_spacing):
    original_spacing = np.array(original_spacing, dtype=np.float32)
    target_spacing = np.array(target_spacing, dtype=np.float32)
    return original_spacing / target_spacing
```

### 3. MRI: standardize with a white-stripe region
```python
import numpy as np


def normalize_with_white_stripe(image, wm_mask):
    wm_values = image[wm_mask > 0]
    mean = wm_values.mean()
    std = wm_values.std() + 1e-6
    return (image - mean) / std
```

### 4. In practice, record these first
- original and target spacing;
- intensity ranges or normalization parameters;
- whether masks were transformed in sync;
- whether training and deployment share the same config.

---

## Summary
In this section you learned that **preprocessing is not image beautification; it is the step that turns heterogeneous scans into inputs that are comparable and learnable.**

The next section moves to segmentation because once the input is under control, we can ask a finer question: **how can a model produce a pixel-level answer for where an organ or lesion actually is?**
