---
title: "5.6 Code Labs / Practice Appendix"
description: The dedicated appendix for Chapter 5 runnable scripts, environment requirements, demo entry points, and where to find full outputs.
---

# 5.6 Code Labs / Practice Appendix

This page is the **practice appendix** for Chapter 5. Its job is different from the four mainline pages:

- **mainline pages (5.1–5.4)** explain concepts and decision logic;
- **this appendix** collects runnable entry points, dependencies, outputs, and implementation pointers;
- **5.5** remains optional frontier reading.

## Where is the full implementation?
All complete scripts, training logic, demos, and generated outputs are under `src/ch05/`.

### Script groups by topic

| Mainline question | Local code area | Representative scripts |
| --- | --- | --- |
| How should data be prepared? | preprocessing | `clip_hu_values/`, `medical_image_resampling/`, `n4itk_bias_correction/`, `white_stripe_normalization/`, `detect_metal_artifacts/`, `visualize_bias_field/` |
| Why does segmentation work? | segmentation | `lung_segmentation_network/`, `medical_segmentation_augmentation/` |
| How should we think about classification and detection? | classification | `medical_image_classification/` |
| When should enhancement or restoration be used? | augmentation / restoration | `medical_image_augmentation/`, `clahe_enhancement/`, plus the MRI bias-field tools |

---

## Recommended reading order inside `src/ch05/`
1. Start with `src/ch05/README_EN.md` for the chapter-wide experiment index.
2. Then open the subfolder README for the script you want to run.
3. Use the local `output/` directory in each experiment to inspect generated images and reports.

---

## Typical run pattern
Most Chapter 5 demos follow the same structure:

```bash
cd src/ch05/<experiment_name>
python main.py
```

Some experiments also provide a simplified entry or an extra test file:

```bash
python simple_augmentation.py
python test.py
```

---

## Environment and dependencies
For Chapter 5 practice, dependencies are documented in:

- `src/ch05/requirements.txt`
- `src/ch05/README_EN.md`
- individual experiment READMEs when extra packages are needed

Typical packages include:

- `numpy`, `matplotlib`, `scipy`, `scikit-image`
- `opencv-python`
- `torch`, `torchvision`
- `pydicom`, `nibabel`, `SimpleITK`

---

## Where are full outputs stored?
Each experiment keeps its own generated artifacts, usually in one of these folders:

- `output/`
- `outputs/`

Examples:

- `src/ch05/lung_segmentation_network/output/`
- `src/ch05/medical_image_classification/output/`
- `src/ch05/medical_image_augmentation/output/`
- `src/ch05/clahe_enhancement/output/`

These folders hold figures, reports, and demo visualizations that would be too heavy for the main tutorial pages.

---

## How should the appendix be used?
Use this appendix when you want to:

- run the chapter code locally;
- inspect full implementation details;
- compare generated outputs;
- understand environment setup and script entry points.

Return to the mainline pages when you want to answer the conceptual questions of Chapter 5.

::: tip One-sentence summary
The mainline pages explain **why**; this appendix and `src/ch05/` show **how to run and inspect the full workflow**.
:::
