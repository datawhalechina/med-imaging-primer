---
description: From Röntgen's serendipitous discovery to modern multi-modal imaging, exploring the evolution of medical imaging technology
outline: deep
---

# Medical Imaging Processing Tutorial Guide

> "I have seen my death!"  
> — Anna Bertha Röntgen, upon seeing the X-ray of her hand, December 22, 1895

On the evening of November 8, 1895, German physicist Wilhelm Röntgen was studying cathode rays in his dimly lit laboratory at the University of Würzburg. When he completely wrapped a Crookes tube in black cardboard and applied electricity, he unexpectedly noticed a fluorescent screen coated with barium platinocyanide glowing several feet away. Even more astonishing, when he placed his hand in the path of the rays, the screen showed an image of his hand bones—this was the first time in human history that the inside of the body could be "seen" without surgery.

Röntgen named these unknown rays "X-rays." On December 22, 1895, he took the first medical image in human history—an X-ray of his wife Anna Bertha's hand. When Anna saw the X-ray showing her hand bones and wedding ring, she exclaimed in shock: "I have seen my death!" This photograph marked the beginning of a new era in medical diagnosis.

![Anna Bertha's Hand X-ray](/images/Anna-hand.jpg)

::: tip 🏆 Nobel Prize Glory
Röntgen received the first Nobel Prize in Physics in 1901 for his discovery of X-rays. He refused to patent this discovery, believing it should belong to all humanity. This decision allowed X-ray technology to spread rapidly, saving countless lives.
:::

## 📜 The Evolution of Medical Imaging

From Röntgen's accidental discovery to today's multi-modal imaging, medical imaging technology has undergone more than a century of development. This is not only technological progress, but also humanity's relentless pursuit of the dream to "see the invisible."

### 🔬 Phase 1: Projection Imaging Era (1895-1970s)

**The Golden Age of X-rays**

After Röntgen's discovery of X-rays, the technology spread at an astonishing rate. In 1896, just months after the discovery, X-ray machines were used on battlefields to locate bullets and shrapnel. However, early X-ray imaging faced a fundamental limitation: it was a **projection imaging** technique.

Imagine shining a flashlight on a complex three-dimensional object—the shadow on the wall compresses all depth information into a two-dimensional image. X-ray imaging works the same way—information from all tissues in the body is superimposed on a flat photograph. A lung tumor might be obscured by ribs, and a small lesion might be lost in the overlapping images of surrounding tissues.

::: warning ⚠️ The Dilemma of Projection Imaging
Traditional X-ray photographs are like taking a photo of a thick book after flattening it—you can see all the text, but it's all overlapped, making it difficult to read each page's content.
:::

### 🎯 Phase 2: Tomographic Imaging Revolution (1970s-1990s)

**Hounsfield's Brilliant Idea**

In 1967, British engineer Godfrey Hounsfield at EMI began pondering a question: Could we reconstruct the three-dimensional structure of an object by taking X-ray photographs from multiple angles and using a computer? This seemingly simple idea required solving two enormous challenges:

1. **Mathematical Problem**: How to reconstruct the original three-dimensional image from multiple projections?
2. **Computational Challenge**: In the 1960s, computer processing power was extremely limited

Hounsfield spent five years combining the mathematical theory proposed by Austrian mathematician Johann Radon in 1917, and finally developed the world's first clinical CT scanner in 1971. The first scan took 5 minutes to acquire data, then 2.5 hours for image reconstruction—but the results were stunning: doctors could clearly see the internal structure of the brain for the first time without craniotomy.

::: tip 🎖️ World-Changing Invention
Hounsfield and South African physicist Allan Cormack, who independently developed CT theory, jointly received the 1979 Nobel Prize in Physiology or Medicine. CT technology has been hailed as "the most important advance in radiology since Röntgen's discovery of X-rays."
:::

**MRI: The Radiation-Free Revolution**

While CT technology was flourishing, another revolutionary technology was brewing. In 1973, American chemist Paul Lauterbur and British physicist Peter Mansfield almost simultaneously but independently proposed methods for imaging using nuclear magnetic resonance (NMR).

Unlike X-rays and CT, MRI does not use ionizing radiation but utilizes the resonance phenomenon of hydrogen nuclei in the human body in a strong magnetic field. It's like installing a tiny "radio transmitter" on every hydrogen atom in the body. By precisely adjusting the magnetic field, we can make hydrogen atoms at specific locations "speak," thereby constructing images.

::: info 💡 Why "Magnetic Resonance" Instead of "Nuclear Magnetic Resonance"?
This technology was originally called "Nuclear Magnetic Resonance Imaging," but the word "nuclear" made the public associate it with nuclear radiation, causing unnecessary panic. Therefore, in clinical applications, it was renamed "Magnetic Resonance Imaging (MRI)," although it has nothing to do with nuclear radiation.
:::

MRI's advantage lies not only in being radiation-free but also in providing **soft tissue contrast** far superior to CT. Gray and white matter in the brain, different layers of muscle, early tumor lesions—structures difficult to distinguish on CT are clearly visible on MRI. Lauterbur and Mansfield received the 2003 Nobel Prize in Physiology or Medicine for this achievement.

### 🌈 Phase 3: Functional and Molecular Imaging Era (1990s-Present)

**From "Seeing Structure" to "Seeing Function"**

Entering the 21st century, medical imaging technology began to evolve from purely anatomical structure imaging to functional and molecular imaging:

- **PET (Positron Emission Tomography)**: No longer satisfied with seeing the shape of tumors, but wanting to see their metabolic activity. Cancer cells, due to rapid growth, consume large amounts of glucose. PET uses radioactively labeled glucose (FDG) to "light up" tumors.

- **Functional MRI (fMRI)**: By detecting changes in blood oxygen levels, we can even see which areas of the brain are activated during thinking—this was science fiction just 20 years ago.

- **Ultrasound Imaging**: From initial obstetric examinations to real-time surgical guidance, hemodynamic assessment, and even treatment (high-intensity focused ultrasound).

::: details 📊 Medical Imaging Technology Timeline
- **1895**: Röntgen discovers X-rays
- **1917**: Radon proposes mathematical theory for tomographic reconstruction
- **1971**: Hounsfield invents the first CT scanner
- **1973**: Lauterbur and Mansfield propose MRI principles
- **1977**: First whole-body MRI scanner developed
- **1990s**: PET/CT fusion imaging technology matures
- **2000s**: Functional MRI widely applied in neuroscience research
- **2010s**: Artificial intelligence begins revolutionizing medical image analysis
:::

## 🗺️ Tutorial Content Overview

From Röntgen's accidental discovery in a dimly lit laboratory, to Hounsfield reconstructing the first CT image with a computer, to today's AI-assisted precision diagnosis—every advance in medical imaging technology stems from humanity's persistent pursuit of "seeing the invisible."

This tutorial adopts a progressive learning path of "Principles → Practice → Application," starting from physical imaging principles, through image reconstruction algorithms, and finally reaching deep learning applications. The entire tutorial is divided into five chapters, each building on the previous one to form a complete knowledge system.

### 📘 Chapter 1: Medical Imaging Basics

**Learning Objective**: Understand the physical principles and clinical applications of mainstream medical imaging modalities

Chapter 1 systematically introduces the physical principles of mainstream imaging modalities such as CT, MRI, X-ray, and PET/US, explains medical image data format standards like DICOM and NIfTI, introduces common open-source tools such as ITK, PyDICOM, and MONAI, and provides in-depth analysis of typical artifacts in each modality and their causes. This is the foundation for entering the field of medical image processing.

### 📗 Chapter 2: Pre-Reconstruction Processing—Modality-Specific Correction Workflows

**Learning Objective**: Master raw data correction techniques for different imaging modalities

Chapter 2 focuses on data preprocessing before image reconstruction, explaining the complete workflow from raw detector signals to corrected data suitable for reconstruction for CT, MRI, and X-ray modalities. This includes key techniques such as beam hardening correction for CT, k-space denoising for MRI, and flat-field correction for X-ray.

### 📙 Chapter 3: Image Reconstruction Algorithms (Organized by Modality)

**Learning Objective**: Deeply understand the reconstruction process from raw data to final images

Chapter 3 is the core of this tutorial, explaining the mathematical principles and algorithm implementations of image reconstruction by modality. From CT's filtered back-projection (FBP) and iterative reconstruction, to MRI's Fourier transform and compressed sensing reconstruction, to X-ray's digital tomosynthesis, systematically master the core technologies of medical image reconstruction.

### 📕 Chapter 4: Reconstruction Practice and Validation (Multi-Modal Examples)

**Learning Objective**: Master the complete workflow of image reconstruction through practical cases

Chapter 4 provides real medical image reconstruction cases, covering the complete workflow from raw data reading, preprocessing, reconstruction algorithm implementation, to final image quality assessment. Using open-source tools like ASTRA, BART, and SigPy, you'll process CT, MRI, and X-ray data hands-on and learn to troubleshoot common issues.

### 📓 Chapter 5: Medical Image Post-Processing (General + Modality Adaptation)

**Learning Objective**: Explore the application of deep learning in medical image processing

Chapter 5 is organized around four mainline questions in medical image post-processing: how to prepare data, why segmentation works, how to think about classification and detection, and when enhancement or restoration should be used. Runnable code, training scripts, demos, and complete outputs are collected in `src/ch05/` and the chapter practice appendix, while SAM and generative AI are positioned as optional frontier reading.

::: info 🎯 Learning Path Recommendations
- **Beginners**: Study Chapters 1-3 in sequence to build a solid theoretical foundation
- **Experienced Learners**: Can start directly from Chapter 3, focusing on reconstruction algorithms
- **AI Direction**: After mastering Chapters 1-2, can jump to Chapter 5 to learn deep learning applications
- **Practice-Oriented**: Cases in Chapter 4 can be studied in conjunction with previous chapters to deepen understanding
:::

::: info 📚 Further Reading
- **"Medical Imaging Physics"** (Hendee & Ritenour): Systematic introduction to the physical foundations of various imaging modalities
- **"The Essential Physics of Medical Imaging"** (Bushberg et al.): In-depth discussion of the mathematics and physics of tomographic imaging
- **"Computed Tomography: Principles, Design, Artifacts, and Recent Advances"** (Xie Qiang): Masterpiece in the CT field
- **Nobel Prize Official Website**: Read the award speeches of Röntgen, Hounsfield, Lauterbur, and others to learn the stories behind these discoveries
:::

---

**Ready? Let's start with Chapter 1 and explore the physical foundations of medical imaging!**

