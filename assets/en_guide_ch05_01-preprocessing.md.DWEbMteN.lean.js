import{V as e,Z as t,et as n,mt as r,rt as i,tt as a,vt as o}from"./chunks/framework.C5xPiHZ1.js";import"./chunks/assets.C2lZWGyA.js";var s=JSON.parse(`{"title":"5.1 Preprocessing (with modality differences in mind)","description":"Understand medical image preprocessing with one consistent template: why it exists, what intuition matters, and how it supports downstream tasks","frontmatter":{"title":"5.1 Preprocessing (with modality differences in mind)","description":"Understand medical image preprocessing with one consistent template: why it exists, what intuition matters, and how it supports downstream tasks"},"headers":[],"relativePath":"en/guide/ch05/01-preprocessing.md","filePath":"en/guide/ch05/01-preprocessing.md"}`),c={name:`en/guide/ch05/01-preprocessing.md`};function l(e,s,c,l,u,d){let f=o(`Mermaid`);return r(),n(`div`,null,[s[1]||=a("",62),t(`details`,null,[s[0]||=t(`summary`,null,`📖 View Original Mermaid Code`,-1),i(f,{id:`mermaid-bfmhbqgol`,code:`flowchart TD
    A[Medical Image Preprocessing Task] --> B{Determine Imaging Modality}

    B -->|CT| C[HU Value Calibration]
    C --> D[Window Level Adjustment]
    D --> E[Outlier Processing]

    B -->|MRI| F[Bias Field Correction]
    F --> G[Intensity Standardization]
    G --> H[Multi-sequence Fusion]

    B -->|X-ray| I[Contrast Enhancement]
    I --> J[Anatomical Region Segmentation]
    J --> K[Local Normalization]

    E --> L[Universal Preprocessing]
    H --> L
    K --> L

    L --> M[Resampling]
    M --> N[Size Standardization]
    N --> O[Data Augmentation]
    O --> P[Final Normalization]`})]),s[2]||=a("",41)])}var u=e(c,[[`render`,l]]);export{s as __pageData,u as default};