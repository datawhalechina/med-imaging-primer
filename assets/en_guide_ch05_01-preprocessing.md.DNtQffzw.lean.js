import{_ as e,I as t,o as l,c as r,ak as i,j as a,J as p,au as h,av as o,aw as d}from"./chunks/framework.BsiU-GuW.js";const f=JSON.parse('{"title":"5.1 Preprocessing (with modality differences in mind)","description":"Understand medical image preprocessing with one consistent template: why it exists, what intuition matters, and how it supports downstream tasks","frontmatter":{"title":"5.1 Preprocessing (with modality differences in mind)","description":"Understand medical image preprocessing with one consistent template: why it exists, what intuition matters, and how it supports downstream tasks"},"headers":[],"relativePath":"en/guide/ch05/01-preprocessing.md","filePath":"en/guide/ch05/01-preprocessing.md"}'),k={name:"en/guide/ch05/01-preprocessing.md"};function c(g,s,m,E,u,y){const n=t("Mermaid");return l(),r("div",null,[s[1]||(s[1]=i("",62)),a("details",null,[s[0]||(s[0]=a("summary",null,"📖 View Original Mermaid Code",-1)),p(n,{id:"mermaid-ur908bw9n",code:`flowchart TD
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
    O --> P[Final Normalization]`})]),s[2]||(s[2]=i("",41))])}const F=e(k,[["render",c]]);export{f as __pageData,F as default};
