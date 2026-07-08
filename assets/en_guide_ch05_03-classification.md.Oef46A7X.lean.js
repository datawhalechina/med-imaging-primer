import{V as e,Z as t,et as n,mt as r,rt as i,tt as a,vt as o}from"./chunks/framework.C5xPiHZ1.js";import"./chunks/assets.C2lZWGyA.js";var s=JSON.parse(`{"title":"5.3 Classification and Detection","description":"Understand medical image classification and detection with one consistent template: when full contours are unnecessary and how models first decide, then localize","frontmatter":{"title":"5.3 Classification and Detection","description":"Understand medical image classification and detection with one consistent template: when full contours are unnecessary and how models first decide, then localize"},"headers":[],"relativePath":"en/guide/ch05/03-classification.md","filePath":"en/guide/ch05/03-classification.md"}`),c={name:`en/guide/ch05/03-classification.md`};function l(e,s,c,l,u,d){let f=o(`Mermaid`);return r(),n(`div`,null,[s[1]||=a("",61),t(`details`,null,[s[0]||=t(`summary`,null,`📖 View Original Mermaid Code`,-1),i(f,{id:`mermaid-5yyr5ojuo`,code:`flowchart TD
    A[Medical Image Analysis Task] --> B{Data Type?}

    B -->|2D X-ray| C{Task Type?}
    B -->|3D CT/MRI| D{Task Type?}
    B -->|Whole Slide Image| E{Task Type?}

    C -->|Classification| F[ResNet/DenseNet<br>+ Transfer Learning]
    C -->|Detection| G[Faster R-CNN/YOLO<br>+ Anchor Adjustment]

    D -->|Classification| H[3D ResNet/3D DenseNet<br>+ Patch-based Training]
    D -->|Segmentation| I[3D U-Net/V-Net<br>+ Skip Connections]

    E -->|Classification| J[Attention MIL<br>+ Tissue Filtering]
    E -->|Detection| K[Patch-level Detection<br>+ Aggregation]

    F --> L[Best Practices]
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L

    L --> M[Class Imbalance Handling]
    L --> N[Data Augmentation]
    L --> O[Ensemble Methods]
    L --> P[Cross-validation]`})]),s[2]||=a("",16)])}var u=e(c,[[`render`,l]]);export{s as __pageData,u as default};