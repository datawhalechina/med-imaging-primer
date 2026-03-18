import{_ as s,I as n,o as r,c as l,ak as a,j as i,J as o,aA as d,aB as c}from"./chunks/framework.BsiU-GuW.js";const y=JSON.parse('{"title":"5.3 Classification and Detection","description":"Understand medical image classification and detection with one consistent template: when full contours are unnecessary and how models first decide, then localize","frontmatter":{"title":"5.3 Classification and Detection","description":"Understand medical image classification and detection with one consistent template: when full contours are unnecessary and how models first decide, then localize"},"headers":[],"relativePath":"en/guide/ch05/03-classification.md","filePath":"en/guide/ch05/03-classification.md"}'),h={name:"en/guide/ch05/03-classification.md"};function p(g,e,m,u,b,f){const t=n("Mermaid");return r(),l("div",null,[e[1]||(e[1]=a("",61)),i("details",null,[e[0]||(e[0]=i("summary",null,"📖 View Original Mermaid Code",-1)),o(t,{id:"mermaid-lwqutgi1k",code:`flowchart TD
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
    L --> P[Cross-validation]`})]),e[2]||(e[2]=a("",16))])}const C=s(h,[["render",p]]);export{y as __pageData,C as default};
