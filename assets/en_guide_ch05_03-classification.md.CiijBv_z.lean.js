import{_ as l,I as e,o as p,c as t,ak as i,j as a,J as h,aG as k,aH as r,aI as E}from"./chunks/framework.CGzjHEBf.js";const m=JSON.parse('{"title":"5.3 Classification and Detection","description":"Classification and detection technologies in medical image analysis using deep learning","frontmatter":{"title":"5.3 Classification and Detection","description":"Classification and detection technologies in medical image analysis using deep learning"},"headers":[],"relativePath":"en/guide/ch05/03-classification.md","filePath":"en/guide/ch05/03-classification.md"}'),d={name:"en/guide/ch05/03-classification.md"};function g(c,s,y,o,F,b){const n=e("Mermaid");return p(),t("div",null,[s[1]||(s[1]=i("",136)),a("details",null,[s[0]||(s[0]=a("summary",null,"📖 View Original Mermaid Code",-1)),h(n,{id:"mermaid-nzdwqkilu",code:`flowchart TD
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
    L --> P[Cross-validation]`})]),s[2]||(s[2]=i("",20))])}const C=l(d,[["render",g]]);export{m as __pageData,C as default};
