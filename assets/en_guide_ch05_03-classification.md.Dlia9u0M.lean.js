import{_ as n,I as l,c as p,o as h,ah as i,J as e,ax as k}from"./chunks/framework.CbFRlRzM.js";const b=JSON.parse('{"title":"5.3 Classification and Detection","description":"Classification and detection technologies in medical image analysis using deep learning","frontmatter":{"title":"5.3 Classification and Detection","description":"Classification and detection technologies in medical image analysis using deep learning"},"headers":[],"relativePath":"en/guide/ch05/03-classification.md","filePath":"en/guide/ch05/03-classification.md"}'),t={name:"en/guide/ch05/03-classification.md"};function r(E,s,d,g,c,y){const a=l("Mermaid");return h(),p("div",null,[s[0]||(s[0]=i("",104)),e(a,{id:"mermaid-jdeb9q7h9",code:`flowchart TD
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
    L --> P[Cross-validation]`}),s[1]||(s[1]=i("",16))])}const o=n(t,[["render",r]]);export{b as __pageData,o as default};
