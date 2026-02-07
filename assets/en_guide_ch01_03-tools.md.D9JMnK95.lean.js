import{_ as e,I as t,o as l,c as r,ak as i,J as n}from"./chunks/framework.CGzjHEBf.js";const u=JSON.parse('{"title":"1.3 Common Open Source Tools","description":"Understanding the open source tool ecosystem for medical imaging processing and mastering best practices for tool selection and usage","frontmatter":{"title":"1.3 Common Open Source Tools","description":"Understanding the open source tool ecosystem for medical imaging processing and mastering best practices for tool selection and usage"},"headers":[],"relativePath":"en/guide/ch01/03-tools.md","filePath":"en/guide/ch01/03-tools.md"}'),p={name:"en/guide/ch01/03-tools.md"};function o(h,s,k,d,g,c){const a=t("Mermaid");return l(),r("div",null,[s[0]||(s[0]=i("",10)),n(a,{id:"mermaid-9yy4j2e6h",code:`graph LR
    A[Data Acquisition] --> B[Format Conversion]
    B --> C[Visualization Check]
    C --> D[Preprocessing]
    D --> E[Analysis/Reconstruction]
    E --> F[Result Output]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1
    style E fill:#f5e1ff
    style F fill:#ffe1e1`}),s[1]||(s[1]=i("",66)),n(a,{id:"mermaid-orkhw4xjo",code:`graph TD
    A[Need to process medical images?] --> B{Data format?}
    B -->|DICOM| C{What do you need?}
    C -->|Just read pixel data| D[pydicom<br/>Simple and fast]
    C -->|Modify DICOM Tags| D
    C -->|Complex processing| E[SimpleITK<br/>Powerful]
    
    B -->|NIfTI| F{What do you need?}
    F -->|Just read/write data| G[nibabel<br/>Lightweight and efficient]
    F -->|Image processing operations| E
    
    B -->|Multiple formats| E
    
    style D fill:#e1f5ff
    style G fill:#ffe1f5
    style E fill:#e1ffe1`}),s[2]||(s[2]=i("",74)),n(a,{id:"mermaid-ihcib66bb",code:`graph LR
    A[DICOM Data] -->|dcm2niix| B[NIfTI Format]
    B -->|SimpleITK| C[Preprocessing]
    C -->|NumPy/SciPy| D[Algorithm Implementation]
    D -->|PyTorch/TF| E[Deep Learning]
    E -->|nibabel| F[Save Results]

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1
    style E fill:#f5e1ff
    style F fill:#ffe1e1`}),s[3]||(s[3]=i("",35)),n(a,{id:"mermaid-s72k4xo30",code:`graph TD
    A[Week 1: Basic Tools] --> B[Week 2: Python Libraries]
    B --> C[Week 3: Visualization Tools]
    C --> D[Week 4: Complete Pipeline]

    A --> A1[dcm2niix basic usage]
    A --> A2[Command-line basics]

    B --> B1[pydicom read DICOM]
    B --> B2[nibabel read/write NIfTI]
    B --> B3[SimpleITK basic operations]

    C --> C1[3D Slicer basic functions]
    C --> C2[ITK-SNAP quick segmentation]

    D --> D1[Build processing pipeline]
    D --> D2[Debug and optimize]

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1`}),s[4]||(s[4]=i("",5))])}const y=e(p,[["render",o]]);export{u as __pageData,y as default};
