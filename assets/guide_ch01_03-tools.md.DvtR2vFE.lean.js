import{_ as l,I as t,o as p,c as e,ak as i,J as n}from"./chunks/framework.CGzjHEBf.js";const b=JSON.parse('{"title":"1.3 常用开源工具","description":"了解医学影像处理的开源工具生态，掌握工具选择与使用的最佳实践","frontmatter":{"title":"1.3 常用开源工具","description":"了解医学影像处理的开源工具生态，掌握工具选择与使用的最佳实践"},"headers":[],"relativePath":"guide/ch01/03-tools.md","filePath":"zh/guide/ch01/03-tools.md"}'),r={name:"guide/ch01/03-tools.md"};function h(k,s,d,g,o,E){const a=t("Mermaid");return p(),e("div",null,[s[0]||(s[0]=i("",10)),n(a,{id:"mermaid-c2r78v0m7",code:`graph LR
    A[数据获取] --> B[格式转换]
    B --> C[可视化检查]
    C --> D[预处理]
    D --> E[分析/重建]
    E --> F[结果输出]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1
    style E fill:#f5e1ff
    style F fill:#ffe1e1`}),s[1]||(s[1]=i("",66)),n(a,{id:"mermaid-qxgxw0392",code:`graph TD
    A[需要处理医学影像?] --> B{数据格式?}
    B -->|DICOM| C{需要什么?}
    C -->|只读取像素数据| D[pydicom<br/>简单快速]
    C -->|修改DICOM Tag| D
    C -->|复杂处理| E[SimpleITK<br/>功能强大]
    
    B -->|NIfTI| F{需要什么?}
    F -->|只读写数据| G[nibabel<br/>轻量高效]
    F -->|图像处理操作| E
    
    B -->|多种格式| E
    
    style D fill:#e1f5ff
    style G fill:#ffe1f5
    style E fill:#e1ffe1`}),s[2]||(s[2]=i("",75)),n(a,{id:"mermaid-2uylf0wgi",code:`graph LR
    A[DICOM数据] -->|dcm2niix| B[NIfTI格式]
    B -->|SimpleITK| C[预处理]
    C -->|NumPy/SciPy| D[算法实现]
    D -->|PyTorch/TF| E[深度学习]
    E -->|nibabel| F[结果保存]

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1
    style E fill:#f5e1ff
    style F fill:#ffe1e1`}),s[3]||(s[3]=i("",35)),n(a,{id:"mermaid-gauyvqxs3",code:`graph TD
    A[第1周：基础工具] --> B[第2周：Python库]
    B --> C[第3周：可视化工具]
    C --> D[第4周：完整流程]

    A --> A1[dcm2niix基本用法]
    A --> A2[命令行基础]

    B --> B1[pydicom读取DICOM]
    B --> B2[nibabel读写NIfTI]
    B --> B3[SimpleITK基本操作]

    C --> C1[3D Slicer基本功能]
    C --> C2[ITK-SNAP快速分割]

    D --> D1[构建处理流程]
    D --> D2[调试和优化]

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1`}),s[4]||(s[4]=i("",5))])}const y=l(r,[["render",h]]);export{b as __pageData,y as default};
