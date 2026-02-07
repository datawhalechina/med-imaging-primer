import{_ as o,I as n,o as s,c as l,ak as i,J as t,j as r}from"./chunks/framework.CGzjHEBf.js";const y=JSON.parse('{"title":"2.3 X-ray: Direct Imaging Corrections","description":"Understand the preprocessing workflow for digital radiography (DR), including flat panel detector principles, core correction steps, and differences versus CT.","frontmatter":{"title":"2.3 X-ray: Direct Imaging Corrections","description":"Understand the preprocessing workflow for digital radiography (DR), including flat panel detector principles, core correction steps, and differences versus CT."},"headers":[],"relativePath":"en/guide/ch02/04-xray-preprocessing.md","filePath":"en/guide/ch02/04-xray-preprocessing.md"}'),c={name:"en/guide/ch02/04-xray-preprocessing.md"};function d(p,e,m,u,h,g){const a=n("Mermaid");return s(),l("div",null,[e[0]||(e[0]=i("",22)),t(a,{id:"mermaid-7xvgcdyxh",code:`graph TD
    A[Raw Projection Data] --> B[Dark Field Correction]
    B --> C[Gain / Flat Field Correction]
    C --> D[Bad Pixel Correction]
    D --> E[Scatter Correction]
    E --> F[Lag (Ghosting) Correction]
    F --> G[Geometric Distortion Correction]
    G --> H[Corrected Projection Data]`}),e[1]||(e[1]=i("",23)),e[2]||(e[2]=r("ul",null,[r("li",null,"Neighborhood mean: I_corrected,i = (1/N_neighbors) Σ_{j∈neighbors} I_corrected,j (4/8-neighborhood)"),r("li",null,"Weighted interpolation: I_corrected,i = (Σ_j w_j I_corrected,j) / (Σ_j w_j), with w_j = 1/d_ij^2"),r("li",{"i+1,j+1":""},"Bilinear interpolation (on regular grids): I_corrected,i,j = (1−α)(1−β) I_{i,j} + α(1−β) I_{i+1,j} + (1−α)β I_{i,j+1} + αβ I_")],-1)),e[3]||(e[3]=i("",65)),t(a,{id:"mermaid-9xoexcupj",code:`graph LR
    A[Raw Projection] --> B[Dark]
    B --> C[Gain]
    C --> D[Bad Pixel]
    D --> E[Scatter]
    E --> F[Lag]
    F --> G[Geometric]
    G --> H[Corrected Projection]
    H --> I[Quality Assessment]
    I --> J{Pass?}
    J -->|Yes| K[To Reconstruction]
    J -->|No| L[Re-acquisition]`}),e[4]||(e[4]=i("",9))])}const b=o(c,[["render",d]]);export{y as __pageData,b as default};
