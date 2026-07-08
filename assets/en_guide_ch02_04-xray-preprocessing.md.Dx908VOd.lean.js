import{V as e,Z as t,et as n,mt as r,rt as i,tt as a,vt as o}from"./chunks/framework.C5xPiHZ1.js";var s=JSON.parse(`{"title":"2.3 X-ray: Direct Imaging Corrections","description":"Understand the preprocessing workflow for digital radiography (DR), including flat panel detector principles, core correction steps, and differences versus CT.","frontmatter":{"title":"2.3 X-ray: Direct Imaging Corrections","description":"Understand the preprocessing workflow for digital radiography (DR), including flat panel detector principles, core correction steps, and differences versus CT."},"headers":[],"relativePath":"en/guide/ch02/04-xray-preprocessing.md","filePath":"en/guide/ch02/04-xray-preprocessing.md"}`),c={name:`en/guide/ch02/04-xray-preprocessing.md`};function l(e,s,c,l,u,d){let f=o(`Mermaid`);return r(),n(`div`,null,[s[0]||=a("",22),i(f,{id:`mermaid-4o8n6qvxy`,code:`graph TD
    A[Raw Projection Data] --> B[Dark Field Correction]
    B --> C[Gain / Flat Field Correction]
    C --> D[Bad Pixel Correction]
    D --> E[Scatter Correction]
    E --> F[Lag (Ghosting) Correction]
    F --> G[Geometric Distortion Correction]
    G --> H[Corrected Projection Data]`}),s[1]||=a("",23),s[2]||=t(`ul`,null,[t(`li`,null,`Neighborhood mean: I_corrected,i = (1/N_neighbors) Σ_{j∈neighbors} I_corrected,j (4/8-neighborhood)`),t(`li`,null,`Weighted interpolation: I_corrected,i = (Σ_j w_j I_corrected,j) / (Σ_j w_j), with w_j = 1/d_ij^2`),t(`li`,{"i+1,j+1":``},`Bilinear interpolation (on regular grids): I_corrected,i,j = (1−α)(1−β) I_{i,j} + α(1−β) I_{i+1,j} + (1−α)β I_{i,j+1} + αβ I_`)],-1),s[3]||=a("",65),i(f,{id:`mermaid-09gb274pa`,code:`graph LR
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
    J -->|No| L[Re-acquisition]`}),s[4]||=a("",9)])}var u=e(c,[[`render`,l]]);export{s as __pageData,u as default};