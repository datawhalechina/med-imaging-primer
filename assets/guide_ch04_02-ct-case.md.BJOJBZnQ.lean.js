import{_ as e,I as p,o as t,c as l,ak as n,J as k,j as s,a,bA as r,bB as d,bC as g,bD as E,bE as o,bF as y,bG as c,bH as b,bI as m,bJ as u,bK as F,bL as f,bM as C,bN as _}from"./chunks/framework.CGzjHEBf.js";const L=JSON.parse('{"title":"4.2 案例一：CT 正弦图回放与重建","description":"从 Phantom 到 sinogram，再到 FBP 重建与可视化","frontmatter":{"title":"4.2 案例一：CT 正弦图回放与重建","description":"从 Phantom 到 sinogram，再到 FBP 重建与可视化"},"headers":[],"relativePath":"guide/ch04/02-ct-case.md","filePath":"zh/guide/ch04/02-ct-case.md"}'),A={name:"guide/ch04/02-ct-case.md"},Q={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},D={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.375ex"},xmlns:"http://www.w3.org/2000/svg",width:"12.756ex",height:"2.129ex",role:"img",focusable:"false",viewBox:"0 -775.2 5638.4 940.8","aria-hidden":"true"};function T(B,i,x,v,P,w){const h=p("Mermaid");return t(),l("div",null,[i[8]||(i[8]=n("",22)),k(h,{id:"mermaid-jjezrm5in",code:`graph TD
    A["生成 Phantom"] --> B["模拟数据采集<br/>(计数域)"]
    B --> C["预处理步骤"]
    C --> D["暗电流校正"]
    D --> E["增益校正"]
    E --> F["空气校正"]
    F --> G["投影值计算"]
    G --> H["射束硬化校正"]
    H --> I["散射校正"]
    I --> J["环形伪影校正"]
    J --> K["FBP 重建"]
    K --> L["滤波器选择"]
    L --> M["反投影"]
    M --> N["后处理"]
    N --> O["噪声抑制"]
    O --> P["边缘增强"]
    P --> Q["质量评估"]
    
    style A fill:#3a3a3a,stroke:#666,color:#e0e0e0
    style B fill:#3a3a3a,stroke:#666,color:#e0e0e0
    style C fill:#4a6a4a,stroke:#666,color:#e0e0e0
    style D fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style E fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style F fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style G fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style H fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style I fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style J fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style K fill:#4a6a4a,stroke:#666,color:#e0e0e0
    style L fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style M fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style N fill:#4a6a4a,stroke:#666,color:#e0e0e0
    style O fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style P fill:#2a2a2a,stroke:#555,color:#e0e0e0
    style Q fill:#3a3a3a,stroke:#666,color:#e0e0e0`}),i[9]||(i[9]=s("h3",{id:"_4-2-数据采集模拟-计数域",tabindex:"-1"},[a("4.2 数据采集模拟（计数域） "),s("a",{class:"header-anchor",href:"#_4-2-数据采集模拟-计数域","aria-label":"Permalink to “4.2 数据采集模拟（计数域）”"},"​")],-1)),i[10]||(i[10]=s("p",null,"CT 数据采集从光子计数开始。X 射线穿过物体后，探测器记录的光子数服从 Poisson 分布。我们通过以下步骤模拟真实采集过程：",-1)),s("ol",null,[i[5]||(i[5]=s("li",null,[s("strong",null,"理想投影"),a("：通过 Radon 变换获取理想正弦图")],-1)),s("li",null,[i[2]||(i[2]=s("strong",null,"光子计数转换",-1)),i[3]||(i[3]=a("：根据 Beer-Lambert 定律 ",-1)),s("mjx-container",Q,[(t(),l("svg",D,[...i[0]||(i[0]=[n("",1)])])),i[1]||(i[1]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"N"),s("mo",null,"="),s("msub",null,[s("mi",null,"N"),s("mn",null,"0")]),s("mo",null,"⋅"),s("msup",null,[s("mi",null,"e"),s("mrow",{"data-mjx-texclass":"ORD"},[s("mo",null,"−"),s("mi",null,"p")])])])],-1))]),i[4]||(i[4]=a(" 计算光子数",-1))]),i[6]||(i[6]=s("li",null,[s("strong",null,"噪声添加"),a("：添加 Poisson 噪声模拟量子噪声")],-1)),i[7]||(i[7]=s("li",null,[s("strong",null,"系统误差"),a("：添加暗电流和增益不均匀性")],-1))]),i[11]||(i[11]=n("",64))])}const M=e(A,[["render",T]]);export{L as __pageData,M as default};
