import{_ as l,I as p,c as h,o as k,aj as i,j as a,J as t,aP as e,aQ as r}from"./chunks/framework.ChhwQ8Kw.js";const o=JSON.parse('{"title":"5.3 分类和检测","description":"医学图像分析中的深度学习分类和检测技术","frontmatter":{"title":"5.3 分类和检测","description":"医学图像分析中的深度学习分类和检测技术"},"headers":[],"relativePath":"guide/ch05/03-classification.md","filePath":"zh/guide/ch05/03-classification.md"}'),E={name:"guide/ch05/03-classification.md"};function d(g,s,y,c,F,b){const n=p("Mermaid");return k(),h("div",null,[s[1]||(s[1]=i("",121)),a("details",null,[s[0]||(s[0]=a("summary",null,"📖 查看原始Mermaid代码",-1)),t(n,{id:"mermaid-52wbzs14y",code:`flowchart TD
    A[医学图像分析任务] --> B{数据类型？}

    B -->|2D X线| C{任务类型？}
    B -->|3D CT/MRI| D{任务类型？}
    B -->|全幻灯片图像| E{任务类型？}

    C -->|分类| F[ResNet/DenseNet<br>+ 迁移学习]
    C -->|检测| G[Faster R-CNN/YOLO<br>+ Anchor调整]

    D -->|分类| H[3D ResNet/3D DenseNet<br>+ 基于Patch的训练]
    D -->|分割| I[3D U-Net/V-Net<br>+ 跳跃连接]

    E -->|分类| J[注意力MIL<br>+ 组织过滤]
    E -->|检测| K[Patch级检测<br>+ 聚合]

    F --> L[最佳实践]
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L

    L --> M[类别不平衡处理]
    L --> N[数据增强]
    L --> O[集成方法]
    L --> P[交叉验证]`})]),s[2]||(s[2]=i("",16))])}const C=l(E,[["render",d]]);export{o as __pageData,C as default};
