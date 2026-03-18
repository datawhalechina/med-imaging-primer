---
title: 5.1 预处理（强调模态差异）
description: 以“为什么需要 → 核心直觉 → 典型场景 → 最小流程 → 片段代码 → 延伸实践”的顺序理解医学影像预处理
---
# 5.1 预处理（强调模态差异）

## 这一节解决什么问题
这一节回答的是：**原始医学影像为什么不能直接丢进模型，以及不同模态为什么要走不同的预处理路线。**

## 为什么前面几章还不够
前几章帮我们理解了成像原理、重建过程和图像质量，但“看懂图像”不等于“能稳定训练模型”。进入深度学习阶段后，还会遇到三个新问题：

1. **空间不统一**：不同设备、不同协议的体素间距和方向不一致。
2. **强度不统一**：CT 有明确的 HU，MRI 强度却依赖序列与设备，X 光又更依赖局部对比度。
3. **任务不统一**：分割、分类、检测对输入分布的容忍度不同。

所以，预处理不是“把图像弄干净”这么简单，而是在**物理意义、临床可读性和模型可学性**之间做平衡。

---

## 为什么需要
医学影像预处理最常见的目标有三类：

| 目标 | 要解决的问题 | 常见做法 |
| --- | --- | --- |
| 统一空间 | 切片厚度、像素间距、方向不同 | 重采样、方向标准化、裁剪/补零 |
| 统一强度 | 不同模态的强度范围差异大 | CT 截断与窗化、MRI 标准化、X 光对比度增强 |
| 去除干扰 | 噪声、偏场、金属伪影会误导模型 | N4 偏场校正、伪影检测、局部增强 |

如果不做这些步骤，模型往往学到的是“扫描协议差异”而不是“疾病特征”。

---

## 核心直觉
可以把预处理理解成一句话：**先把数据变得可比，再把信息变得可学。**

- **CT**：重点是保留 HU 的物理意义，同时把极端动态范围压缩到任务相关区间。
- **MRI**：重点是减少设备、线圈和序列带来的强度漂移，让不同样本的“同类组织”更可比。
- **多模态/多中心数据**：重点是先统一空间，再讨论强度。

![医学影像预处理层次结构](/images/ch05/01-preprocessing-hierarchy-zh.png)
*图：预处理通常从空间统一开始，再进入模态特异性的强度处理，最后才连接到任务导向的数据准备。*

---

## 典型场景

### 场景 1：CT 肺结节或器官分析
- **为什么需要**：原始 HU 范围太大，空气、软组织、骨和金属会同时出现。
- **典型操作**：HU 截断 → 窗化/归一化 → 重采样到统一体素间距。
- **本地源码**：
  - `src/ch05/clip_hu_values/main.py`
  - `src/ch05/medical_image_resampling/main.py`
  - `src/ch05/detect_metal_artifacts/main.py`

### 场景 2：MRI 脑影像建模
- **为什么需要**：MRI 没有跨设备统一的绝对强度，偏场会让同一组织在图像不同位置亮度不同。
- **典型操作**：偏场校正 → 强度标准化 → 统一尺寸/分辨率。
- **本地源码**：
  - `src/ch05/n4itk_bias_correction/main.py`
  - `src/ch05/white_stripe_normalization/main.py`
  - `src/ch05/visualize_bias_field/main.py`

### 场景 3：多中心数据汇总训练
- **为什么需要**：模型容易把医院、设备、扫描协议当成“标签线索”。
- **典型操作**：固定 spacing、固定强度规则、保留处理元数据，保证训练/推理一致。
- **建议**：把预处理配置写成脚本参数，不要只保留在实验笔记里。

---

## 最小流程
下面给出一个足够通用的“最小预处理流程”，可作为阅读后续章节的共同入口。

1. **读入影像并确认元数据**：shape、spacing、方向、模态类型。
2. **做空间统一**：例如重采样到 `1.0 × 1.0 × 1.0 mm³`。
3. **做模态特异性强度处理**：
   - CT：HU 截断/窗化。
   - MRI：偏场校正 + White Stripe 等标准化。
4. **做任务对齐**：分类常保留全图，分割常保留 ROI 与 mask 对齐。
5. **保存处理后的影像和处理参数**：避免训练与部署两套规则。

一个实用的输入输出约定可以是：

- **输入**：原始体数据 `image`，可选标签 `mask`，以及 `spacing/origin/direction`。
- **输出**：处理后的 `image_processed`、同步变换后的 `mask_processed`、以及记录参数的 `report.json`。

---

## 片段代码
正文只保留帮助理解的关键片段；完整实现请直接查看 `src/ch05/` 下的脚本。

### 1. CT：把极端 HU 值裁回任务区间
```python
import numpy as np


def clip_hu(image, hu_min=-1000, hu_max=1000):
    image = np.asarray(image, dtype=np.float32)
    return np.clip(image, hu_min, hu_max)
```

- **输入**：原始 CT 体数据，单位仍是 HU。
- **输出**：截断后的 CT 数据，便于后续归一化或窗化。
- 完整实现见 `src/ch05/clip_hu_values/`。

### 2. 重采样：先算缩放比例，再统一体素间距
```python
import numpy as np


def scale_factors(original_spacing, target_spacing):
    original_spacing = np.array(original_spacing, dtype=np.float32)
    target_spacing = np.array(target_spacing, dtype=np.float32)
    return original_spacing / target_spacing
```

- **输入**：如 `(0.7, 0.7, 5.0)` 的原始 spacing。
- **输出**：对应目标 spacing 的缩放因子。
- 完整实现见 `src/ch05/medical_image_resampling/`。

### 3. MRI：用白质带做强度标准化
```python
import numpy as np


def normalize_with_white_stripe(image, wm_mask):
    wm_values = image[wm_mask > 0]
    mean = wm_values.mean()
    std = wm_values.std() + 1e-6
    return (image - mean) / std
```

- **输入**：MRI 图像与白质区域掩膜或其近似估计。
- **输出**：组织间更可比的标准化结果。
- 完整实现见 `src/ch05/white_stripe_normalization/`。

### 4. 金属伪影：先做一个高阈值检测
```python

def detect_metal(image, threshold=3000):
    return image > threshold
```

- **输入**：CT 图像。
- **输出**：疑似金属区域的布尔掩膜。
- 完整实现见 `src/ch05/detect_metal_artifacts/`。

---

## 延伸实践
1. **把“空间统一”和“强度统一”分成两个独立函数**，便于训练与推理共用。
2. **同时保存处理前后的 spacing 与强度范围**，方便回溯实验差异。
3. **尝试比较不同 CT 窗口**：肺窗、软组织窗、骨窗是否能作为多通道输入。
4. **比较 MRI 标准化策略**：White Stripe、z-score、百分位归一化对下游分割的影响。
5. **阅读并运行本地源码**：
   - `src/ch05/clip_hu_values/main.py`
   - `src/ch05/medical_image_resampling/main.py`
   - `src/ch05/n4itk_bias_correction/main.py`
   - `src/ch05/white_stripe_normalization/main.py`
   - `src/ch05/detect_metal_artifacts/main.py`

完成这些练习后，再进入分割、分类和增强章节，你会更容易看懂“为什么同一个网络在不同数据上表现差异巨大”。
