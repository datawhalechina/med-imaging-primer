---
title: 5.4 图像增强与恢复
description: 以“为什么需要 → 核心直觉 → 典型场景 → 最小流程 → 片段代码 → 延伸实践”的顺序理解医学图像增强与恢复
---

# 5.4 图像增强与恢复

## 这一节解决什么问题
这一节解决的是：**当数据太少、图像太差、对比度不够或存在退化时，如何在不破坏医学意义的前提下，让模型更稳、图像更可用。**

## 为什么前面几章还不够
前面的预处理、分割、分类都默认一个前提：输入图像“已经足够好”。但真实数据并不总满足这个前提：

- 训练数据量少，模型容易过拟合；
- X 光或 CT 局部对比度不足，小病灶不明显；
- MRI/CT 中存在噪声、偏场或伪影，影响视觉判断和模型学习。

所以这一节补上两个现实问题：

1. **增强（augmentation）**：为训练制造更多“合理变化”。
2. **恢复/增强（enhancement & restoration）**：改善原图可见性，让关键信息更突出。

---

## 为什么需要
可以把它们区分成两个方向：

| 方向 | 直接目标 | 典型收益 |
| --- | --- | --- |
| 数据增强 | 扩大训练分布 | 提高泛化、缓解过拟合 |
| 图像增强/恢复 | 改善可见性或修复退化 | 提升对比度、降低干扰、辅助观察 |

医学影像里的关键约束是：**变化必须合理，不能把病灶“增强没了”，也不能凭空造出误导性结构。**

---

## 核心直觉
这一节最重要的直觉有两条：

- **增强不是越猛越好，而是越贴近临床采集变化越好。**
- **恢复不是让图更“好看”，而是让目标结构更“可判读”。**

例如：

- 对 CT 做小角度旋转、轻微噪声扰动，通常合理；
- 对胸片随意做大角度几何变换，就可能破坏解剖关系；
- 对 X 光做 CLAHE，目的不是制造戏剧化对比，而是提升局部结构可见性。

---

## 典型场景

### 场景 1：训练阶段的数据增强
- **用途**：缓解小样本问题，提高模型对采集波动的鲁棒性。
- **典型操作**：旋转、平移、缩放、翻转、噪声、局部遮挡、弹性变形。
- **本地源码**：`src/ch05/medical_image_augmentation/main.py`、`src/ch05/medical_image_augmentation/simple_augmentation.py`。

### 场景 2：X 光或低对比度图像的可见性增强
- **用途**：让边界、纹理和密度变化更容易观察。
- **典型方法**：CLAHE。
- **本地源码**：`src/ch05/clahe_enhancement/main.py`。

### 场景 3：质量退化的恢复性处理
- **用途**：减少偏场、非均匀照明或局部退化的影响。
- **相关源码**：`src/ch05/visualize_bias_field/main.py`、`src/ch05/n4itk_bias_correction/main.py`。

---

## 最小流程
无论是训练增强还是图像恢复，都可以先从下面这个最小闭环开始：

1. **明确目标**：是为了训练鲁棒性，还是为了增强可见性。
2. **限定医学约束**：哪些结构不能变形过度，哪些强度范围不能破坏。
3. **只保留少量、可解释的变换**：先小规模，再逐步扩展。
4. **输出前后对比图和定量指标**：例如对比度、熵、边缘强度、Dice 变化。
5. **把完整实现留在独立脚本中**：正文只解释关键片段与输入输出。

---

## 片段代码
正文只保留帮助理解的短片段；完整实现、参数扫描和可视化请看 `src/ch05/` 下脚本。

### 1. 一个最小的几何增强片段
```python
from skimage.transform import rotate


def small_rotation(image, angle=5):
    return rotate(image, angle, preserve_range=True)
```

- **输入**：单张医学图像。
- **输出**：小角度旋转后的图像。
- 完整实现见 `src/ch05/medical_image_augmentation/`。

### 2. 一个最小的强度扰动片段
```python
import numpy as np


def adjust_contrast(image, factor=1.1):
    mean = np.mean(image)
    return (image - mean) * factor + mean
```

- **输入**：原图与对比度系数。
- **输出**：保持整体结构、轻微改变局部对比的结果。
- **适用**：多中心数据带来的亮度/对比度差异模拟。

### 3. CLAHE 的关键调用
```python
import cv2


def run_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)
```

- **输入**：8-bit 灰度图像。
- **输出**：局部对比度增强后的图像。
- 完整实现见 `src/ch05/clahe_enhancement/`。

---

## 关键命令 + 结果截图 + 结果解读
下面把原先冗长的运行日志收缩成最有用的三件事：命令、截图、解释。

### 关键命令
```bash
cd src/ch05/medical_image_augmentation
python simple_augmentation.py

cd ../clahe_enhancement
python main.py
```

### 结果截图

**数据增强结果示意：**

![医学图像增强结果截图](/images/ch05/medical_image_augmentation_ct_demo.png)

### 结果解读
- 上图展示的是**同一张 CT 输入经过多种增强后的对比**，目的不是替代原图，而是扩展训练时模型能见到的数据分布。
- 如果你看到增强后器官轮廓仍合理、病灶相关区域没有被严重扭曲，说明增强策略基本可用。
- 如果局部结构被拉扯得不再符合解剖常识，就说明参数过强，应该回退。
- 对于 CLAHE，建议结合 `src/ch05/clahe_enhancement/output/` 下的图像一起看：重点观察**边缘是否更清楚**，而不是单纯看“图像是否更亮”。

---

## 延伸实践
1. **先固定一套保守增强**：小角度旋转、轻度对比度扰动、少量噪声，观察下游指标变化。
2. **比较“增强前训练”和“增强后训练”**：不要只看训练集分数，更要看验证集与外部数据。
3. **做一次 CLAHE 参数扫描**：比较不同 `clip_limit` 与 `tile_grid_size` 对边缘和噪声的影响。
4. **把增强与分割/分类任务联动**：同样的增强策略，在分类上有效，不代表在分割上也有效。
5. **继续阅读本地源码**：
   - `src/ch05/medical_image_augmentation/main.py`
   - `src/ch05/medical_image_augmentation/simple_augmentation.py`
   - `src/ch05/clahe_enhancement/main.py`
   - `src/ch05/visualize_bias_field/main.py`
   - `src/ch05/n4itk_bias_correction/main.py`

当你把“增强”和“恢复”分清楚之后，就能更有针对性地决定：什么时候该扩增训练分布，什么时候该改善原始图像质量。
