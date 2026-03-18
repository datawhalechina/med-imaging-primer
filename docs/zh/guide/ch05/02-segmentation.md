---
title: 5.2 图像分割：U-Net 及其变体
description: 以“为什么需要 → 核心直觉 → 典型场景 → 最小流程 → 片段代码 → 延伸实践”的顺序理解医学图像分割
---
# 5.2 图像分割：U-Net 及其变体

## 这一节解决什么问题
这一节解决的是：**怎样让模型不仅知道“有没有病”，还知道“病灶或器官到底在哪儿、边界到哪里”。**

## 为什么前面几章还不够
前一节的预处理解决了“输入能不能稳定喂给模型”的问题，但临床上很多任务需要的不是一个总标签，而是更细粒度的空间答案，例如：

- 肺野边界在哪里；
- 肿瘤体积多大；
- 器官是否被侵犯；
- 放疗勾画范围该到哪一层切片结束。

也就是说，仅有分类分数还不够，模型必须给出**像素级或体素级定位**。这正是分割任务出现的原因。

---

## 为什么需要
医学分割最核心的价值，是把“影像读片”转成“可测量区域”。

| 任务 | 分割输出能带来什么 | 典型临床价值 |
| --- | --- | --- |
| 器官分割 | 轮廓、面积、体积 | 手术规划、剂量计算 |
| 病灶分割 | 病灶边界与负荷 | 疗效评估、随访比较 |
| 结构分割 | 血管、气道、肺叶等 | 介入路径、解剖分析 |

相较于自然图像，医学分割更难，因为它往往同时要求：**小样本、强先验、边界精确、类别极不平衡**。

---

## 核心直觉
U-Net 之所以经典，不是因为它“足够深”，而是因为它抓住了医学分割的一个核心矛盾：

- 深层特征更懂“这是什么”；
- 浅层特征更懂“它在哪儿”。

U-Net 用**编码器-解码器**处理“语义”，再用**跳跃连接**把浅层空间细节送回来，于是模型同时兼顾了**全局上下文**和**局部边界**。

![U-Net架构演进](/images/ch05/unet-architecture-zh.png)
*图：U-Net 的关键不只是下采样和上采样，而是在恢复分辨率时把浅层细节重新接回来。*

---

## 典型场景

### 场景 1：肺野分割
- **目标**：从胸部 CT 中得到肺部区域掩膜。
- **价值**：为后续结节检测、感染分析、定量测量提供 ROI。
- **本地源码**：`src/ch05/lung_segmentation_network/main.py`。

### 场景 2：器官或肿瘤分割
- **目标**：肝脏、胰腺、脑肿瘤、前列腺等精确轮廓。
- **价值**：体积估计、病灶负荷监测、放疗靶区勾画。
- **延伸方向**：2D U-Net、3D U-Net、Attention U-Net、nnU-Net。

### 场景 3：分割任务中的数据增强
- **目标**：让 image 与 mask 在同一几何变换下同步变化。
- **本地源码**：`src/ch05/medical_segmentation_augmentation/main.py`。
- **提醒**：对分割任务来说，增强不仅要变换图像，还要同步变换标签。

---

## 最小流程
如果把一个分割项目压缩成最小闭环，通常就是下面 5 步：

1. **准备配对数据**：`image` 与 `mask` 一一对应。
2. **统一预处理**：spacing、方向、强度范围尽量一致。
3. **切块或缩放到固定输入大小**：如 `256×256` 或 `128×128×128`。
4. **模型前向得到 logits**：输出与输入同尺度或可上采样回原尺度。
5. **后处理与评估**：阈值化、连通域筛选、Dice/IoU/敏感性。

一个常见输入输出约定是：

- **输入**：`[B, C, H, W]` 或 `[B, C, D, H, W]` 的图像张量。
- **输出**：`[B, K, H, W]` 或 `[B, K, D, H, W]` 的类别概率图。
- **标签**：单通道 mask 或 one-hot mask。

---

## 片段代码
正文只保留帮助理解网络结构的关键片段；完整训练、评估与可视化脚本请看 `src/ch05/`。

### 1. 一个最小的卷积块
```python
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )
```

- **输入**：某一层特征图。
- **输出**：通道数增加、局部表征更强的特征图。
- **完整实现**：`src/ch05/lung_segmentation_network/main.py`。

### 2. 跳跃连接的关键动作：拼接
```python
import torch


def fuse_skip(upsampled, shallow_feature):
    return torch.cat([upsampled, shallow_feature], dim=1)
```

- **输入**：上采样后的深层特征，与编码器侧的浅层特征。
- **输出**：同时包含语义与边界信息的融合特征。
- **意义**：这是 U-Net 能保住细节的关键一步。

### 3. Dice 的最小实现
```python
import torch


def dice_score(pred_mask, true_mask, eps=1e-6):
    inter = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    return (2 * inter + eps) / (union + eps)
```

- **输入**：预测掩膜与真值掩膜。
- **输出**：`0~1` 的重叠度量。
- **完整评估与报告**：`src/ch05/lung_segmentation_network/main.py` 及其 `output/` 结果文件。

---

## 延伸实践
1. **先跑通肺野分割示例**：阅读 `src/ch05/lung_segmentation_network/main.py`，理解从数据生成到结果可视化的闭环。
2. **比较 2D 与 3D**：二维模型更省显存，三维模型更能利用层间上下文。
3. **尝试替换损失函数**：Dice Loss、BCE + Dice、Focal Loss 在小病灶场景下差异明显。
4. **给分割任务加增强**：阅读 `src/ch05/medical_segmentation_augmentation/main.py`，重点关注 image/mask 同步变换。
5. **补一个后处理实验**：只保留最大连通域，观察 Dice 是否提升。

当你能把“输入、输出、评估”三件事串起来时，U-Net 就不再只是结构图，而会变成一个真正可操作的分割工作流。


::: tip 代码实验 / 实践附录
运行命令、环境依赖、完整输出和可运行 demo 已统一迁移到 [5.6 代码实验 / 实践附录](./06-code-labs.md) 与 `src/ch05/README.md`。
:::
