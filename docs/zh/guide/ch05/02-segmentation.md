---
title: 5.2 图像分割：U-Net 及其变体
description: 用统一模板理解医学图像分割：为什么需要像素级答案，以及 U-Net 为什么有效
---
# 5.2 图像分割：U-Net 及其变体

## 开场问题
这一节回答的是：**怎样让模型不仅知道“有没有病”，还知道“病灶或器官到底在哪儿、边界到哪里”。**

读者通常会在这些场景里感到分类不够用：

- 肿瘤体积要怎么量；
- 肺野边界到底落在哪一圈像素；
- 放疗靶区要勾到哪一层切片结束；
- 手术规划时器官和病灶的空间关系怎么描述。

这些问题都说明，临床并不总满足于一个总标签，它往往还需要**像素级或体素级定位**。

---

## 直觉解释
分割可以先不理解成“复杂网络”，而理解成：**把影像读片结果变成可测量区域。**

分类只回答“有没有”“像不像”；分割则进一步回答：

- 这块区域在哪里；
- 面积有多大；
- 体积是多少；
- 边界和周围结构怎么接触。

U-Net 之所以经典，也不是因为它“层数很多”，而是它抓住了一个核心矛盾：

- 深层特征更擅长判断“这是什么”；
- 浅层特征更擅长保留“它在哪儿”。

于是它一边往下提取语义，一边在上采样时把浅层细节接回来。

![U-Net架构演进](/images/ch05/unet-architecture-zh.png)
*图：U-Net 的关键不是单纯下采样和上采样，而是通过跳跃连接把边界细节重新接回来。*

---

## 核心方法
这一节先抓住 4 个关键点。

### 1. 明确分割输出是什么
分割输出通常不是一个总分，而是一张与输入同尺度或可映射回原图尺度的概率图或掩膜。

### 2. 同时保留语义与边界
编码器负责抽取更高层的语义；解码器负责恢复空间分辨率；跳跃连接保证小结构和边界不会在下采样过程中完全丢掉。

### 3. 让标签和图像严格对齐
对分割任务来说，最怕的不是“增强不够多”，而是 image 和 mask 变换不同步。一旦错位，模型学到的就不再是真正边界。

### 4. 评估不能只看损失下降
Dice、IoU、敏感性、连通域表现、后处理效果，往往比单纯的训练 loss 更接近实际任务质量。

---

## 典型案例
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
- **提醒**：增强图像时，标签也必须同步变换。

---

## 实践提示
正文只保留帮助理解结构的关键片段；完整训练、评估与可视化请看 `src/ch05/`。

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

### 2. 跳跃连接的关键动作：拼接
```python
import torch


def fuse_skip(upsampled, shallow_feature):
    return torch.cat([upsampled, shallow_feature], dim=1)
```

### 3. Dice 的最小实现
```python
import torch


def dice_score(pred_mask, true_mask, eps=1e-6):
    inter = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    return (2 * inter + eps) / (union + eps)
```

### 4. 分割项目里最容易忽略的检查
- image / mask 是否一一对应；
- 预处理后标签是否还和原图对齐；
- 小病灶是否在缩放时被抹掉；
- 后处理是否错误删掉了真实病灶。

---

## 小结
这一节学会了：**分割的价值在于把“看见病灶”变成“圈出范围并可测量”，而 U-Net 的直觉核心是同时保留语义和边界。**

下一节接着讲分类和检测，是因为并不是所有任务都需要精细轮廓；很多临床流程更先需要回答：**有没有病、是什么病、可疑区域大概在哪儿。**
