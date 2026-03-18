---
title: 5.6 代码实验 / 实践附录
description: 第五章统一的实践附录，集中承接运行方法、环境依赖、完整输出与源码入口。
---

# 5.6 代码实验 / 实践附录

这一页是第五章的**实践附录**，职责与 5.1–5.4 的主线章节不同：

- **主线章节**负责解释概念、判断标准和任务思路；
- **本附录**负责集中承接运行方式、依赖、完整实现与输出结果；
- **5.5** 则保持为前沿导读 / 选学。

## 完整实现在哪里？
第五章所有完整脚本、训练逻辑、demo 与结果输出，统一放在 `src/ch05/` 下。

### 按主线问题对照代码目录

| 主线问题 | 对应代码职责 | 代表脚本 |
| --- | --- | --- |
| 数据怎么准备 | 预处理与标准化 | `clip_hu_values/`、`medical_image_resampling/`、`n4itk_bias_correction/`、`white_stripe_normalization/`、`detect_metal_artifacts/`、`visualize_bias_field/` |
| 分割为什么有效 | 分割模型与标签同步增强 | `lung_segmentation_network/`、`medical_segmentation_augmentation/` |
| 分类 / 检测怎么思考 | 分类实验与报告输出 | `medical_image_classification/` |
| 增强 / 恢复何时使用 | 数据增强、对比度增强、恢复性处理 | `medical_image_augmentation/`、`clahe_enhancement/`，以及 MRI 偏场相关脚本 |

---

## 推荐阅读顺序
1. 先看 `src/ch05/README.md`，把第五章代码总入口建立起来。
2. 再进入具体实验子目录，阅读各自 `README.md`。
3. 最后查看各目录下的 `output/` 或 `outputs/`，把图像结果、报告文件和脚本实现连起来。

---

## 常见运行方式
第五章大多数实验都遵循同样的启动模式：

```bash
cd src/ch05/<实验目录>
python main.py
```

有些实验还提供简化入口或测试脚本：

```bash
python simple_augmentation.py
python test.py
```

---

## 环境依赖看哪里？
实践相关依赖说明统一参考：

- `src/ch05/requirements.txt`
- `src/ch05/README.md`
- 各实验子目录中的 `README.md`

常见依赖包括：

- `numpy`、`matplotlib`、`scipy`、`scikit-image`
- `opencv-python`
- `torch`、`torchvision`
- `pydicom`、`nibabel`、`SimpleITK`

---

## 完整输出放在哪里？
各实验目录通常把生成结果放在：

- `output/`
- `outputs/`

例如：

- `src/ch05/lung_segmentation_network/output/`
- `src/ch05/medical_image_classification/output/`
- `src/ch05/medical_image_augmentation/output/`
- `src/ch05/clahe_enhancement/output/`

这些图像、报告和完整输出不再挤进主线文档，而由本附录与 `src/ch05/` 统一承接。

---

## 应该怎么使用这页？
当你要做下面这些事时，就来这一页：

- 想运行第五章代码；
- 想查环境依赖；
- 想找完整 demo 或训练脚本；
- 想看完整输出和报告文件。

当你要回答“为什么这样做”时，再回到 5.1–5.4 主线章节。

::: tip 一句话总结
第五章主线负责解释 **为什么**，本附录和 `src/ch05/` 负责承接 **怎么跑、怎么查、怎么看完整结果**。
:::
