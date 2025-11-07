
# X-ray 成像中的断层成像与数字断层融合


X-射线成像技术经过多年发展，除了传统二维投影成像，还出现了“直接断层成像”（Tomosynthesis）和“数字断层融合/数字断层合成”（Digital Tomosynthesis, DT）两种技术路径。本文首先介绍直接断层合成虽无完整“重建”意义，但通过有限角度投影生成多个平面图像的方法和数学思路；然后深入探讨数字断层融合尤其在乳腺成像中的应用的方法，包括其几何模型、迭代重建框架、正则化机制与代码示例。

## 1. 直接成像（断层合成 Tomosynthesis）

### 1.1 概念说明

“直接成像”在这里是指通过 X-射线源（或探测器）在有限角度范围内（而不是完整 180° 或 360° 旋转）获取一系列投影，并不做传统CT形式的完整重建，而是通过“断层合成”（tomosynthesis）方式生成多个平面（slice）或仿断层图像。这种技术常用于乳腺摄影、牙科成像、胸部成像等场合。其优点包括射线剂量较低、扫描时间短、硬件结构相对简单。从数学上看，因为投影角度少、覆盖范围有限，故无法满足经典 Radon 变换可逆条件，重建质量和深度分辨率受限。但仍能生成“平面聚焦”效果，即对某一深度层面聚焦，而将其它层面模糊化。

### 1.2 几何模型与数学原理

设物体的衰减系数分布为 $f(x,y,z)$ 。X-射线源从若干角度 $\beta\in[\beta_{\min},\beta_{\max}]$ 发射，探测器固定或同步移动，但总角度范围远小于完整 180°。对于源射线第 i 个角度 $(\beta_i)$ ，探测器记录的投影可表示为：

$$
P_i(u,v) = \int f\big(x,y,z\big)\delta\big(u - u(x,y,z,\beta_i)\big)\delta\big(v - v(x,y,z,\beta_i)\big)dxdydz
$$

其中 $(u,v)$ 是探测器坐标，$u(x,y,z,\beta_i)), (v(x,y,z,\beta_i)$ 表示从源点出发穿过 $(x,y,z)$ 到探测器平面上的交点坐标。
为了生成某一深度 $z = z_0$ 层面的合成图像，可将来自不同角度的投影反投影聚合，采用“聚焦”加权策略：

$$
g(x,y; z_0) = \sum_{i} w_i(z_0);P_i\big(u(x,y,z_0,\beta_i),,v(x,y,z_0,\beta_i)\big)
$$

其中 $w_i(z_0)$ 是基于几何距离、焦距、模糊效应等设计的加权系数，使得在 $z = z_0$ 平面上成像清晰，而远离该平面的信号被削弱（模糊化）。这种方式可看作是“有限角度反投影（back-projection）”但不同于完整CT的滤波反投影。
优点：可快速生成多平面图像、剂量低、结构清晰。缺点：深度分辨率差、模糊层较厚、伪影较多。

### 1.3 举例说明

以乳腺成像为例，传统二维 X-射线乳房摄影容易出现组织重叠、微钙化被遮蔽的问题。使用 Tomosynthesis 技术，X-射线源在 ±15° 左右的小角度范围内移动，获取约 15–30张投影，然后生成多层面图像（例如从前到后的1 mm步距切片）。读片医生可以沿不同深度滑动查看，使得重叠结构减少、病灶更易发现。
例如，在某 DBT 系统中，典型角度范围为 ±15°，拍摄约 9–15 个视角。

### 1.4 代码实战：断层合成示例

下面使用 Python 演示一个简化的模拟：假设二维场景 (f(x,y))，源在多个角度获取投影，然后生成聚焦于某深度 (y=y_0) 的“断层合成”图像。

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# 简化模型：使用 2D Phantom 代替 3D
image = shepp_logan_phantom()
n = image.shape[0]
theta = np.linspace(-15., 15., 21)  # ±15°小角度
sinogram = radon(image, theta=theta, circle=True)

# “聚焦”权重：假设深度对应行数 i0
i0 = n//2
weights = np.exp(-((np.arange(n)-i0)**2)/(2*(n*0.1)**2))

# 生成合成“深度”图像 g(x)（沿 y-方向聚焦）
# 简单方式：对 sinogram 每列加权反投影
recons = iradon(sinogram, theta=theta, filter_name=None, circle=True)
g = recons * weights[:, None]

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Original Phantom")
plt.imshow(image, cmap='gray')
plt.subplot(1,3,2)
plt.title("Sinogram (±15°)")
plt.imshow(sinogram, cmap='gray', aspect='auto')
plt.subplot(1,3,3)
plt.title("Focussed Synthesized Slice")
plt.imshow(g, cmap='gray')
plt.tight_layout()
plt.show()
```

该代码虽然是二维简化模型，但体现了“有限角度投影 + 加权反投影生成专深度层面图像”的思路。若换成三维、加上真实几何即可扩展为X-ray Tomosynthesis系统。


## 2. 数字断层融合（Digital Tomosynthesis）重建简介

### 2.1 概念与应用背景

数字断层融合是在 X-射线乳腺成像等场合，一次有限角度扫描获取若干个投影图像，然后通过重建算法生成一个体积或一系列切片，从而实现低剂量、快速扫描的三维或拟三维成像，与传统 CT 相比，其角度范围更小、投影数更少、更适合临床筛查，由于投影数少、角度覆盖有限，重建问题成为“欠定”或“有限角度”的反问题，必须采用专门算法、先验正则化来改善质量。

### 2.2 几何模型与数学框架

设物体衰减系数分布 $f(x,y,z)$ 。源和探测器在一个弧轨迹上运动，获取 N 个角度 $\beta_i,,i=1\ldots N$ 。第 $i$ 个角度所记录投影为：

$$
P_i(u,v) = \int f(x,y,z);\delta\big(u - u(x,y,z,\beta_i)\big)\delta\big(v - v(x,y,z,\beta_i)\big)dxdydz
$$

整个数据获取过程可写为：

$$
g = A,f + \varepsilon
$$

其中：

*  $A$ 是系统矩阵（反映每个体素在每条射线中的贡献）；
*  $f$ 是矢量化的体（体素）值；
*  $g$ 是观测投影数据向量；
*  $\varepsilon$ 是噪声。

由于 $A$ 通常是 “窄角度 + 少视角”，即 (N\ll)正常 CT 视角数，系统为病态或欠定，则常采用以下优化模型重建：

$$
\min_{f\ge0} ; \frac12|A,f - g|_2^2 + \lambda,R(f)
$$

其中 $R(f)$ 为正则化项，如TV（总变分）、Tikhonov等。 

### 2.3 常见重建方法对比

* 有限角度滤波反投影（FBP-限角）：将传统 FBP 方法应用于有限视角情况下，速度快但伪影严重。
* 代数重建方法（ART/SART）：基于解线性系统 $A,f=g$ 的迭代方法，适合少视角。
* 模型-基（Model-Based）迭代重建：将系统物理模型、噪声统计、正则化整合到优化框架内，效果更佳。
* 压缩感知／稀疏重建：借助TV或ℓ1正则化减少伪影、恢复细节。

### 2.4 数学推导示例

以 TV 正则化模型为例，目标函数：

$$
\min_{f\ge0} \frac12|A,f - g|*2^2 + \lambda;\sum*{ijk}\sqrt{\big(D_x f_{ijk}\big)^2 + \big(D_y f_{ijk}\big)^2 + \big(D_z f_{ijk}\big)^2}
$$

其中 $(D_x, D_y, D_z)$ 分别是体素在三个方向上的差分算子。
梯度下降或变分 求解流程可写成：

$$
f^{(k+1)} = \max \Big( 0, ; f^{(k)} - \alpha \big[ A^T (A f^{(k)} - g) + \lambda , \nabla R(f^{(k)}) \big] \Big)
$$

其中：

* $f^{(k)}$ ：第 $k$ 次迭代的重建体积。
* $\alpha$ ：步长（learning rate）。
* $A^T(A f^{(k)} - g)$ ：数据保真项梯度。
* $\nabla R(f^{(k)})$ ：TV 正则化梯度
* $\lambda$ ：TV 正则化系数。
* $\max(0, \cdot)$ ：保证 $f \ge 0$（非负约束）。

对于 TV 项，其sub-gradient近似可为：

$$
\nabla R(f)*{ijk} \approx -\mathrm{div}\left(\frac{\nabla f*{ijk}}{|\nabla f_{ijk}|+\varepsilon}\right)
$$

上述框架适用于 DBT 场景，因为投影数据欠定，正则化尤为重要。

### 2.5 举例说明
在乳腺 DBT 系统中，由于角度通常为 ±15°–±30°、投影数约 9–25 张，传统 CT 滤波反投影会产生严重层析模糊与伪影。通过模型-基迭代重建结合TV正则化，可显著改善微钙化、结构扭曲检测性能。

### 2.6 代码实战：数字断层融合简化实现

下面演示一个简化2D有限角度重建Python示例，使用scikit-image实现FBP限角，再用迭代方法（梯度下降 + TV）做正则化。

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# 生成 phantom
image = shepp_logan_phantom()
n = image.shape[0]
theta = np.linspace(-20., 20., 41)  # ±20° 限角
sinogram = radon(image, theta=theta, circle=True)

# (1) 有限角度 FBP 重建
recon_fbp = iradon(sinogram, theta=theta, filter_name='ramp', circle=True)

# (2) 迭代重建 + TV 正则化
# 简化梯度下降实现
# 构造系统矩阵 A 显式不现实，此处用简化 pseudo-inverse 指令模拟
recon = np.zeros_like(image)
lambda_tv = 0.1
alpha = 0.5
num_iter = 50

for k in range(num_iter):
    # 模拟数据拟合项梯度（这里用重投影-误差反投影近似）
    resid = radon(recon, theta=theta, circle=True) - sinogram
    backproj = iradon(resid, theta=theta, filter_name=None, circle=True)
    grad_data = backproj

    # TV 梯度近似
    grad_tv = np.zeros_like(image)
    eps = 1e-3
    # 计算简单 2D 差分
    dx = np.roll(recon, -1, axis=1) - recon
    dy = np.roll(recon, -1, axis=0) - recon
    magnitude = np.sqrt(dx*dx + dy*dy + eps)
    grad_tv = (dx/ magnitude) - (np.roll(dx,1,axis=1)/np.roll(magnitude,1,axis=1)) \
            + (dy/ magnitude) - (np.roll(dy,1,axis=0)/np.roll(magnitude,1,axis=0))

    # 更新
    recon = recon - alpha*(grad_data + lambda_tv * grad_tv)
    recon = np.clip(recon, 0, None)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Limited-Angle Sinogram")
plt.imshow(sinogram, cmap='gray', aspect='auto')
plt.subplot(1,3,2)
plt.title("FBP (±20°) Reconstruction")
plt.imshow(recon_fbp, cmap='gray')
plt.subplot(1,3,3)
plt.title("Iterative + TV Reconstruction")
plt.imshow(recon, cmap='gray')
plt.tight_layout()
plt.show()
```

## 3. 深度学习辅助的数字断层融合（DL-DT Reconstruction）

### 3.1 背景与动机

传统 DT 重建（如 FBP、SART、MBIR）受限于投影角度少、数据欠定性强，易出现层析模糊与伪影。近年来，深度学习（Deep Learning, DL）通过从大量配对样本中学习投影→体积/切片的非线性映射，能显著提升重建质量。
深度学习在 Digital Tomosynthesis 中的应用主要集中于以下三类思路：

1). 端到端学习（End-to-End Learning）
   直接从投影或有限角度 FBP 重建图像映射到目标高质量断层图像：
   
   $$
   f_{\text{DL}} = \mathcal{N}*\theta(g)
   $$
   
   其中 $\mathcal{N}*\theta$ 为深度网络如U-Net、ResNet、Transformer等。

2). 物理约束型（Physics-Informed DL）
   将系统矩阵 $A$ 的前向模型与网络结合：
   
   $$
   f^{(k+1)} = f^{(k)} - \alpha,A^T(Af^{(k)} - g) - \mathcal{N}_\theta(f^{(k)})
   $$
   
   即在迭代框架中嵌入网络更新项，保证物理一致性。

3). 混合式（Hybrid Model）
   网络与传统算法交替使用：
   
   * 用 FBP 生成初始重建；
   * 网络补偿伪影或恢复细节；
   * 或者在每步迭代后用 CNN 进行正则化平滑。


### 3.2 数学建模框架

设观测方程：

$$
g = A f + \varepsilon
$$

DL-DT重建可写为以下优化学习形式：

$$
\min_{\theta,f} ; |A f - g|*2^2 + \lambda,R*\theta(f)
$$

其中 $R_\theta(f)$ 表示由神经网络 $\mathcal{N}_\theta$ 隐式学习到的先验正则化。

若采用“学习正则项”的可学习迭代框架（Learned Iterative Reconstruction, LIR），第 $k$ 步更新为：

$$
f^{(k+1)} = f^{(k)} - \alpha_k,A^T(A f^{(k)} - g) - \beta_k,\mathcal{N}*\theta^{(k)}(f^{(k)})
$$

其中 $\mathcal{N}*\theta^{(k)}$ 可为U-Net或ResNet模块。



### 3.3 主流模型架构示例

| 类别                 | 方法                                          | 特点                            |
| ------------------ | ------------------------------------------- | ----------------------------- |
| **端到端重建网络**        | FBPConvNet ([Jin et al., 2017])             | 以 FBP 结果为输入，U-Net 修复伪影        |
| **学习迭代模型**         | Learned Primal-Dual ([Adler & Öktem, 2018]) | 将前向算子 $A$ 和 $A^T$ 融入网络循环中     |
| **深度正则化**          | RED-Net ([Romano et al., 2017])             | CNN 充当隐式正则化项                  |
| **Transformer 重建** | TransRecon ([Zhao et al., 2022])            | 使用 Vision Transformer 编码全局上下文 |

> **示意流程：**
>
> ```
> 投影数据 g  ─>  FBP 初步重建 ─> 深度网络补偿伪影 ─> 高质量体/切片输出
> ```
>
> 或在 Learnable Iterative 框架下：
>
> ```
> (f^k, g) ─> A^T(Af^k - g) ─> CNN/Transformer 修正 ─> f^{k+1}
> ```


### 3.4 PyTorch 实战示例（简化2DTomosynthesis）

下面的代码演示了一个小规模 FBPConvNet 式网络,输入为有限角度FBP结果，输出为增强重建（通过 U-Net 结构实现）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 简化版 U-Net ===
class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU())
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        d1 = self.up1(e2)
        cat = torch.cat([d1, e1], dim=1)
        out = self.dec1(cat)
        return self.out_conv(out)

# === 网络使用示例 ===
model = UNet2D()
fbp_input = torch.randn(1, 1, 256, 256)  # 模拟有限角度 FBP 重建
enhanced = model(fbp_input)
print(enhanced.shape)  # [1, 1, 256, 256]
```

这种网络训练时常采用配对监督学习：

$$
\mathcal{L}(\theta) = | \mathcal{N}*\theta(g*{\text{FBP}}) - f_{\text{GT}}|*1 + \lambda |\nabla \mathcal{N}*\theta(g_{\text{FBP}})|_1
$$

其中第二项可为 TV 或梯度一致性约束，用于保持结构锐度。


### 3.5 深度学习模型与物理模型融合趋势

近年来出现了“物理-网络融合”的框架，如：

$$
f^{(k+1)} = \mathrm{CNN}_\theta^{(k)}!\big(f^{(k)}, A^T(g - A f^{(k)})\big)
$$

即在每次更新中同时输入数据一致性项与当前重建，网络自动学习融合策略,这种 “Learned Primal-Dual” 框架在有限角度DT场景下表现优异，可减少伪影、提升微结构识别能力。


### 3.6 前沿研究与发展方向

| 方向                 | 代表工作                 | 核心思路           |
| ------------------ | -------------------- | -------------- |
| **3D 卷积重建网络**      | DBT-Net (2022)       | 直接学习 3D 体积特征   |
| **Transformer 重建** | TransTomosyn (2023)  | 长程依赖建模         |
| **自监督重建**          | Noise2Inverse (2021) | 无需配对高质量 GT     |
| **联合学习**           | ReconFormer (2024)   | 融合多层次注意力与物理一致性 |
| **AI + 物理一致性优化**   | DeepMBIR (2023)      | 神经网络辅助的模型-基重建  |

>这些方向的共同目标是：
>
> 减少伪影 + 提升纹理细节 + 保留几何结构一致性
> 在保持低剂量和少角度的前提下，获得接近 CT 品质的层析成像。


## 4. 总结
直接成像（断层合成）技术在 X-射线中通过有限角度投影和加权反投影生成多个切片图像，是一种快速、低剂量的拟三维成像手段,数字断层融合是将有限角度投影数据通过数学重建（尤其迭代 + 正则化）生成体积或切片的技术，尤其在乳腺成像中应用广泛,虽然角度和投影数远少于传统 CT，但通过模型-基重建与正则化如 TV可以显著提高图像质量。深度学习极大增强了 DT 的重建质量，使有限角度问题得到数据驱动修复，“端到端 + 物理约束 + 正则化学习” 成为主流路线。未来发展方向包括：物理一致性深度网络；自监督/无配对重建；跨模态（X-ray → CT）知识迁移；多能谱 Tomosynthesis 与分层学习重建。


## 参考文献
[1] M. Ertas et al., “Digital breast tomosynthesis image reconstruction using 2D …”, PMC, 2013.

[2] J. Chung et al., “Numerical Algorithms for Polyenergetic Digital Breast Tomosynthesis”, SIAM, 2010. 

[3] R. Cavicchioli et al., “GPU acceleration of a model-based iterative method for Digital Breast Tomosynthesis”, Scientific Reports, 2020. 

[4] I. Samiry, I. Ait Lbachir, et al., “Digital Breast Tomosynthesis Reconstruction Techniques in Healthcare Systems: A Review”, LNBI, 2023.

[5] G. Yang, J.H. Hipwell, D.J. Hawkes, S.R. Arridge, “Numerical Methods for Coupled Reconstruction and Registration in Digital Breast Tomosynthesis”, arXiv, 2013. 

[6] I. Reiser, J. Bian, R.M. Nishikawa, E.Y. Sidky, X. Pan, “Comparison of reconstruction algorithms for digital breast tomosynthesis”, arXiv, 2009. 

[7] Jin K.H. et al., *Deep Convolutional Neural Network for Inverse Problems in Imaging (FBPConvNet)*, IEEE TMI, 2017.

[8] Adler J., Öktem O., *Learned Primal-Dual Reconstruction*, IEEE TIP, 2018.

[9] Romano Y., Elad M., Milanfar P., *RED: Regularization by Denoising*, SIAM JIS, 2017.

[10] Zhao Z. et al., *TransRecon: Transformer-based Image Reconstruction for Limited-Angle Tomosynthesis*, MedIA, 2022.

[11] Chen Y. et al., *DeepMBIR: Model-Based Iterative Reconstruction using Deep Learning Priors*, IEEE TMI, 2023.

