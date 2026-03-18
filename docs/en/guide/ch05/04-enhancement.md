---
title: 5.4 Image Augmentation and Recovery
description: Medical image augmentation and recovery techniques
---

# 5.4 Image Augmentation and Recovery

> This mainline page answers the Chapter 5 question **"When should enhancement or restoration be used?"** Commands, dependencies, full logs, and output folders are intentionally moved to [5.6 Code Labs / Practice Appendix](./06-code-labs.md) and `src/ch05/README_EN.md`.

> "Data augmentation is the 'poor man's weapon' in medical imaging deep learning, while image recovery is a 'time machine' that can reconstruct lost information."  A classic metaphor in medical imaging research

In the previous chapters, we learned about the core technologies of preprocessing, segmentation, classification, and detection. Now, we'll explore two critical topics: **image augmentation** and **image recovery**. While these techniques have different goals, both are dedicated to improving the quality and information content of medical images.

The field of medical imaging faces unique challenges: data scarcity, variations in acquisition conditions, noise interference, and inevitable image quality degradation. Image augmentation enhances model generalization by generating more diverse training data, while image recovery aims to repair degraded image quality. Let's dive deep into these two important areas.


## 🎨 Medical Image Augmentation Techniques

### Basic Data Augmentation

#### Geometric Transformations

Geometric transformations for medical images require special consideration, as anatomical spatial relationships cannot be arbitrarily altered:

[📖 **Complete Code Example**: `data_augmentation/`](https://github.com/datawhalechina/med-imaging-primer/tree/main/src/ch05/) - Complete medical image augmentation implementation, 2D/3D transformations, and modality adaptation]

**Execution Results Analysis:**

```
Create CT image augmentation pipeline:
  Image size: (256, 256)
  Augmentation probability: 0.8
  Rotation range: ±5°
  Translation range: ±5.0%
  Scaling range: ±10.0%

Execute spatial augmentation...
  Applied rotation: 3.2°
  Applied translation: (2.1, -1.8) pixels
  Applied scaling: 1.05x
  Applied elastic deformation: α=1000, σ=8

Execute intensity augmentation...
  Applied contrast adjustment: 1.15x
  Added Gaussian noise: σ=12.3 HU
  Output range check: [-1000, 1000] HU

Augmentation complete:
  Original image size: (256, 256)
  Augmented image size: (256, 256)
  Anatomy preservation: Yes
  Pathology preservation: Yes
```

**Algorithm Analysis:** Medical image augmentation increases training data diversity through geometric and intensity transformations. The execution results show that CT image rotation is limited to ±5°, translation range to ±5%, ensuring anatomical structure reasonableness. Elastic deformation parameters (α=1000, σ=8) provide moderate deformation intensity while increasing data diversity and maintaining clinical significance. Noise addition simulates electronic noise from real CT acquisition, improving model robustness.

### Medical Constraints Framework for Augmentation

#### Three-Level Medical Constraints

Medical image augmentation must satisfy three critical constraint levels, distinguishing it fundamentally from natural image augmentation:

##### Level 1: Anatomical Integrity Constraints
- **Anatomical Structure Preservation**: Transformations must respect anatomical relationships that are fixed in nature
- **Organ Boundary Maintenance**: Cannot distort organ boundaries beyond physiological limits
- **Spatial Relationship Preservation**: Relative positions between organs must remain consistent
- **Examples of violations**: Extreme rotation (>15°) violates natural head/body alignment; displacement >10% of image size may violate vascular path physics

##### Level 2: Pathology Authenticity Constraints
- **Lesion Feature Preservation**: Pathological features must remain recognizable and clinically relevant
- **Disease Pattern Consistency**: Augmentation cannot create unrealistic disease morphologies
- **Progression Plausibility**: Augmented pathology must follow realistic disease progression patterns
- **Examples of violations**: Noise addition that obscures lesion boundaries; intensity changes that make disease undiagnosable; rotation that prevents radiologist recognition

##### Level 3: Clinical Applicability Constraints
- **Acquisition Method Realism**: Augmentation must simulate realistic variations from actual acquisition protocols
- **Equipment Variation Simulation**: Can model different scanner generations, but not physically impossible scenarios
- **Clinical Decision Impact**: Augmentation must not change clinical decision thresholds
- **Examples of violations**: Creating image quality worse than worst clinical scenario; simulating artifacts from non-existent equipment; introducing noise patterns never seen clinically

#### Modality-Specific Augmentation Requirements

| Imaging Modality | Key Challenge | Recommended Augmentation | Prohibited Operations | Clinical Validation | Risk Level |
|---|---|---|---|---|---|
| **CT** | Preserve HU values physical meaning | Window/level adjustment, elastic deformation (±5°), noise injection | Extreme rotation (>10°), arbitrary intensity scaling | Compare with multi-protocol scans | HIGH |
| **MRI** | Preserve sequence-specific contrast | Intensity transformation within sequence range, elastic deformation, motion simulation | Sequence mixing, arbitrary signal inversion | Ensure tissue T1/T2 relationships intact | HIGH |
| **X-ray** | Preserve projection geometry and density | Elastic deformation (mild), intensity variation, noise addition | Geometric distortion (>15°), extreme scaling | Ensure silhouettes match radiological signs | MEDIUM |
| **Ultrasound** | Preserve speckle patterns | Speckle reduction, gain adjustment, focal point variation | Remove speckle completely, change beam angle | Maintain acoustic shadow/enhancement patterns | MEDIUM |

##### Clinical Validation Requirement

**Critical Requirement**: All augmentation strategies must undergo radiologist verification to ensure they produce clinically realistic variations rather than introducing non-clinical artifacts.

- **Validation Process**:
  1. Generate augmented image samples
  2. Radiologist review and classification
  3. Compare with real clinical variants
  4. Approve if indistinguishable from clinical reality

- **Approval Criteria**:
  - ✓ Augmentation creates realistic clinical variants
  - ✓ No introduction of non-clinical artifacts
  - ✓ Pathological features remain diagnostically relevant
  - ✗ Reject if clinically non-realistic


### Core Principles of Medical Image Augmentation:

1. **Anatomical Reasonableness**: Transformations must maintain correct anatomical relationships
2. **Pathology Preservation**: Do not alter or obscure key pathological features
3. **Modality-Specific Properties**: Adapt augmentation strategies for different imaging modalities
4. **Clinical Relevance**: Augmentation effects should have practical clinical significance

### Advanced Augmentation Techniques

#### Medical-specific Augmentation Strategies

```python
class MedicalSpecificAugmentation:
    """
    Medical image-specific augmentation strategies
    """
    def __init__(self, modality='ct'):
        self.modality = modality.lower()

    def ct_augmentation(self, image, mask=None):
        """
        CT image-specific augmentation
        """
        # Random HU value range adjustment
        def adjust_hu_window(img, center=None, width=None):
            if center is None:
                center = np.random.uniform(-100, 100)
            if width is None:
                width = np.random.uniform(200, 400)

            # Apply window/level
            img_min = center - width // 2
            img_max = center + width // 2

            img_clipped = np.clip(img, img_min, img_max)
            img_normalized = ((img_clipped - img_min) / (img_max - img_min)) * 255

            return img_normalized.astype(np.uint8)

        # Simulate different scanning parameters
        def simulate_scan_parameters(img):
            # Add noise (simulate different mAs)
            noise_level = np.random.uniform(1, 10)
            noise = np.random.normal(0, noise_level, img.shape)
            img_noisy = img + noise

            # Simulate artifacts (like motion artifacts)
            if np.random.random() < 0.3:  # 30% probability of adding artifacts
                motion_blur = cv2.GaussianBlur(img, (15, 15), 3)
                alpha = np.random.uniform(0.1, 0.3)
                img_noisy = (1 - alpha) * img_noisy + alpha * motion_blur

            return img_noisy

        # Apply augmentation
        augmented = image.copy()
        augmented = adjust_hu_window(augmented)
        augmented = simulate_scan_parameters(augmented)

        if mask is not None:
            return augmented, mask
        return augmented

    def mri_augmentation(self, image, mask=None):
        """
        MRI image-specific augmentation
        """
        # Bias field simulation
        def simulate_bias_field(img):
            x, y = np.meshgrid(np.linspace(-1, 1, img.shape[0]),
                              np.linspace(-1, 1, img.shape[1]))
            bias_field = 1.0 + 0.2 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
            return img * bias_field

        # SNR variation simulation
        def simulate_snr_variation(img):
            snr_factor = np.random.uniform(0.5, 1.5)
            noise = np.random.normal(0, np.std(img) / snr_factor, img.shape)
            return img + noise

        # Apply augmentation
        augmented = image.copy()
        augmented = simulate_bias_field(augmented)
        augmented = simulate_snr_variation(augmented)

        if mask is not None:
            return augmented, mask
        return augmented

    def xray_augmentation(self, image, mask=None):
        """
        X-ray image-specific augmentation
        """
        # Simulate different exposure conditions
        def simulate_exposure_variation(img):
            exposure_factor = np.random.uniform(0.7, 1.3)
            return np.clip(img * exposure_factor, 0, 255)

        # Simulate scatter artifacts
        def simulate_scatter_artifact(img):
            scatter_strength = np.random.uniform(0, 20)
            scatter = np.random.normal(scatter_strength, scatter_strength/4, img.shape)
            return np.clip(img + scatter, 0, 255)

        # Apply augmentation
        augmented = image.copy()
        augmented = simulate_exposure_variation(augmented)
        augmented = simulate_scatter_artifact(augmented)

        if mask is not None:
            return augmented, mask
        return augmented
```

### Advanced Augmentation Techniques

#### Mixup and CutMix

```python
import torch.nn.functional as F

class MedicalMixup:
    """
    Medical image Mixup techniques
    """
    def __init__(self, alpha=1.0, cutmix_prob=0.5):
        self.alpha = alpha
        self.cutmix_prob = cutmix_prob

    def mixup_data(self, x, y, alpha=1.0):
        """
        Standard Mixup implementation
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y, alpha=1.0):
        """
        CutMix implementation
        """
        assert alpha > 0
        lam = np.random.beta(alpha, alpha)

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda to match actual cropped area proportion
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        y_a, y_b = y, y[index]

        return x, y_a, y_b, lam

    def rand_bbox(self, size, lam):
        """
        Generate random bounding box
        """
        W = size[2]
        H = size[3]

        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform distribution
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def forward(self, x, y):
        """
        Mixed augmentation strategy
        """
        if np.random.random() < self.cutmix_prob:
            return self.cutmix_data(x, y, self.alpha)
        else:
            return self.mixup_data(x, y, self.alpha)
```

#### Adversarial Augmentation

```python
import torch.nn as nn

class AdversarialAugmentation:
    """
    Adversarial augmentation
    """
    def __init__(self, model, epsilon=0.01, alpha=0.003, num_iter=5):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def fgsm_attack(self, image, label, epsilon=None):
        """
        FGSM adversarial attack
        """
        if epsilon is None:
            epsilon = self.epsilon

        image.requires_grad = True
        output = self.model(image)
        loss = F.cross_entropy(output, label)

        self.model.zero_grad()
        loss.backward()

        # Get gradient
        data_grad = image.grad.data

        # Generate adversarial samples
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image

    def pgd_attack(self, image, label, epsilon=None, alpha=None, num_iter=None):
        """
        PGD attack
        """
        if epsilon is None:
            epsilon = self.epsilon
        if alpha is None:
            alpha = self.alpha
        if num_iter is None:
            num_iter = self.num_iter

        perturbed_image = image.clone().detach()
        perturbed_image.requires_grad = True

        for _ in range(num_iter):
            output = self.model(perturbed_image)
            loss = F.cross_entropy(output, label)

            self.model.zero_grad()
            loss.backward()

            data_grad = perturbed_image.grad.data

            # PGD step
            perturbed_image = perturbed_image + alpha * data_grad.sign()
            perturbation = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = image + perturbation
            perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()
            perturbed_image.requires_grad = True

        return perturbed_image
```


## 🤖 Deep Learning-driven Augmentation

### Learning Augmentation Strategies

#### AutoAugmentation

```python
import torch.optim as optim

class AutoAugmentation:
    """
    Automatic augmentation strategy learning
    """
    def __init__(self, num_policies=5, num_operations=10):
        self.num_policies = num_policies
        self.num_operations = num_operations
        self.policies = self._initialize_policies()

    def _initialize_policies(self):
        """
        Initialize augmentation strategies
        """
        # Medical image-specific operations
        operations = [
            'rotate', 'translate_x', 'translate_y', 'shear_x', 'shear_y',
            'contrast', 'brightness', 'gamma', 'noise', 'blur'
        ]

        policies = []
        for _ in range(self.num_policies):
            policy = []
            for _ in range(2):  # Each policy contains 2 sub-operations
                op = np.random.choice(operations)
                prob = np.random.uniform(0.1, 0.9)
                magnitude = np.random.uniform(0.1, 1.0)
                policy.append((op, prob, magnitude))
            policies.append(policy)

        return policies

    def apply_policy(self, image, policy_index):
        """
        Apply specified augmentation policy
        """
        policy = self.policies[policy_index]
        augmented = image.copy()

        for op, prob, magnitude in policy:
            if np.random.random() < prob:
                augmented = self._apply_operation(augmented, op, magnitude)

        return augmented

    def _apply_operation(self, image, operation, magnitude):
        """
        Apply single operation
        """
        if operation == 'rotate':
            angle = magnitude * 30  # Maximum 30 degree rotation
            return ndimage.rotate(image, angle, reshape=False)

        elif operation == 'translate_x':
            shift = int(magnitude * image.shape[1] * 0.1)
            return np.roll(image, shift, axis=1)

        elif operation == 'translate_y':
            shift = int(magnitude * image.shape[0] * 0.1)
            return np.roll(image, shift, axis=0)

        elif operation == 'contrast':
            return np.clip(image * (1 + (magnitude - 0.5) * 0.5), 0, 255)

        elif operation == 'brightness':
            return np.clip(image + (magnitude - 0.5) * 50, 0, 255)

        elif operation == 'gamma':
            gamma = 0.5 + magnitude * 1.5
            return np.power(image / 255.0, gamma) * 255.0

        elif operation == 'noise':
            noise = np.random.normal(0, magnitude * 20, image.shape)
            return np.clip(image + noise, 0, 255)

        elif operation == 'blur':
            kernel_size = int(magnitude * 5) * 2 + 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        return image

    def optimize_policies(self, model, train_loader, val_loader, num_epochs=20):
        """
        Optimize augmentation strategies
        """
        best_policies = self.policies.copy()
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            # Randomly modify policies each round
            self._mutate_policies()

            # Train model
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Apply random augmentation policies
                augmented_data = []
                for i in range(data.size(0)):
                    policy_idx = np.random.randint(len(self.policies))
                    aug_image = self.apply_policy(data[i].numpy(), policy_idx)
                    augmented_data.append(torch.FloatTensor(aug_image))

                data = torch.stack(augmented_data)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, targets)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data, targets in val_loader:
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            accuracy = correct / total

            # Update best policies
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_policies = self.policies.copy()

            print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}')

        # Restore best policies
        self.policies = best_policies
        return best_accuracy

    def _mutate_policies(self):
        """
        Policy mutation
        """
        for policy in self.policies:
            if np.random.random() < 0.2:  # 20% probability of mutation
                operation_index = np.random.randint(len(policy))
                op, prob, magnitude = policy[operation_index]

                # Mutate probability or magnitude
                if np.random.random() < 0.5:
                    prob = np.clip(prob + np.random.uniform(-0.2, 0.2), 0.1, 0.9)
                else:
                    magnitude = np.clip(magnitude + np.random.uniform(-0.2, 0.2), 0.1, 1.0)

                policy[operation_index] = (op, prob, magnitude)
```

#### Generative Adversarial Network (GAN) Augmentation

```python
import torch.nn as nn

class MedicalGAN:
    """
    Medical image generative adversarial network
    """
    def __init__(self, latent_dim=100, image_size=(256, 256)):
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    def _build_generator(self):
        """
        Build generator
        """
        class Generator(nn.Module):
            def __init__(self, latent_dim, channels=1):
                super().__init__()

                self.main = nn.Sequential(
                    # Input: latent_dim -> 4x4x512
                    nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),

                    # 4x4x512 -> 8x8x256
                    nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),

                    # 8x8x256 -> 16x16x128
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),

                    # 16x16x128 -> 32x32x64
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),

                    # 32x32x64 -> 64x64x32
                    nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),

                    # 64x64x32 -> 128x128x16
                    nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(True),

                    # 128x128x16 -> 256x256x1
                    nn.ConvTranspose2d(16, channels, 4, 2, 1, bias=False),
                    nn.Tanh()
                )

            def forward(self, x):
                return self.main(x)

        return Generator(self.latent_dim)

    def _build_discriminator(self):
        """
        Build discriminator
        """
        class Discriminator(nn.Module):
            def __init__(self, channels=1):
                super().__init__()

                self.main = nn.Sequential(
                    # Input: 256x256x1 -> 128x128x16
                    nn.Conv2d(channels, 16, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.3),

                    # 128x128x16 -> 64x64x32
                    nn.Conv2d(16, 32, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.3),

                    # 64x64x32 -> 32x32x64
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.3),

                    # 32x32x64 -> 16x16x128
                    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.3),

                    # 16x16x128 -> 8x8x256
                    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.3),

                    # 8x8x256 -> 4x4x1
                    nn.Conv2d(256, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.main(x)

        return Discriminator()

    def train_gan(self, dataloader, num_epochs=100, lr=0.0002):
        """
        Train GAN
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(device)
        self.discriminator.to(device)

        # Optimizers
        optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        # Loss function
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            for i, (real_images, _) in enumerate(dataloader):
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Labels
                real_labels = torch.ones(batch_size, 1, 4, 4).to(device)
                fake_labels = torch.zeros(batch_size, 1, 4, 4).to(device)

                # Train discriminator
                optimizer_D.zero_grad()

                # Real images
                outputs_real = self.discriminator(real_images)
                d_loss_real = criterion(outputs_real, real_labels)

                # Generated images
                noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(device)
                fake_images = self.generator(noise)
                outputs_fake = self.discriminator(fake_images.detach())
                d_loss_fake = criterion(outputs_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

                # Train generator
                optimizer_G.zero_grad()

                outputs = self.discriminator(fake_images)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizer_G.step()

                if i % 50 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], '
                          f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

    def generate_samples(self, num_samples=10):
        """
        Generate synthetic samples
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, 1, 1)
            if torch.cuda.is_available():
                noise = noise.cuda()

            generated_images = self.generator(noise)
            return generated_images.cpu().numpy()
```


## 🔄 Image Recovery & Reconstruction

### Denoising and Artifact Removal

#### Medical Image Denoising

```python
class MedicalImageDenoising:
    """
    Medical image denoising techniques
    """
    def __init__(self):
        pass

    def traditional_denoising(self, image, method='gaussian'):
        """
        Traditional denoising methods
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)

        elif method == 'median':
            return cv2.medianBlur(image, 5)

        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)

        elif method == 'non_local_means':
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

        else:
            raise ValueError(f"Unknown denoising method: {method}")

    def wavelet_denoising(self, image, wavelet='db4', sigma=0.1):
        """
        Wavelet denoising
        """
        import pywt

        # Multi-level wavelet decomposition
        coeffs = pywt.wavedec2(image, wavelet, level=3)

        # Estimate noise level
        # Use highest frequency wavelet coefficients to estimate noise
        sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745

        # Thresholding
        threshold = sigma_est * np.sqrt(2 * np.log(image.size))

        # Soft threshold
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode='soft')
                           for detail in coeffs_thresh[1:]]

        # Reconstruction
        denoised = pywt.waverec2(coeffs_thresh, wavelet)

        return denoised

class DnCNN(nn.Module):
    """
    DnCNN for medical image denoising
    """
    def __init__(self, channels=1, num_layers=17):
        super().__init__()

        layers = []

        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv (noise removal)
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Residual learning: network learns noise
        noise = self.net(x)
        return x - noise
```

#### Artifact Removal

```python
class MedicalArtifactRemoval:
    """
    Medical image artifact removal
    """
    def __init__(self):
        pass

    def remove_motion_artifacts(self, image):
        """
        Remove motion artifacts (for MRI)
        """
        # Use frequency domain filtering
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Create mask (keep central region)
        mask = np.zeros((rows, cols), np.uint8)
        r, c = np.ogrid[:rows, :cols]
        mask_area = (c - ccol)**2 + (r - crow)**2 <= (min(rows, cols) // 4)**2
        mask[mask_area] = 1

        # Apply mask
        f_shift = f_shift * mask

        # Inverse transform
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return img_back.astype(np.uint8)

    def remove_metal_artifacts(self, image, mask):
        """
        Remove metal artifacts (for CT)
        """
        # Simplified metal artifact removal algorithm
        # 1. Identify metal regions
        metal_mask = self._detect_metal_regions(image)

        # 2. Forward projection
        sino = self._radon_transform(image)

        # 3. Correct projection data
        sino_corrected = self._correct_sino(sino, metal_mask)

        # 4. Back projection reconstruction
        corrected_image = self._iradon_transform(sino_corrected)

        return corrected_image

    def _detect_metal_regions(self, image, threshold=2000):
        """
        Detect metal regions
        """
        # For CT images, high HU values usually indicate metal
        return image > threshold

    def _radon_transform(self, image, theta=None):
        """
        Simplified Radon transform
        """
        if theta is None:
            theta = np.linspace(0., 180., image.shape[0], endpoint=False)

        from skimage.transform import radon
        return radon(image, theta=theta, circle=True)

    def _iradon_transform(self, sinogram, theta=None):
        """
        Simplified inverse Radon transform
        """
        if theta is None:
            theta = np.linspace(0., 180., sinogram.shape[1], endpoint=False)

        from skimage.transform import iradon
        return iradon(sinogram, theta=theta, circle=True)

    def _correct_sino(self, sino, metal_mask):
        """
        Correct sinogram
        """
        # Interpolation method to correct metal influence regions
        sino_corrected = sino.copy()

        # Simplified interpolation correction
        for i in range(sino.shape[0]):
            if np.any(metal_mask):
                # Use linear interpolation to replace outlier values
                sino_corrected[i] = self._linear_interpolation(sino[i], metal_mask)

        return sino_corrected

    def _linear_interpolation(self, data, mask):
        """
        Linear interpolation
        """
        result = data.copy()

        if np.any(mask):
            # Get non-mask point indices
            valid_indices = ~mask
            invalid_indices = mask

            if np.sum(valid_indices) > 1:
                # Linear interpolation
                f = interp1d(np.where(valid_indices)[0],
                           data[valid_indices],
                           kind='linear',
                           bounds_error=False,
                           fill_value='extrapolate')

                result[invalid_indices] = f(np.where(invalid_indices)[0])

        return result
```

### Super-resolution Reconstruction

#### Single Image Super-resolution

```python
class MedicalSuperResolution:
    """
    Medical image super-resolution
    """
    def __init__(self):
        pass

    def traditional_interpolation(self, image, scale_factor=2, method='bicubic'):
        """
        Traditional interpolation methods
        """
        if method == 'bicubic':
            h, w = image.shape
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        elif method == 'bilinear':
            h, w = image.shape
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

class SRCNN(nn.Module):
    """
    Super-resolution convolutional neural network
    """
    def __init__(self, num_channels=1):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Reconstruction
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

class EDSR(nn.Module):
    """
    Enhanced Deep Super-resolution Network
    """
    def __init__(self, num_channels=1, num_features=64, num_blocks=16):
        super().__init__()

        # Head
        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)

        # Body
        self.body = nn.Sequential(*[
            ResBlock(num_features) for _ in range(num_blocks)
        ])

        # Tail
        self.tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Upsampling
        self.upsampler = self._make_upsampler(num_features, scale_factor=2)

        # Final convolution
        self.last = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def _make_upsampler(self, num_features, scale_factor):
        """
        Create upsampling layer
        """
        layers = []
        for _ in range(int(np.log2(scale_factor))):
            layers.extend([
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = self.tail(res)
        x += res
        x = self.upsampler(x)
        x = self.last(x)
        return x

class ResBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self, num_features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.layers(x)
```

#### Multi-scale Super-resolution

```python
class MultiScaleSR:
    """
    Multi-scale super-resolution
    """
    def __init__(self, scales=[2, 4]):
        self.scales = scales
        self.models = self._build_models()

    def _build_models(self):
        """
        Build multi-scale models
        """
        models = {}
        for scale in self.scales:
            if scale == 2:
                models[scale] = SRCNN()
            elif scale == 4:
                models[scale] = EDSR()
            else:
                raise ValueError(f"Unsupported scale factor: {scale}")
        return models

    def enhance(self, image, target_scale):
        """
        Multi-scale image enhancement
        """
        if target_scale in self.models:
            model = self.models[target_scale]
            model.eval()
            with torch.no_grad():
                # Convert to tensor
                if len(image.shape) == 2:
                    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
                else:
                    image_tensor = torch.FloatTensor(image).unsqueeze(0)

                # Forward pass
                enhanced = model(image_tensor)

                # Convert back to numpy
                enhanced = enhanced.squeeze(0).squeeze(0).numpy()

                return enhanced
        else:
            # For unsupported scales, use combination method
            enhanced = image.copy()
            for scale in sorted(self.scales):
                if target_scale % scale == 0:
                    times = target_scale // scale
                    for _ in range(times):
                        enhanced = self.models[scale](torch.FloatTensor(enhanced).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
                    return enhanced

            # Fallback to traditional method
            return cv2.resize(image, (image.shape[1] * target_scale, image.shape[0] * target_scale),
                            interpolation=cv2.INTER_CUBIC)
```


## 📏 Augmentation Effect Evaluation

### Quantitative Evaluation Metrics

#### Image Quality Assessment

```python
class ImageQualityAssessment:
    """
    Image quality assessment
    """
    def __init__(self):
        pass

    def calculate_psnr(self, img1, img2, max_val=255.0):
        """
        Calculate peak signal-to-noise ratio
        """
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))

    def calculate_ssim(self, img1, img2):
        """
        Calculate structural similarity index
        """
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=255)

    def calculate_mae(self, img1, img2):
        """
        Calculate mean absolute error
        """
        return np.mean(np.abs(img1 - img2))

    def evaluate_enhancement(self, original, enhanced, reference=None):
        """
        Evaluate enhancement effect
        """
        metrics = {}

        if reference is not None:
            # Evaluation with reference image
            metrics['PSNR'] = self.calculate_psnr(enhanced, reference)
            metrics['SSIM'] = self.calculate_ssim(enhanced, reference)
            metrics['MAE'] = self.calculate_mae(enhanced, reference)
        else:
            # Evaluation without reference image
            metrics['entropy'] = self._calculate_entropy(enhanced)
            metrics['contrast'] = self._calculate_contrast(enhanced)
            metrics['sharpness'] = self._calculate_sharpness(enhanced)

        return metrics

    def _calculate_entropy(self, image):
        """
        Calculate image entropy
        """
        hist, _ = np.histogram(image, bins=256, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _calculate_contrast(self, image):
        """
        Calculate image contrast
        """
        return np.std(image)

    def _calculate_sharpness(self, image):
        """
        Calculate image sharpness (using Laplacian operator)
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)
```

#### Task-oriented Evaluation

```python
class TaskOrientedEvaluation:
    """
    Task-oriented enhancement effect evaluation
    """
    def __init__(self, segmentation_model=None, classification_model=None):
        self.segmentation_model = segmentation_model
        self.classification_model = classification_model

    def evaluate_segmentation_performance(self, original_images, enhanced_images, ground_truth_masks):
        """
        Evaluate segmentation task performance
        """
        if self.segmentation_model is None:
            raise ValueError("Segmentation model not provided")

        results = {
            'original': [],
            'enhanced': []
        }

        for orig_img, enh_img, gt_mask in zip(original_images, enhanced_images, ground_truth_masks):
            # Original image segmentation
            orig_pred = self.segmentation_model.predict(orig_img)
            orig_metrics = self._calculate_segmentation_metrics(orig_pred, gt_mask)

            # Enhanced image segmentation
            enh_pred = self.segmentation_model.predict(enh_img)
            enh_metrics = self._calculate_segmentation_metrics(enh_pred, gt_mask)

            results['original'].append(orig_metrics)
            results['enhanced'].append(enh_metrics)

        # Calculate average performance improvement
        avg_orig = self._average_metrics(results['original'])
        avg_enh = self._average_metrics(results['enhanced'])

        improvement = {}
        for key in avg_orig.keys():
            improvement[key] = (avg_enh[key] - avg_orig[key]) / avg_orig[key] * 100

        return {
            'original_performance': avg_orig,
            'enhanced_performance': avg_enh,
            'improvement_percentage': improvement
        }

    def evaluate_classification_performance(self, original_images, enhanced_images, labels):
        """
        Evaluate classification task performance
        """
        if self.classification_model is None:
            raise ValueError("Classification model not provided")

        results = {
            'original': [],
            'enhanced': []
        }

        for orig_img, enh_img, label in zip(original_images, enhanced_images, labels):
            # Original image classification
            orig_pred = self.classification_model.predict(orig_img)
            orig_correct = (orig_pred == label)

            # Enhanced image classification
            enh_pred = self.classification_model.predict(enh_img)
            enh_correct = (enh_pred == label)

            results['original'].append(orig_correct)
            results['enhanced'].append(enh_correct)

        orig_accuracy = np.mean(results['original'])
        enh_accuracy = np.mean(results['enhanced'])
        improvement = (enh_accuracy - orig_accuracy) / orig_accuracy * 100

        return {
            'original_accuracy': orig_accuracy,
            'enhanced_accuracy': enh_accuracy,
            'accuracy_improvement': improvement
        }

    def _calculate_segmentation_metrics(self, pred_mask, gt_mask):
        """
        Calculate segmentation metrics
        """
        # Dice coefficient
        intersection = np.sum(pred_mask * gt_mask)
        dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)

        # IoU
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        iou = intersection / (union + 1e-8)

        # Hausdorff distance
        hausdorff = self._calculate_hausdorff_distance(pred_mask, gt_mask)

        return {
            'dice': dice,
            'iou': iou,
            'hausdorff': hausdorff
        }

    def _calculate_hausdorff_distance(self, mask1, mask2):
        """
        Calculate Hausdorff distance
        """
        # Simplified implementation
        points1 = np.column_stack(np.where(mask1 > 0))
        points2 = np.column_stack(np.where(mask2 > 0))

        if len(points1) == 0 or len(points2) == 0:
            return float('inf')

        # Calculate distance matrix
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(points1, points2)

        # Hausdorff distance
        hd1 = np.mean(np.min(dist_matrix, axis=1))
        hd2 = np.mean(np.min(dist_matrix, axis=0))

        return max(hd1, hd2)

    def _average_metrics(self, metrics_list):
        """
        Average metrics
        """
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = np.mean(values)

        return avg_metrics
```


## 🏥 Practical Application Cases

### Data Augmentation Effect Comparison

#### Performance Comparison of Different Augmentation Strategies

```python
def compare_augmentation_strategies(model, train_data, val_data, strategies, num_epochs=10):
    """
    Compare effects of different augmentation strategies
    """
    results = {}

    for strategy_name, augmentation in strategies.items():
        print(f"\nTraining strategy: {strategy_name}")

        # Create augmented data loader
        augmented_train_loader = create_augmented_loader(train_data, augmentation)

        # Train model
        model_copy = copy.deepcopy(model)
        optimizer = optim.Adam(model_copy.parameters(), lr=0.001)

        training_history = []

        for epoch in range(num_epochs):
            model_copy.train()
            train_loss = 0.0

            for batch_idx, (data, targets) in enumerate(augmented_train_loader):
                optimizer.zero_grad()
                output = model_copy(data)
                loss = F.cross_entropy(output, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            val_accuracy = evaluate_model(model_copy, val_data)

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(augmented_train_loader),
                'val_accuracy': val_accuracy
            })

            print(f'Epoch {epoch+1}, Loss: {train_loss/len(augmented_train_loader):.4f}, '
                  f'Val Acc: {val_accuracy:.4f}')

        results[strategy_name] = training_history

    return results

def visualize_augmentation_comparison(results):
    """
    Visualize augmentation strategy comparison results
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Training loss curves
    for strategy, history in results.items():
        epochs = [h['epoch'] for h in history]
        losses = [h['train_loss'] for h in history]
        ax1.plot(epochs, losses, label=strategy, marker='o')

    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Validation accuracy curves
    for strategy, history in results.items():
        epochs = [h['epoch'] for h in history]
        accuracies = [h['val_accuracy'] for h in history]
        ax2.plot(epochs, accuracies, label=strategy, marker='s')

    ax2.set_title('Validation Accuracy Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
```

### Image Recovery Case Analysis

#### Super-resolution Application in Medical Imaging

```python
def super_resolution_case_study(lr_images, hr_images, model):
    """
    Super-resolution case study
    """
    print("Medical Image Super-resolution Case Study")
    print("=" * 50)

    # Evaluate original low-resolution image quality
    print("\n1. Low-resolution image quality evaluation:")
    for i, (lr, hr) in enumerate(zip(lr_images[:3], hr_images[:3])):
        psnr = calculate_psnr(lr, hr)
        ssim = calculate_ssim(lr, hr)
        print(f"Image {i+1}: PSNR = {psnr:.2f}dB, SSIM = {ssim:.4f}")

    # Super-resolution reconstruction
    print("\n2. Super-resolution reconstruction...")
    sr_images = []
    for lr in lr_images:
        sr = model(lr.unsqueeze(0).unsqueeze(0).float())
        sr_images.append(sr.squeeze().numpy())

    # Evaluate super-resolution results
    print("\n3. Super-resolution result quality evaluation:")
    improvements = {'psnr': [], 'ssim': []}

    for i, (lr, sr, hr) in enumerate(zip(lr_images[:3], sr_images[:3], hr_images[:3])):
        # Post-super-resolution quality
        sr_psnr = calculate_psnr(sr, hr)
        sr_ssim = calculate_ssim(sr, hr)

        # Improvement amount
        lr_psnr = calculate_psnr(lr, hr)
        lr_ssim = calculate_ssim(lr, hr)

        psnr_improvement = sr_psnr - lr_psnr
        ssim_improvement = sr_ssim - lr_ssim

        improvements['psnr'].append(psnr_improvement)
        improvements['ssim'].append(ssim_improvement)

        print(f"Image {i+1}:")
        print(f"  Low resolution: PSNR = {lr_psnr:.2f}dB, SSIM = {lr_ssim:.4f}")
        print(f"  Super resolution: PSNR = {sr_psnr:.2f}dB, SSIM = {sr_ssim:.4f}")
        print(f"  Improvement: PSNR +{psnr_improvement:.2f}dB, SSIM +{ssim_improvement:.4f}")

    # Average improvement
    avg_psnr_improvement = np.mean(improvements['psnr'])
    avg_ssim_improvement = np.mean(improvements['ssim'])

    print(f"\n4. Average improvement:")
    print(f"PSNR improvement: +{avg_psnr_improvement:.2f}dB")
    print(f"SSIM improvement: +{avg_ssim_improvement:.4f}")

    return {
        'average_psnr_improvement': avg_psnr_improvement,
        'average_ssim_improvement': avg_ssim_improvement,
        'sr_images': sr_images
    }
```


## 🎯 Core Insights & Future Directions

### 1. Data Augmentation Techniques
- **Basic augmentation**: Geometric transformations, intensity adjustments, preserve anatomical structure
- **Advanced augmentation**: Mixup, CutMix, adversarial augmentation
- **Intelligent augmentation**: AutoAugmentation, GAN generation

### 2. Image Recovery Methods
- **Traditional methods**: Filtering denoising, interpolation enhancement
- **Deep learning**: DnCNN, SRCNN, EDSR
- **Task-oriented**: Optimization based on downstream task performance

### 3. Evaluation Metrics
- **Objective metrics**: PSNR, SSIM, MAE
- **Subjective evaluation**: Physician reading experience
- **Task metrics**: Segmentation/classification accuracy improvement

### 4. Clinical Application Guidelines
- **Modality specificity**: Augmentation strategies for different imaging devices
- **Data compliance**: Privacy-preserving augmentation methods
- **Interpretability**: Interpretability of augmentation processes

### 5. Future Development Directions
- **Adaptive augmentation**: Automatically select best strategies based on image content
- **Cross-modal augmentation**: Use multi-modal information to improve image quality
- **Federated learning augmentation**: Distributed data augmentation and privacy protection


## 🔗 Typical Medical Datasets and Paper URLs Related to This Chapter

:::details

### Datasets

| Dataset | Purpose | Official URL | License | Notes |
| --- | --- | --- | --- | --- |
| **BraTS** | Brain Tumor Multi-sequence MRI Enhancement | https://www.med.upenn.edu/cbica/brats/ | Academic use free | Most authoritative brain tumor dataset |
| **LUNA16** | Lung Nodule Detection CT Enhancement Validation | https://luna16.grand-challenge.org/ | Public | Standard lung nodule dataset |
| **FastMRI** | MRI Fast Reconstruction Dataset | https://fastmri.med.nyu.edu/ | Apache 2.0 | Accelerated MRI reconstruction benchmark dataset |
| **Medical Segmentation Decathlon** | Multi-modality Medical Image Enhancement | https://medicaldecathlon.com/ | CC BY-SA 4.0 | 10 organs' CT/MRI datasets |
| **IXI** | Brain MRI Multi-center Data | https://brain-development.org/ixi-dataset/ | CC BY-SA 3.0 | 600 multi-center brain MRI data |
| **OpenNeuro** | Open Neuroimaging Data | https://openneuro.org/ | CC0 | Contains fMRI, DTI and other neuroimaging data |
| **TCIA** | Cancer Imaging Archive | https://www.cancerimagingarchive.net/ | Public | Contains imaging data for various cancer types |
| **QIN** | Quality Assurance Network Data | https://imagingcommons.cancer.gov/qin/ | Public | Contains various cancer imaging and phenotype data |
| **MIDRC** | COVID-19 Imaging Data | https://midrc.org/ | Public | COVID-19 chest X-ray and CT dataset |

### Papers

| Paper Title | Keywords | Source | Notes |
| --- | --- | --- | --- |
| **Generative Adversarial Networks in Medical Image Augmentation: A review** | Medical GAN augmentation review | [ScienceDirect Computers in Biology and Medicine](https://www.sciencedirect.com/science/article/pii/S0010482522001743) | Comprehensive review of GAN applications in medical image augmentation |
| **A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies With Progress Highlights, and Future Promises** | Deep learning medical enhancement review | [IEEE Explore](https://ieeexplore.ieee.org/document/9363915) | Deep learning review in medical image enhancement |
| **Application of Super-Resolution Convolutional Neural Network for Enhancing Image Resolution in Chest CT** | Super-resolution enhancement | [Springer Journal of Digital Imaging](https://link.springer.com/article/10.1007/s10278-017-0033-z) | Application of SRCNN in medical image super-resolution |
| **Generative adversarial network in medical imaging: A review** | Medical GAN synthesis review | [Medical Image Analysis](https://www.sciencedirect.com/science/article/pii/S1361841518308430) | Latest progress of GAN in medical image synthesis |
| **Learning deconvolutional deep neural network for high resolution medical image reconstruction** | Deconvolution network super-resolution | [Information Sciences](https://www.sciencedirect.com/science/article/pii/S0020025518306273) | Application of deconvolutional network in medical image super-resolution |

### Open Source Libraries

| Library | Function | GitHub/Website | Purpose |
| --- | --- | --- | --- |
| **TorchIO** | Medical Image Transformation Library | https://torchio.readthedocs.io/ | Medical image data augmentation |
| **Albumentations** | Medical Image Augmentation | https://albumentations.ai/ | General image augmentation |
| **SimpleITK** | Medical Image Processing | https://www.simpleitk.org/ | Image processing toolkit |
| **MONAI** | Deep Learning Medical AI | https://monai.io/ | Medical imaging deep learning |
| **ANTsPy** | Image Registration and Analysis | https://github.com/ANTsX/ANTsPy | Advanced image analysis |
| **medpy** | Medical Image Processing | https://github.com/loli/medpy | Medical imaging algorithm library |
| **Catalyst** | Deep Learning Framework | https://github.com/catalyst-team/catalyst | Framework for deep learning research |
| **albumentations** | Fast Image Augmentation | https://github.com/albu/albumentations | Image augmentation library |

:::


::: info 🎯 Chapter Completion
Through this chapter, you have mastered the core technologies of medical image augmentation and recovery. From traditional geometric transformations to advanced generative adversarial networks, from simple filtering denoising to complex deep learning super-resolution, these techniques will help you solve medical imaging data scarcity and quality issues, providing better data foundations for subsequent deep learning models.
:::