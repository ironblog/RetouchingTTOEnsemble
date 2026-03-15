# Heterogeneous Test-Time Optimization Ensemble for Photography Retouching Transfer

**Team:** [Team Name]
**Authors:** [Author 1, Author 2, ...]
**Affiliation:** [Affiliation]

---

## 1. Introduction

Photography retouching transfer requires applying an arbitrary editing preset, demonstrated by a single reference pair (before, after), to a new input image. The primary difficulty is generalization: the test set contains presets entirely unseen during development, so methods that memorize preset-specific patterns during training tend to fail.

We observed early on that per-sample test-time optimization (TTO) consistently outperformed pre-trained models on unseen presets. A small network optimized directly on the reference pair captures the specific color and tone transformation without relying on learned priors that may not transfer. Building on this observation, we designed a heterogeneous ensemble of three structurally distinct TTO paradigms, producing decorrelated errors that cancel under averaging.

## 2. Methodology

### 2.1 Core Architecture: Implicit Neural Representation

Our primary component is an implicit neural representation (INR) based on the CNNDWSplitSiren architecture (Kinli et al., 2024). The model has only 11,491 trainable parameters organized into three branches:

- **Position branch:** Maps normalized 2D pixel coordinates through two 1x1 convolutional layers with sine activations to a 32-channel spatial feature.
- **Signal branch:** Maps the input RGB value through an identical structure to a 32-channel color feature.
- **Merge branch:** Concatenates both features (64 channels) and processes them through depthwise-separable convolution blocks to predict a 3-channel residual, added to the input via a global skip connection.

For each test sample, we optimize the INR from scratch on the reference pair using L1 loss for 1000 steps with Adam (lr: 1e-2, cosine decay to 1e-4). To manage memory on high-resolution images, we randomly sample 484 non-overlapping 12x12 windows per step. Gradient clipping at 0.01 and per-branch weight decay stabilize optimization. After fitting, we apply the learned INR to the input image to produce the output.

### 2.2 Two-Stage Bilateral Grid + INR

Our second paradigm decomposes the retouching into global tone mapping and local residual correction:

1. **Bilateral Grid (BG):** We fit a 3D bilateral grid with learnable affine transforms (4x4x12 spatial-luma resolution) to the reference pair. The grid captures spatially-varying global adjustments (exposure, white balance, vignetting) in approximately 2 seconds.
2. **INR Residual:** We apply the fitted BG to the reference "before" image, then train a second INR to map BG(before) to the reference "after." This INR learns only what the BG cannot express (local contrast, nonlinear color shifts).
3. **Application:** The output is INR(BG(input)), combining the global correction of the BG with the residual expressiveness of the INR.

This two-stage approach produces outputs with different error characteristics than the single-stage INR, providing genuine diversity in the ensemble.

### 2.3 Pre-Trained Parametric Filters (DeepLPF)

Our third paradigm uses a small encoder-decoder network trained on the development set to predict parameters of interpretable photographic filters: per-channel tone curves (17 control points each), a linear graduated filter, and a radial vignette filter. At test time, we additionally fine-tune the predicted parameters on the reference pair for 40 steps.

This component contributes a data-driven prior that complements the purely optimization-based INR and BG methods, particularly on presets involving standard photographic adjustments.

### 2.4 Test-Time Augmentation

For each TTO component, we apply geometric test-time augmentation (TTA): we optimize and apply the model on both the original and a flipped version of the reference/input, then average the two outputs. This halves the variance introduced by random window sampling during optimization. We use horizontal flip, vertical flip, or both, depending on the component.

## 3. Ensemble Strategy

Our final submission blends 11 component outputs using a fixed weighted average in BGR color space. The components are designed to maximize error decorrelation through four diversity axes:

| Diversity Source | Variants |
|-----------------|----------|
| **Architecture** | Standard INR (m=2), SIREN-initialized INR (m=1), BG+INR two-stage, DeepLPF |
| **TTA geometry** | Horizontal flip, vertical flip |
| **Random seed** | Seeds 0, 7, 42 (affects window sampling order) |
| **Optimization intensity** | 1000 vs. 1500 steps; 484 vs. 784 windows per step |

The ensemble weights were determined on the development set and held fixed for the test phase. The weighted average cancels uncorrelated high-frequency optimization artifacts across components. Because each component is optimized independently with different random states and architectural biases, their per-pixel errors are approximately uncorrelated, and averaging reduces the noise floor proportionally to the effective number of independent estimates.

## 4. Ablation: Compact Heterogeneous Ensemble

To evaluate the contribution of architectural diversity, we tested a compact 5-component variant on the development set:

| Configuration | Components | Relative Performance |
|---------------|-----------|---------------------|
| Full 11-way | 8 INR + 2 BG+INR + 1 DeepLPF | Baseline |
| Pure INR 3-way | 3 best INR only | Significant drop |
| Heterogeneous 5-way | 3 INR + 1 BG+INR + 1 DeepLPF | Recovers most of the gap |

Removing the BG and DeepLPF components (Pure INR 3-way) causes a notable performance drop despite retaining the three strongest individual components. Restoring one BG+INR and one DeepLPF component (Heterogeneous 5-way) recovers most of the gap, confirming that architectural heterogeneity — not component count — drives ensemble performance. The structurally different error patterns of BG (smooth, spatially-varying) and DeepLPF (parametric, filter-based) complement the INR's high-frequency residual learning.

## 5. Implementation Details

- **Runtime:** Approximately 2.5 hours for the full 11-way ensemble on 2x NVIDIA GPUs (200 test samples).
- **Output format:** PNG (lossless), matching input resolution exactly.
- **Memory:** Peak ~2 GB VRAM per component (single sample at a time).
- **Dependencies:** PyTorch, NumPy, OpenCV. No external model zoos or large pre-trained backbones.
- **Reproducibility:** All TTO components are initialized from scratch per sample. Only the DeepLPF predictor (10% weight) uses pre-trained weights.

## 6. References

1. Kinli, F., Ozcan, B., & Kirac, F. (2024). INRetouch: Context Aware Implicit Neural Representation for Photography Retouching. *arXiv:2412.03848*.
2. Sitzmann, V., Martel, J.N.P., Bergman, A.W., Lindell, D.B., & Wetzstein, G. (2020). Implicit Neural Representations with Periodic Activation Functions. *NeurIPS*.
3. Chen, J., Adams, A., Wadhwa, N., & Hasinoff, S.W. (2016). Bilateral Guided Upsampling. *ACM TOG*.
