# Embryo Development Phase Classification
### Deep Learning for Medical Imaging — Custom Ordinal-Focal Loss

---

## Project Overview

This project tackles **automated classification of human embryo development phases** from time-lapse microscopy images. The pipeline classifies embryos across **15 sequential developmental phases** (tPB2 → tHB), addressing two core challenges:

- **Class imbalance** — early blastocyst phases are far rarer than cleavage stages
- **Ordinal structure** — phases follow a biological sequence; a prediction of `t5` when the true class is `t4` is a milder error than predicting `tHB`

A custom **FinalOrdinalLoss** is proposed and evaluated against a standard cross-entropy baseline across four architectures: MobileNetV2, VGG16, VGG19, and InceptionV3.

---

## 📁 Repository Structure

```
├── notebook.ipynb                  # Full training pipeline
├── custom_loss_report.tex          # LaTeX report — mathematical justification
├── custom_loss_report.pdf          # Compiled PDF report
└── README.md
```

---

## Dataset

- **Source**: [Embryo Dataset — Kaggle](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset)
- **Phases**: 16 original developmental phases, merged to **K = 15** classes (`tB` + `tHB` combined)
- **Split**: Embryo-level stratified split — 70% train / 15% val / 15% test (no data leakage across embryos)
- **Sampling**: Every 6th frame sampled per phase annotation

| Split | Size |
|-------|------|
| Train | ~36,000 frames |
| Val   | ~7,700 frames  |
| Test  | ~7,700 frames  |

---

## Custom Loss: FinalOrdinalLoss

The custom loss combines three components:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \left\langle (1 - p_y)^{\gamma} \cdot \mathcal{L}_{\text{ord}} \right\rangle_{\mathcal{B}}$$

| Component | Role |
|-----------|------|
| **Class-weighted CE + label smoothing** (ε = 0.1) | Handles class imbalance; prevents overconfidence |
| **CDF-based ordinal loss** | Penalises phase-distant errors via squared CDF distance |
| **Focal gate** $(1 - p_y)^\gamma$ | Amplifies penalty on hard/uncertain samples; auto-curriculum |

### Why CDF loss?

The predicted and true CDFs are compared element-wise:

```
cdf_pred = cumsum(softmax(logits), dim=1)
cdf_true = (class_indices >= true_label)   # step function at y
ord_loss = mean((cdf_pred - cdf_true)^2, dim=1)
```

This is the **Cramér distance** — a 1-D Wasserstein-family metric. It naturally encodes the ordinal ordering: predicting `t3` when the truth is `t4` incurs a smaller CDF deviation than predicting `tHB`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_ord` | 0.5 | Weight of ordinal term |
| `gamma` | 2.0 | Focal sharpness |
| `label_smoothing` | 0.1 | Smoothing in CE |

```python
loss_fn = FinalOrdinalLoss(
    alpha=class_weights,   # balanced weights from sklearn
    num_classes=15,
    lambda_ord=0.5,
    gamma=2.0
)
```

---

## Models

All models use **ImageNet pretrained weights** with fine-tuned classification heads:

| Model | Head Modification |
|-------|------------------|
| MobileNetV2 | `Dropout(0.3) → Linear(last_channel, 15)` |
| VGG16 | `classifier[6] = Linear(4096, 15)` |
| VGG19 | `classifier[6] = Linear(4096, 15)` |
| InceptionV3 | `fc = Linear(2048, 15)` + aux head replaced |

**Optimizer**: Adam, lr=1e-4, weight_decay=1e-4  
**Grad clipping**: max norm = 1.0  
**Epochs**: 5 per model  
**Batch size**: 32 (16 for Inception)

---

## Results

### Custom Loss (FinalOrdinalLoss) vs Baseline (CE Loss)

| Model | Baseline Acc | Baseline F1 | Custom Acc | Custom F1 | Δ Acc | Δ F1 |
|-------|-------------|-------------|------------|-----------|-------|------|
| **MobileNetV2** | 0.3035 | 0.3177 | **0.3181** | **0.3292** | +1.46% | +1.15% |
| VGG16 | **0.2731** | **0.2785** | 0.2257 | 0.2364 | −4.74% | −4.21% |
| VGG19 | **0.2633** | **0.2624** | 0.2482 | 0.2453 | −1.51% | −1.71% |
| InceptionV3 | **0.3119** | **0.3124** | 0.3062 | 0.3060 | −0.57% | −0.64% |

### Training Loss — Custom Loss Models

| Model | Epoch 0 Loss | Epoch 4 Loss | Val Acc (best) |
|-------|-------------|-------------|----------------|
| MobileNetV2 | 2.4071 | 1.9142 | 0.3155 |
| VGG16 | 2.4320 | 2.0707 | 0.2747 |
| VGG19 | 2.4331 | 2.1270 | 0.2752 |
| InceptionV3 | 3.2674 | 2.2416 | 0.3122 |

### Training Loss — Baseline CE Models

| Model | Epoch 0 Loss | Epoch 4 Loss | Val Acc (best) |
|-------|-------------|-------------|----------------|
| MobileNetV2 | 2.3671 | 1.8926 | 0.2978 |
| VGG16 | 2.3877 | 2.0462 | 0.2671 |
| VGG19 | 2.4156 | 2.1285 | 0.2822 |
| InceptionV3 | 3.2275 | 2.2212 | 0.3497 |

---

## Analysis

**MobileNetV2** was the only model where the custom loss outperformed the baseline — gaining **+1.5% accuracy** and **+1.2% F1**. This suggests the ordinal-focal objective provides the most benefit in lighter, faster-converging architectures where the ordinal curriculum has more room to guide learning.

For **VGG16, VGG19, and InceptionV3**, the custom loss underperforms. Likely reasons:

- **5 epochs is too short** for deeper models to benefit from the ordinal curriculum — the focal gate keeps the ordinal penalty active longer in models that converge more slowly, introducing conflicting gradient signals early
- **InceptionV3's auxiliary head** adds a second CE loss during training that was not integrated with the ordinal term
- VGG architectures, being deeper and more parameter-heavy, may be more sensitive to the additional ordinal gradient noise before the focal gate suppresses it

Overall, all models achieve **~25–35% accuracy on a 15-class problem** (random chance = 6.7%), indicating that the pretrained features transfer meaningfully but more training epochs and learning rate scheduling would substantially improve all results.

---

## 📐 Mathematical Justification

A formal report proving the loss function satisfies all theoretical requirements for a valid training objective is included in `custom_loss_report.pdf` / `.tex`.

**Proved properties:**
1.  **Non-negativity** — $\mathcal{L} \geq 0$ always
2.  **Convergence** — $\mathcal{L} \to 0$ iff predictions are perfectly correct and confident
3.  **Boundedness** — ordinal term $\in [0,1]$; CE bounded by $\max(w_k) \cdot \log K$
4.  **Piecewise differentiability** — valid autograd gradients at every step
5.  **Monotonicity** — $\partial\mathcal{L}/\partial p_y < 0$; minimising loss = maximising correct-class confidence
6.  **Stable convergence** — focal curriculum ensures fast early correction, CE-only minimum at convergence

---

## ⚙️ Setup & Reproduction

### Requirements

```bash
pip install torch torchvision scikit-learn pandas numpy Pillow tqdm
```

### Run

1. Download the dataset from Kaggle and update `DATA_DIR` / `ANN_DIR` in the notebook
2. Open `notebook.ipynb` in Kaggle or Jupyter
3. Run all cells — trains custom loss models, then baselines, then prints comparison

---

## 🛠️ Transforms

```python
# Training
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
transforms.CenterCrop(224)
transforms.Normalize([0.5]*3, [0.5]*3)

# InceptionV3 (299×299 input required)
transforms.Resize(320)
transforms.CenterCrop(299)
```

---

## References

- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
- Niu et al., *Ordinal Regression with Multiple Output CNN*, CVPR 2016
- Cramer, H., *Mathematical Methods of Statistics*, 1946 (Cramér distance)
- PyTorch `nn.CrossEntropyLoss` with `label_smoothing` — [docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

---

## Files

| File | Description |
|------|-------------|
| `notebook.ipynb` | Full pipeline: data loading, training, evaluation, comparison |
| `custom_loss_report.tex` | LaTeX source — mathematical soundness proofs |
| `custom_loss_report.pdf` | Compiled PDF report |

---

*Deep Learning for Medical Imaging — Embryo Phase Classification*
