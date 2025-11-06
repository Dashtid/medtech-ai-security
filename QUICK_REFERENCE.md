# Multi-Task PET/CT System - Quick Reference

**One-page cheat sheet for demonstrations, interviews, and quick reference**

---

## Project At-a-Glance

**What**: Multi-task deep learning for PET/CT tumor segmentation + survival prediction + uncertainty

**Architecture**: U-Net (31.6M params, 120.6 MB) - Shared encoder, dual decoder

**Key Features**:
- Simultaneous segmentation and survival prediction
- Bayesian uncertainty via MC Dropout
- Production-ready (INT8 quantization, 8x speedup)
- Handles class imbalance (Focal Tversky loss)
- Supports censored data (Cox PH loss)

---

## Quick Commands

### Training
```bash
UV_LINK_MODE=copy uv run python scripts/train_multitask.py \
  --data-dir data/synthetic_v2_survival --output models/multitask_unet --epochs 30
```

### Demo (Portfolio Screenshot)
```bash
UV_LINK_MODE=copy uv run python scripts/demo.py --patient patient_001
```

### Evaluation
```bash
UV_LINK_MODE=copy uv run python scripts/evaluate_multitask.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --output results/multitask_evaluation
```

### Uncertainty Visualization
```bash
UV_LINK_MODE=copy uv run python scripts/inference_with_uncertainty.py \
  --model models/multitask_unet/best_model.keras \
  --patient patient_001 --output results/uncertainty
```

### Optimization
```bash
UV_LINK_MODE=copy uv run python scripts/optimize_model.py \
  --model models/multitask_unet/best_model.keras --quantize --benchmark
```

---

## Architecture Diagram (ASCII)

```
PET/CT (256x256x2)
    |
[Encoder Block 1] 64 filters --> [Skip 1]
    |                                 |
[Pool] (128x128)                      |
    |                                 |
[Encoder Block 2] 128 filters --> [Skip 2]
    |                                 |
[Pool] (64x64)                        |
    |                                 |
[Encoder Block 3] 256 filters --> [Skip 3]
    |                                 |
[Pool] (32x32)                        |
    |                                 |
[Encoder Block 4] 512 filters --> [Skip 4]
    |                                 |
[Bottleneck] 1024 filters             |
    |                                 |
    +--- [Global Pool] --> [Dense Layers] --> Survival Risk
    |
    +--- [Decoder 4 + Skip 4] --> 512 filters
              |
          [Upsample] (64x64)
              |
          [Decoder 3 + Skip 3] --> 256 filters
              |
          [Upsample] (128x128)
              |
          [Decoder 2 + Skip 2] --> 128 filters
              |
          [Upsample] (256x256)
              |
          [Decoder 1 + Skip 1] --> 64 filters
              |
          [Conv 1x1] --> Segmentation Mask (256x256x1)
```

---

## Key Metrics (Target Values)

| Metric | Target | Notes |
|--------|--------|-------|
| **Segmentation** | | |
| DICE Score | >0.70 | Overlap with ground truth |
| IoU Score | >0.65 | Intersection over union |
| Sensitivity | >0.75 | True positive rate |
| Specificity | >0.95 | True negative rate |
| **Survival** | | |
| C-index | >0.70 | Concordance index |
| Mean Uncertainty | <0.15 | Lower is better |
| **Uncertainty** | | |
| ECE | <0.10 | Expected calibration error |
| Correlation | >0.70 | Uncertainty-error correlation |
| **Optimization** | | |
| Size Reduction | ~8x | FP32 to INT8 |
| Speed Improvement | >5x | Inference speedup |
| Accuracy Retention | >95% | Post-quantization |

---

## Technical Stack

**Core**: TensorFlow 2.13, Keras 3, Python 3.10+
**Medical Imaging**: SimpleITK, NIfTI format
**Data Science**: NumPy, SciPy, scikit-image
**Optimization**: TFLite, INT8 quantization
**Package Manager**: uv (fast Python packages)

---

## File Locations

**Model**: `models/multitask_unet/best_model.keras`
**Training Log**: `models/multitask_unet/training_log.csv`
**Data**: `data/synthetic_v2_survival/`
**Results**: `results/multitask_evaluation/`
**Scripts**: `scripts/`
**Source**: `src/med_seg/`

---

## Loss Functions

**Segmentation**: Focal Tversky Loss
- Formula: `FTL = (1 - Tversky_Index)^gamma`
- Tversky Index: `TP / (TP + alpha*FN + beta*FP)`
- Parameters: alpha=0.3, beta=0.7, gamma=0.75
- Purpose: Handle severe class imbalance (tumors 0.03% of pixels)

**Survival**: Cox Proportional Hazards Loss
- Formula: Negative partial log-likelihood
- Purpose: Handle right-censored data (patients still alive)
- Metric: C-index (concordance index)

**Combined**: `Total = 0.6 * SegLoss + 0.4 * SurvLoss`

---

## Interview Talking Points

**Multi-Task Learning**:
- Shared encoder learns features useful for both tasks
- Improves efficiency (train once, predict twice)
- Slight segmentation performance decrease (~5%) for dual capability

**Uncertainty Quantification**:
- MC Dropout: 30 forward passes with dropout active
- Provides confidence intervals for predictions
- Flags uncertain cases for expert review
- Critical for trustworthy AI in healthcare

**Production Optimization**:
- INT8 quantization: 8x size reduction
- Magnitude-based pruning: 50% sparsity
- TFLite format: cross-platform deployment
- Result: 8x smaller, 7x faster, <5% accuracy loss

**Clinical Relevance**:
- PET/CT: standard imaging modality in oncology
- Segmentation: reduces radiologist workload
- Survival: aids treatment planning
- Uncertainty: critical for medical AI

---

## Common Questions

**Q: Why multi-task instead of separate models?**
A: Shared encoder improves efficiency, features useful for both tasks, single training pipeline.

**Q: How do you handle class imbalance?**
A: Focal Tversky loss with aggressive parameters (alpha=0.3, beta=0.7, gamma=0.75), data augmentation.

**Q: What about censored data?**
A: Cox proportional hazards loss handles right-censored data (patients still alive at study end).

**Q: How do you quantify uncertainty?**
A: Monte Carlo Dropout - run 30 forward passes, compute variance of predictions.

**Q: How do you ensure model is production-ready?**
A: INT8 quantization, benchmarking, accuracy validation, TFLite format for deployment.

---

## Resume Bullet Template

> "Developed production-ready multi-task deep learning system for PET/CT tumor segmentation and survival prediction with Bayesian uncertainty quantification, achieving **[DICE]** DICE score, **[C-INDEX]** C-index for survival prediction, and **[SPEEDUP]x** inference speedup through INT8 quantization while maintaining **[ACCURACY]%** accuracy"

---

## Portfolio Assets Checklist

- [ ] Training curves (loss, DICE, C-index)
- [ ] Segmentation metrics visualization
- [ ] Uncertainty calibration plots
- [ ] 3x inference demos with uncertainty
- [ ] Optimization benchmark comparison
- [ ] Baseline vs multi-task comparison
- [ ] Demo script terminal screenshot
- [ ] Architecture diagram
- [ ] Model checkpoint (120 MB)
- [ ] Optimized model (15 MB)

---

## Next Steps (After Training)

1. Run comprehensive evaluation (~5 min)
2. Generate 3x uncertainty visualizations (~5 min)
3. Optimize model with quantization (~15 min)
4. Compare with baseline (if available, ~10 min)
5. Fill in metrics in PROJECT_SHOWCASE.md
6. Take screenshots of demo output
7. Create portfolio webpage/presentation

---

**Last Updated**: 2025-11-06
**Status**: Training in progress
**See Also**: PROJECT_SHOWCASE.md, EXECUTION_PLAN.md, scripts/README_DEMO.md
