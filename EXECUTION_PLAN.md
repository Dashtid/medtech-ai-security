# Multi-Task Model Evaluation - Execution Plan

**Status:** Training in progress (Epoch 15/30)
**Estimated completion:** ~10 minutes remaining

## Ready to Execute (When Training Completes)

All scripts verified and result directories created. Execute in this order:

### 0. Quick Demo (OPTIONAL - For Screenshots)
```bash
# Run professional demo on a single patient
UV_LINK_MODE=copy uv run python scripts/demo.py --patient patient_001
```

**Purpose:** Portfolio-ready terminal output showing both segmentation and survival prediction with uncertainty quantification. Perfect for screenshots and demonstrations.

**Expected output:**
- Professional formatted terminal output
- Segmentation metrics (DICE, IoU, uncertainty)
- Survival risk prediction with confidence intervals
- Takes ~10 seconds on CPU

See `scripts/README_DEMO.md` for full documentation.

---

### 1. Comprehensive Evaluation (~5 min)
```bash
UV_LINK_MODE=copy uv run python scripts/evaluate_multitask.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --output results/multitask_evaluation \
  --n-mc-samples 30
```

**Expected outputs:**
- `results/multitask_evaluation/segmentation_metrics.png`
- `results/multitask_evaluation/uncertainty_calibration.png`
- `results/multitask_evaluation/evaluation_results.json`

**Metrics to capture:**
- Segmentation: DICE, IoU, Sensitivity, Specificity
- Survival: C-index, Mean uncertainty
- Calibration: ECE, Uncertainty-error correlation

---

### 2. Uncertainty Inference Demos (~5 min total)
```bash
# Patient 001
UV_LINK_MODE=copy uv run python scripts/inference_with_uncertainty.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --patient patient_001 \
  --n-samples 30 \
  --output results/uncertainty

# Patient 005
UV_LINK_MODE=copy uv run python scripts/inference_with_uncertainty.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --patient patient_005 \
  --n-samples 30 \
  --output results/uncertainty

# Patient 010
UV_LINK_MODE=copy uv run python scripts/inference_with_uncertainty.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --patient patient_010 \
  --n-samples 30 \
  --output results/uncertainty
```

**Expected outputs:** 3 uncertainty visualization PNGs (6-panel layout each)

---

### 3. Model Optimization (~15 min)
```bash
UV_LINK_MODE=copy uv run python scripts/optimize_model.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --output models/optimized \
  --quantize \
  --benchmark
```

**Expected outputs:**
- `models/optimized/model_fp32.tflite`
- `models/optimized/model_quantized_int8.tflite`
- `models/optimized/benchmark_results.json`

**Metrics to capture:**
- Original size (MB)
- Quantized size (MB)
- Size reduction (%)
- Original speed (ms)
- Quantized speed (ms)
- Speedup (x)

---

### 4. Baseline Comparison (~10 min)
```bash
# Only if baseline model exists
UV_LINK_MODE=copy uv run python scripts/compare_models.py \
  --models \
    models/petct_unet_v2/best_model.keras \
    models/multitask_unet/best_model.keras \
  --logs \
    models/petct_unet_v2/training_log.csv \
    models/multitask_unet/training_log.csv \
  --labels "Baseline (Seg Only)" "Multi-Task (Seg+Surv)" \
  --data-dir data/synthetic_v2 \
  --output results/comparison
```

**Expected outputs:**
- `results/comparison/training_comparison.png`
- `results/comparison/metrics_comparison.png`
- `results/comparison/comparison_report.txt`

---

## Resume Metrics Template

Fill in after all evaluations complete:

```
Segmentation Performance:
- DICE Score: _______
- IoU Score: _______
- Sensitivity: _______
- Specificity: _______

Survival Prediction:
- C-index: _______
- Mean risk uncertainty: _______

Uncertainty Quantification:
- Expected Calibration Error: _______
- Uncertainty-error correlation: _______

Model Optimization:
- Size reduction: _______ x
- Speed improvement: _______ x
- Accuracy retention: _______ %
```

### Resume Bullet (Final)

> "Developed production-ready multi-task deep learning system for PET/CT tumor segmentation and survival prediction with Bayesian uncertainty quantification, achieving **[DICE]** DICE score, **[C-INDEX]** C-index for survival prediction, and **[SPEEDUP]x** inference speedup through INT8 quantization while maintaining **[ACCURACY]%** accuracy"

---

## Portfolio Assets

Collect these files for portfolio:
- `results/multitask_evaluation/segmentation_metrics.png`
- `results/multitask_evaluation/uncertainty_calibration.png`
- Best uncertainty demo: `results/uncertainty/patient_XXX_uncertainty.png`
- `results/comparison/metrics_comparison.png` (if baseline exists)
- `models/multitask_unet/best_model.keras`
- `models/optimized/model_quantized_int8.tflite`

---

**Total execution time:** ~35 minutes
**Portfolio assets:** 4-6 visualizations + 2 models
