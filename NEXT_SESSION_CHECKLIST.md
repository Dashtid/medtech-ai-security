# ✅ Next Session Checklist

**Estimated time: 2 hours | Last updated: Now**

Use this checklist to complete your impressive multi-task medical AI project.

---

## 🎯 Goal

Transform your project from "implemented" to "resume-ready" with actual metrics and visualizations.

---

## 📋 Pre-Session Checks

- [ ] **Check baseline training status**
  ```bash
  # See if models/petct_unet_v2 training finished
  tail -n 5 models/petct_unet_v2/training_log.csv
  ```

- [ ] **Verify survival data exists**
  ```bash
  ls data/synthetic_v2_survival/survival_data.json
  # Should show: survival_data.json with 10 patients
  ```

- [ ] **Ensure all scripts are ready**
  ```bash
  ls scripts/train_multitask.py
  ls scripts/evaluate_multitask.py
  ls scripts/inference_with_uncertainty.py
  ls scripts/optimize_model.py
  ```

---

## 🚀 Execution Checklist (Run in Order)

### ☐ Step 1: Train Multi-Task Model (30 min)

```bash
cd C:\Code\biomedical-ai

python scripts/train_multitask.py \
  --data-dir data/synthetic_v2_survival \
  --output models/multitask_unet \
  --epochs 30 \
  --batch-size 8 \
  --seg-weight 0.6 \
  --surv-weight 0.4 \
  --dropout 0.3
```

**Expected:**
- Training starts immediately
- See both segmentation and survival metrics
- Progress bar shows 30 epochs
- Best model saved automatically
- Time: ~30 minutes on CPU

**Watch for:**
- Validation DICE should reach 0.6-0.75
- Survival C-index should reach 0.65-0.80
- Early stopping may trigger if no improvement

**Checkpoint:**
- [ ] Training completed
- [ ] Best model saved: `models/multitask_unet/best_model.keras`
- [ ] Training log exists: `models/multitask_unet/training_log.csv`
- [ ] Note final validation DICE: `_____`
- [ ] Note final C-index: `_____`

---

### ☐ Step 2: Comprehensive Evaluation (10 min)

```bash
python scripts/evaluate_multitask.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --output results/multitask_evaluation \
  --n-mc-samples 30
```

**Expected:**
- Progress bar for evaluation
- Prints metrics summary
- Creates 2 visualization PNGs
- Saves JSON results

**Checkpoint:**
- [ ] Evaluation completed
- [ ] Segmentation metrics printed
  - DICE: `_____`
  - IoU: `_____`
  - Sensitivity: `_____`
- [ ] Survival metrics printed
  - C-index: `_____`
  - Mean uncertainty: `_____`
- [ ] Uncertainty metrics printed
  - ECE: `_____`
  - Correlation: `_____`
- [ ] Files created:
  - [ ] `results/multitask_evaluation/segmentation_metrics.png`
  - [ ] `results/multitask_evaluation/uncertainty_calibration.png`
  - [ ] `results/multitask_evaluation/evaluation_results.json`

---

### ☐ Step 3: Uncertainty Demo (5 min)

```bash
# Run on 3 different patients for portfolio
python scripts/inference_with_uncertainty.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --patient patient_001 \
  --n-samples 30 \
  --output results/uncertainty

python scripts/inference_with_uncertainty.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --patient patient_005 \
  --n-samples 30 \
  --output results/uncertainty

python scripts/inference_with_uncertainty.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --patient patient_010 \
  --n-samples 30 \
  --output results/uncertainty
```

**Expected:**
- Each run takes ~1 minute
- Creates 6-panel visualization
- Shows uncertainty heatmaps
- Displays survival risk distribution

**Checkpoint:**
- [ ] 3 uncertainty visualizations created
- [ ] Files exist in `results/uncertainty/`
- [ ] Pick best one for portfolio: `patient_____`

---

### ☐ Step 4: Model Optimization (15 min)

```bash
python scripts/optimize_model.py \
  --model models/multitask_unet/best_model.keras \
  --data-dir data/synthetic_v2_survival \
  --output models/optimized \
  --quantize \
  --benchmark
```

**Expected:**
- Creates TFLite models
- Runs benchmarks on 50 samples
- Shows speed comparison
- Shows size reduction

**Checkpoint:**
- [ ] Optimization completed
- [ ] TFLite models created:
  - [ ] `models/optimized/model_fp32.tflite`
  - [ ] `models/optimized/model_quantized_int8.tflite`
- [ ] Benchmark results:
  - Original size: `_____ MB`
  - Quantized size: `_____ MB`
  - Size reduction: `_____ %`
  - Original speed: `_____ ms`
  - Quantized speed: `_____ ms`
  - Speedup: `_____ x`
- [ ] File exists: `models/optimized/benchmark_results.json`

---

### ☐ Step 5: Baseline Comparison (10 min)

```bash
python scripts/compare_models.py \
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

**Expected:**
- Evaluates both models
- Creates comparison plots
- Generates text report

**Checkpoint:**
- [ ] Comparison completed
- [ ] Files created:
  - [ ] `results/comparison/training_comparison.png`
  - [ ] `results/comparison/metrics_comparison.png`
  - [ ] `results/comparison/comparison_report.txt`
- [ ] Baseline DICE: `_____`
- [ ] Multi-task DICE: `_____`
- [ ] Performance difference: `_____ %`

---

## 📊 Results Collection

### Fill in Your Metrics

Copy these to your resume:

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

### Resume Bullet (Fill In)

> "Developed production-ready multi-task deep learning system for PET/CT tumor segmentation and survival prediction with Bayesian uncertainty quantification, achieving **[DICE]** DICE score, **[C-INDEX]** C-index for survival prediction, and **[SPEEDUP]x** inference speedup through INT8 quantization while maintaining **[ACCURACY]%** accuracy"

---

## 🎨 Portfolio Assets Created

Collect these for your portfolio:

- [ ] `results/multitask_evaluation/segmentation_metrics.png`
- [ ] `results/multitask_evaluation/uncertainty_calibration.png`
- [ ] `results/uncertainty/patient_XXX_uncertainty.png` (pick best)
- [ ] `results/comparison/metrics_comparison.png`
- [ ] `models/multitask_unet/best_model.keras`
- [ ] `models/optimized/model_quantized_int8.tflite`

---

## 🐛 Troubleshooting

### Training fails
- **Out of memory**: Reduce `--batch-size 4` or `--image-size 128`
- **Slow training**: Expected on CPU, ~30 min is normal
- **Poor metrics**: Try `--seg-weight 0.8 --surv-weight 0.2`

### Evaluation fails
- **Model not found**: Check path, should be `.keras` file
- **Data error**: Verify `survival_data.json` exists
- **MC Dropout slow**: Reduce `--n-mc-samples 20`

### Optimization fails
- **Missing tfmot**: Run `pip install tensorflow-model-optimization`
- **TFLite error**: Try without `--benchmark` flag first
- **Slow benchmark**: Reduce sample size in script

---

## ✍️ Post-Session Tasks

- [ ] **Update README with final metrics**
- [ ] **Create LinkedIn post** with best visualization
- [ ] **Push to GitHub** (public repo)
- [ ] **Add to resume** with filled metrics
- [ ] **Write blog post** (optional)
- [ ] **Record demo video** (optional)

---

## 🎯 Success Criteria

You're done when you have:

✅ Trained multi-task model with both tasks working
✅ Validation DICE >0.65 (good) or >0.70 (excellent)
✅ Survival C-index >0.65 (acceptable) or >0.70 (good)
✅ Uncertainty calibration ECE <0.10
✅ Optimized model with >5x speedup
✅ 3-5 portfolio-ready visualizations
✅ Resume bullet with actual metrics

---

## 🚀 You're Ready!

Everything is implemented. Just run the commands above in order.

**Estimated time: 70 minutes**
**Buffer: 50 minutes for exploration**

**Good luck! 🎉**
