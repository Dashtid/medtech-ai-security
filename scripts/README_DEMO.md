# Multi-Task PET/CT Tumor Analysis Demo

Professional demonstration script for the trained multi-task U-Net model.

## Overview

This demo script provides a portfolio-ready CLI interface for running inference on PET/CT medical imaging data, demonstrating:

- **Tumor Segmentation** with DICE and IoU metrics
- **Survival Risk Prediction** with risk categorization
- **Uncertainty Quantification** via Monte Carlo Dropout
- **Professional Formatting** suitable for screenshots and presentations

## Quick Start

```bash
# Run demo on patient 001 with default settings
UV_LINK_MODE=copy uv run python scripts/demo.py --patient patient_001

# Run with more uncertainty samples for better estimation
UV_LINK_MODE=copy uv run python scripts/demo.py --patient patient_005 --n-samples 50

# Use custom model checkpoint
UV_LINK_MODE=copy uv run python scripts/demo.py \
  --patient patient_010 \
  --model models/multitask_unet/best_model.keras \
  --n-samples 30
```

## Usage

```
python scripts/demo.py [OPTIONS]

Required Arguments:
  --patient PATIENT_ID    Patient ID (e.g., patient_001, patient_005)

Optional Arguments:
  --model PATH           Path to trained model
                         (default: models/multitask_unet/best_model.keras)
  --data-dir DIR         Directory containing patient data
                         (default: data/synthetic_v2_survival)
  --n-samples N          Number of MC Dropout samples for uncertainty
                         (default: 30)
  --threshold FLOAT      Segmentation threshold
                         (default: 0.5)
```

## Example Output

```
======================================================================
  MULTI-TASK PET/CT TUMOR ANALYSIS SYSTEM
  Tumor Segmentation + Survival Prediction + Uncertainty
======================================================================

[Loading Model]
----------------------------------------------------------------------
  Model path: models/multitask_unet/best_model.keras
  [OK] Model loaded successfully
  Total parameters: 31,621,762

[Loading Patient Data]
----------------------------------------------------------------------
  Patient ID: patient_001
  Data directory: data/synthetic_v2_survival
  [OK] Loaded 128 slices
  [OK] Found 45 tumor-containing slices: [42, 43, 44, ..., 86]
  [OK] Using slice 64 for demo

[Running Monte Carlo Dropout Inference]
----------------------------------------------------------------------
  MC samples: 30
  [Processing...]
  [OK] Inference complete

[Tumor Segmentation Results]
----------------------------------------------------------------------
  DICE Score................................ 0.8234
  IoU Score................................. 0.7012
  Ground Truth Tumor Pixels................. 1847 px
  Predicted Tumor Pixels.................... 1923 px
  Mean Uncertainty.......................... 0.0234
  Max Uncertainty........................... 0.1456
  Tumor Region Uncertainty.................. 0.0189

[Survival Prediction Results]
----------------------------------------------------------------------
  Risk Score................................ 0.3421
  Risk Category............................. MODERATE RISK
  Uncertainty (Std Dev)..................... 0.0876
  95% CI Lower Bound........................ 0.1703
  95% CI Upper Bound........................ 0.5139

  Risk Interpretation:
    - Risk score range: [0.1703, 0.5139]
    - Prediction confidence: 92%
    [OK] Acceptable uncertainty level

[Summary]
----------------------------------------------------------------------
  Patient: patient_001
  Slice analyzed: 64
  Segmentation quality: DICE = 0.8234
  Survival risk: MODERATE RISK (score = 0.3421)
  Uncertainty: Segmentation = 0.0234, Survival = 0.0876

======================================================================
  ANALYSIS COMPLETE
======================================================================
```

## Features

### 1. Tumor Segmentation Analysis
- **DICE Score**: Overlap between predicted and ground truth masks
- **IoU Score**: Intersection over union metric
- **Pixel Counts**: Visual assessment of segmentation extent
- **Uncertainty Metrics**: Mean, max, and tumor-region-specific uncertainty

### 2. Survival Risk Prediction
- **Risk Score**: Cox proportional hazards model output
- **Risk Category**: LOW / MODERATE / HIGH risk classification
- **Confidence Interval**: 95% CI for risk score
- **Uncertainty Warning**: Flags high-uncertainty predictions

### 3. Monte Carlo Dropout Uncertainty
- Runs multiple forward passes with dropout active
- Computes mean and standard deviation of predictions
- Provides confidence intervals for predictions
- Identifies unreliable predictions

## Technical Details

- **Model**: Multi-task U-Net with shared encoder
- **Input**: 256x256 PET/CT slices (2 channels)
- **Outputs**: Segmentation mask + survival risk score
- **Uncertainty Method**: MC Dropout (30 samples default)
- **Inference Time**: ~5 seconds for 30 MC samples (CPU)

## Portfolio Use

This script is designed to produce professional output suitable for:
- **Portfolio Screenshots**: Clean, formatted terminal output
- **Presentations**: Clear metrics and risk interpretation
- **Demonstrations**: Fast inference on any patient
- **Documentation**: Shows both segmentation and survival capabilities

## Notes

- Uses middle tumor-containing slice for consistent results
- Automatically finds tumor regions in the data
- Handles uncertainty quantification transparently
- Professional formatting optimized for readability

## Requirements

- Trained multi-task model checkpoint
- Patient data with PET/CT/Segmentation volumes
- Python 3.10+ with TensorFlow 2.13+

## Related Scripts

- `train_multitask.py`: Train the multi-task model
- `evaluate_multitask.py`: Comprehensive evaluation with metrics
- `inference_with_uncertainty.py`: Generate uncertainty visualizations
- `optimize_model.py`: Model quantization and optimization
