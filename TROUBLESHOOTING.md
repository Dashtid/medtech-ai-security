# Troubleshooting Guide

**Common issues and solutions for the multi-task PET/CT system**

---

## Training Issues

### Out of Memory (OOM)

**Symptoms**: Training crashes with "Out of memory" error

**Solutions**:
```bash
# Reduce batch size
python scripts/train_multitask.py --batch-size 4  # instead of 8

# Reduce image size
python scripts/train_multitask.py --image-size 128  # instead of 256

# Use mixed precision (if GPU available)
export TF_ENABLE_MIXED_PRECISION=1
```

### Training Too Slow

**Symptoms**: >30 seconds per batch on CPU

**Solutions**:
```bash
# Reduce number of batches (fewer patients)
# Edit data generator to use smaller subset

# Reduce image size
python scripts/train_multitask.py --image-size 128

# Use GPU if available (speeds up 10-20x)
# Install: pip install tensorflow[and-cuda]
```

### Poor Segmentation Metrics

**Symptoms**: DICE < 0.50 after 10+ epochs

**Possible Causes**:
1. **Class weight imbalance too aggressive**
   - Try: `--seg-weight 0.8 --surv-weight 0.2`

2. **Learning rate too high**
   - Try: `--learning-rate 0.00001` (10x smaller)

3. **Data quality issues**
   - Check: tumor-to-background ratio
   - Verify: segmentation masks are non-zero

4. **Loss function parameters**
   - Adjust Focal Tversky: alpha, beta, gamma in `survival_losses.py`

### Poor Survival Metrics

**Symptoms**: C-index < 0.55 (close to random)

**Possible Causes**:
1. **Survival weight too low**
   - Try: `--seg-weight 0.4 --surv-weight 0.6`

2. **Insufficient survival signal**
   - Check: survival_data.json has event variation
   - Verify: some censored, some events occurred

3. **Bottleneck too focused on segmentation**
   - Try: increase dropout `--dropout 0.5`

### Training Stops Early

**Symptoms**: Training stops at epoch 10-15

**Cause**: Early stopping triggered (no improvement for 10 epochs)

**Solutions**:
```bash
# Increase patience
# Edit train_multitask.py: patience=20

# Decrease learning rate
python scripts/train_multitask.py --learning-rate 0.00005

# Try different loss weights
python scripts/train_multitask.py --seg-weight 0.7 --surv-weight 0.3
```

---

## Data Issues

### "No tumor found in patient" Error

**Symptoms**: Data generator fails with ValueError

**Solutions**:
```bash
# Check segmentation files
python -c "
import nibabel as nib
seg = nib.load('data/synthetic_v2_survival/patient_001/SEG.nii.gz')
print('Tumor pixels:', seg.get_fdata().sum())
"

# Regenerate data
python scripts/create_synthetic_petct.py
python scripts/generate_survival_data.py
```

### "Survival data not found" Error

**Symptoms**: FileNotFoundError for survival_data.json

**Solutions**:
```bash
# Regenerate survival data
UV_LINK_MODE=copy uv run python scripts/generate_survival_data.py \
  --input data/synthetic_v2 \
  --output data/synthetic_v2_survival

# Check file exists
ls data/synthetic_v2_survival/survival_data.json
```

### Shape Mismatch Errors

**Symptoms**: ValueError during data loading

**Solutions**:
```bash
# Verify all volumes have same dimensions
python -c "
from pathlib import Path
import nibabel as nib

data_dir = Path('data/synthetic_v2_survival')
for patient_dir in data_dir.glob('patient_*'):
    ct = nib.load(patient_dir / 'CT.nii.gz')
    pet = nib.load(patient_dir / 'SUV.nii.gz')
    seg = nib.load(patient_dir / 'SEG.nii.gz')
    print(f'{patient_dir.name}: CT={ct.shape}, PET={pet.shape}, SEG={seg.shape}')
"
```

---

## Evaluation Issues

### "Model not found" Error

**Symptoms**: FileNotFoundError for best_model.keras

**Solutions**:
```bash
# Check model exists
ls models/multitask_unet/best_model.keras

# If missing, training didn't complete
# Re-run training: python scripts/train_multitask.py ...
```

### Evaluation Produces NaN

**Symptoms**: Metrics show NaN values

**Possible Causes**:
1. **No tumor in validation set**
   - Check: data/synthetic_v2_survival has sufficient patients

2. **Model predicts all zeros**
   - Check: model checkpoint is valid
   - Try: load model and run prediction manually

### MC Dropout Too Slow

**Symptoms**: Evaluation takes >10 minutes

**Solutions**:
```bash
# Reduce MC samples
python scripts/evaluate_multitask.py --n-mc-samples 10  # instead of 30

# Reduce number of test patients
# Edit evaluate_multitask.py to use subset
```

---

## Optimization Issues

### Quantization Fails

**Symptoms**: Error during TFLite conversion

**Solutions**:
```bash
# Try without quantization first
python scripts/optimize_model.py --model ... --output ...
# (remove --quantize flag)

# Check TensorFlow Lite compatibility
python -c "import tensorflow as tf; print(tf.__version__)"
# Should be >= 2.13.0

# Try representative dataset with fewer samples
# Edit optimize_model.py: n_representative=10
```

### Accuracy Drops Significantly

**Symptoms**: Quantized model DICE < 0.50 (original > 0.70)

**Solutions**:
```bash
# Use full precision
python scripts/optimize_model.py --no-quantize

# Try float16 instead of int8
# Edit optimize_model.py: change quantization dtype

# Use larger representative dataset
# Edit optimize_model.py: n_representative=100
```

---

## Demo Script Issues

### "No tumor found" in Demo

**Symptoms**: Demo fails to find tumor slices

**Solutions**:
```bash
# Check patient has tumor
python -c "
import nibabel as nib
seg = nib.load('data/synthetic_v2_survival/patient_001/SEG.nii.gz')
data = seg.get_fdata()
tumor_slices = [i for i in range(data.shape[2]) if data[:,:,i].sum() > 0]
print(f'Tumor in slices: {tumor_slices}')
"

# Try different patient
python scripts/demo.py --patient patient_005
```

### Uncertainty Values Too High

**Symptoms**: Uncertainty > 0.5 (model not confident)

**Possible Causes**:
1. **Model not converged**
   - Train longer or check training curves

2. **Dropout rate too high**
   - Retrain with `--dropout 0.2` instead of 0.3

3. **MC samples too few**
   - Use `--n-samples 50` instead of 30

---

## Environment Issues

### UV Package Manager Errors

**Symptoms**: "uv: command not found"

**Solutions**:
```bash
# Install uv
pip install uv

# Verify installation
uv --version

# Use pip instead
pip install -r requirements.txt
python scripts/train_multitask.py ...  # without UV_LINK_MODE
```

### TensorFlow Import Errors

**Symptoms**: "No module named 'tensorflow'"

**Solutions**:
```bash
# Activate virtual environment
source .venv/Scripts/activate  # Git Bash
.venv\Scripts\activate  # Windows CMD

# Install dependencies
UV_LINK_MODE=copy uv sync

# Or use pip
pip install tensorflow>=2.13.0
```

### SimpleITK Errors

**Symptoms**: "No module named 'SimpleITK'"

**Solutions**:
```bash
# Install SimpleITK
pip install simpleitk>=2.3.0

# Verify installation
python -c "import SimpleITK as sitk; print(sitk.__version__)"
```

---

## Performance Optimization

### Speed Up Training

**Best Practices**:
```bash
# 1. Use GPU (10-20x faster)
pip install tensorflow[and-cuda]

# 2. Reduce image size
python scripts/train_multitask.py --image-size 128

# 3. Use fewer augmentations
# Edit data generator: set augment=False for validation

# 4. Use mixed precision (GPU only)
export TF_ENABLE_MIXED_PRECISION=1
```

### Reduce Memory Usage

**Best Practices**:
```bash
# 1. Reduce batch size
python scripts/train_multitask.py --batch-size 4

# 2. Use gradient accumulation
# Edit train_multitask.py: accumulate gradients over multiple batches

# 3. Clear session between runs
python -c "
from tensorflow import keras
keras.backend.clear_session()
"

# 4. Enable memory growth (GPU only)
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

---

## Debugging Tips

### Check Model Architecture

```python
from tensorflow import keras
model = keras.models.load_model('models/multitask_unet/best_model.keras')
model.summary()
```

### Verify Data Pipeline

```python
from src.med_seg.data.survival_generator import create_survival_generators
from src.med_seg.data.petct_preprocessor import PETCTPreprocessor

preprocessor = PETCTPreprocessor(target_size=(256, 256), normalize=True)
train_gen, val_gen = create_survival_generators(
    'data/synthetic_v2_survival', preprocessor, batch_size=8
)

# Get one batch
inputs, targets = train_gen[0]
print(f'Input shape: {inputs.shape}')
print(f'Seg shape: {targets["segmentation"].shape}')
print(f'Surv shape: {targets["survival"].shape}')
```

### Test Inference

```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model('models/multitask_unet/best_model.keras')

# Random input (batch=1, H=256, W=256, C=2)
dummy_input = np.random.randn(1, 256, 256, 2).astype(np.float32)

# Run inference
seg_pred, surv_pred = model(dummy_input, training=False)
print(f'Seg output: {seg_pred.shape}')
print(f'Surv output: {surv_pred.shape}')
```

---

## Getting Help

### Check Logs

```bash
# Training log
cat models/multitask_unet/training_log.csv

# Last 10 epochs
tail -10 models/multitask_unet/training_log.csv

# Check for errors in output
grep -i "error\|exception" output.log
```

### Validate Installation

```bash
# Check Python version
python --version  # Should be >= 3.10

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Check GPU availability (if applicable)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check all dependencies
python -c "
import tensorflow
import numpy
import scipy
import matplotlib
import SimpleITK
print('All dependencies OK')
"
```

### Report Issues

When reporting issues, include:
1. Python version: `python --version`
2. TensorFlow version: `python -c "import tensorflow as tf; print(tf.__version__)"`
3. Full error traceback
4. Command that caused the error
5. Operating system (Windows/Linux/Mac)
6. GPU availability (if applicable)

---

## Common Workarounds

### Windows Path Issues

```bash
# Use forward slashes in paths
python scripts/train_multitask.py --data-dir data/synthetic_v2_survival

# Or use raw strings in Python
data_dir = r"C:\Users\...\data\synthetic_v2_survival"
```

### Permission Errors

```bash
# Windows: Run as administrator
# Linux/Mac: Use sudo or change ownership
chmod -R 755 models/
chmod -R 755 data/
```

### Disk Space Issues

```bash
# Check disk space
df -h  # Linux/Mac
dir    # Windows

# Clean up old checkpoints
rm -rf models/old_experiment/

# Use smaller datasets
# Edit data generation scripts to create fewer patients
```

---

**Last Updated**: 2025-11-06
**See Also**: README.md, QUICK_REFERENCE.md, PROJECT_SHOWCASE.md
