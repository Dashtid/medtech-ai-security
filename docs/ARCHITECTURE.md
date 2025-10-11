# Architecture Documentation

## Overview

This document describes the architecture of the medical image segmentation package, including model architectures, data flow, and design patterns.

## Package Structure

```
med_seg/
├── models/          # Neural network architectures
├── data/            # Data loading and preprocessing
├── training/        # Training infrastructure
├── evaluation/      # Evaluation and metrics
└── utils/           # Utility functions
```

## Model Architectures

### 1. U-Net (Standard)

The classic U-Net architecture with 4 encoding and 4 decoding levels.

**Key Features:**
- Spatial dropout for regularization
- Batch normalization
- Skip connections between encoder and decoder
- Sigmoid activation for binary segmentation

**Architecture:**
```
Input (256x256x1)
    |
[Conv-BN-ReLU] x2 (base_filters)
    |
MaxPool + SpatialDropout
    |
[Conv-BN-ReLU] x2 (base_filters*2)
    |
MaxPool + SpatialDropout
    |
[Conv-BN-ReLU] x2 (base_filters*4)
    |
MaxPool + SpatialDropout
    |
[Conv-BN-ReLU] x2 (base_filters*8)
    |
MaxPool
    |
[Bottleneck: Conv-BN-ReLU] x2 (base_filters*16)
    |
[TransposeConv + Concat + Conv] (base_filters*8)  <-- Skip connection
    |
[TransposeConv + Concat + Conv] (base_filters*4)  <-- Skip connection
    |
[TransposeConv + Concat + Conv] (base_filters*2)  <-- Skip connection
    |
[TransposeConv + Concat + Conv] (base_filters)    <-- Skip connection
    |
Conv(1x1) + Sigmoid
    |
Output (256x256x1)
```

**Use Cases:**
- Standard medical image segmentation
- Good balance between performance and computational cost

### 2. Deep U-Net

Extended U-Net with 5 encoding levels for capturing finer details.

**Differences from Standard U-Net:**
- One additional encoding/decoding level
- Bottleneck at 32x base_filters
- Better for larger images or complex structures

**Use Cases:**
- Detailed segmentation tasks
- When computational resources are available

### 3. U-Net with LSTM Decoder

Replaces standard concatenation in decoder with ConvLSTM2D layers.

**Key Innovation:**
- Treats skip connection and upsampled features as temporal sequence
- ConvLSTM2D integrates features across "time" dimension
- Better feature integration in decoder path

**Decoder Block:**
```
Skip Connection (from encoder)
    |
Reshape to (1, H, W, C)
    |                           Upsampled Features
    |                                 |
    +-----> Concatenate (axis=1) <----+
                  |
            ConvLSTM2D
                  |
        [Conv-BN-ReLU] x2
```

**Use Cases:**
- Multi-channel inputs (multi-sequence MRI)
- When feature integration is critical

### 4. Deep U-Net with LSTM and Spatial Dropout

Combines benefits of all approaches:
- Deep architecture (5 levels)
- LSTM-based decoder
- Spatial dropout regularization

**Use Cases:**
- State-of-the-art performance requirements
- Complex segmentation tasks with sufficient data

## Data Flow

### Training Pipeline

```
Raw Medical Images (NIfTI/DICOM)
    |
    v
[MedicalImageLoader]
    - Load images with SimpleITK
    - Apply intensity windowing (for CT)
    - Normalize to [0, 1]
    - Pad/crop to target size
    |
    v
NumPy Arrays (N, H, W, C)
    |
    v
[DataGenerator]
    - Random rotation (±10°)
    - Width/height shift (±10%)
    - Horizontal flip (optional)
    - Synchronized image/mask augmentation
    |
    v
Batches for Training
    |
    v
[Model Training]
    - Forward pass
    - DICE loss computation
    - Backpropagation
    - Metrics calculation
    |
    v
[Callbacks]
    - Model checkpointing
    - Early stopping
    - Learning rate reduction
    - TensorBoard logging
    |
    v
Trained Model
```

### Inference Pipeline

```
New Medical Image
    |
    v
[MedicalImageLoader]
    - Load and preprocess
    - Same preprocessing as training
    |
    v
Single Image Array (H, W, C)
    |
    v
Add Batch Dimension (1, H, W, C)
    |
    v
[Trained Model]
    - Forward pass
    - Probability map output
    |
    v
Remove Batch Dimension (H, W, 1)
    |
    v
[Post-processing]
    - Binarization at threshold
    - Optional CRF/morphological ops
    |
    v
Segmentation Mask
```

### Multi-Expert Ensemble

```
Expert 1 Annotations    Expert 2 Annotations    ...    Expert N Annotations
         |                       |                            |
         v                       v                            v
    Model 1                 Model 2                      Model N
         |                       |                            |
         v                       v                            v
  Predictions 1           Predictions 2                Predictions N
         |                       |                            |
         +-----------------------------------+----------------+
                                 |
                                 v
                    [Ensemble (Mean/Median)]
                                 |
                                 v
                        Ensembled Prediction
                                 |
                                 v
                 [Multi-Threshold Evaluation]
                                 |
                                 v
                     DICE Scores & Metrics
```

## Loss Functions

### 1. DICE Loss (Primary)

```
DICE = (2 * |X ∩ Y|) / (|X| + |Y|)
Loss = 1 - DICE
```

**Advantages:**
- Handles class imbalance well
- Directly optimizes segmentation metric
- Smooth gradients

### 2. Combined Loss

```
Loss = α * DICE_Loss + β * BCE_Loss
```

**Advantages:**
- DICE handles imbalance
- BCE provides stable gradients
- Best of both worlds

### 3. Focal Loss

```
FL = -α(1-p)^γ log(p)
```

**Advantages:**
- Down-weights easy examples
- Focuses on hard examples
- Handles severe imbalance

### 4. Tversky Loss

```
TI = TP / (TP + α*FP + β*FN)
Loss = 1 - TI
```

**Advantages:**
- Adjustable precision/recall trade-off
- α and β control FP/FN penalties

## Design Patterns

### 1. Builder Pattern (Models)

Models are built using builder classes:

```python
builder = UNet(input_size=256, base_filters=16)
model = builder.build()
```

**Benefits:**
- Clear configuration
- Reusable components
- Easy to extend

### 2. Strategy Pattern (Losses)

Loss functions are interchangeable:

```python
trainer = ModelTrainer(loss_function=dice_loss)
# or
trainer = ModelTrainer(loss_function=combined_loss(0.7, 0.3))
```

### 3. Configuration Pattern

YAML-based configuration separates code from parameters:

```yaml
model:
  architecture: "UNetDeep"
  base_filters: 16

training:
  epochs: 500
  learning_rate: 0.0001
```

## Performance Considerations

### Memory Optimization

1. **Clear Sessions:** Use `K.clear_session()` between expert trainings
2. **Batch Size:** Keep at 1 for large images
3. **Mixed Precision:** Consider for GPU memory constraints

### Training Speed

1. **Data Augmentation:** Use generator to augment on-the-fly
2. **Callbacks:** Early stopping prevents unnecessary epochs
3. **Learning Rate Scheduling:** Faster convergence

### Inference Speed

1. **Batch Predictions:** Process multiple images together
2. **Model Optimization:** TensorRT/ONNX conversion
3. **Half Precision:** FP16 inference

## Extension Points

### Adding New Models

1. Create new file in `med_seg/models/`
2. Implement builder class with `build()` method
3. Add to `__init__.py`
4. Update configuration schema

### Adding New Loss Functions

1. Add function to `med_seg/training/losses.py`
2. Follow signature: `(y_true, y_pred) -> scalar`
3. Add unit tests
4. Document in this file

### Adding New Datasets

1. Create YAML configuration in `configs/`
2. Ensure data follows expected structure
3. Adjust preprocessing if needed
4. Run training script with new config

## Best Practices

### Model Training

1. Always use validation set
2. Monitor multiple metrics (DICE, IoU, precision, recall)
3. Save best model based on validation DICE
4. Use early stopping to prevent overfitting
5. Log to TensorBoard for visualization

### Data Preprocessing

1. Normalize images to [0, 1]
2. Apply intensity windowing for CT scans
3. Maintain aspect ratio during resizing
4. Use same preprocessing for train/val/test

### Evaluation

1. Evaluate at multiple thresholds
2. Report ensemble performance
3. Include per-expert metrics
4. Visualize predictions on sample images

---

**Last Updated:** October 11, 2025
**Version:** 1.0
