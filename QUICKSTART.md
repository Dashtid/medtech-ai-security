# Quick Start Guide

## Get Started in 5 Minutes

### 1. Install Dependencies

```bash
cd medical-image-segmentation
uv sync
```

### 2. Prepare Your Data

Place your medical images in this structure:
```
data/
└── brain-growth/
    ├── training/
    ├── validation/
    └── test/
```

### 3. Update Configuration

Edit `configs/brain_growth.yaml`:
```yaml
data:
  train_dir: "data/brain-growth/training"
  val_dir: "data/brain-growth/validation"
  test_dir: "data/brain-growth/test"
```

### 4. Train Your First Model

```bash
# Train for one expert (fastest way to test)
uv run python scripts/train.py --config configs/brain_growth.yaml --expert 1
```

### 5. Monitor Training

```bash
# In a separate terminal
tensorboard --logdir logs/brain-growth
```

Open http://localhost:6006 in your browser.

---

## Common Commands

```bash
# Train all experts
uv run python scripts/train.py --config configs/brain_growth.yaml

# Train specific expert
uv run python scripts/train.py --config configs/kidney.yaml --expert 1

# Use specific GPU
uv run python scripts/train.py --config configs/brain_growth.yaml --gpu 1

# Run tests (when implemented)
uv run pytest

# Format code
uv run black src/

# Type check
uv run mypy src/
```

---

## Troubleshooting

**Issue:** `No images found in data directory`
- **Solution:** Check your data paths in the config file match your actual data location

**Issue:** `Out of memory error`
- **Solution:** Reduce `batch_size` in config or use a smaller model (fewer `base_filters`)

**Issue:** `ImportError for tensorflow`
- **Solution:** Run `uv sync` to ensure all dependencies are installed

---

## Next Steps

1. Check [README.md](README.md) for full documentation
2. Explore example configurations in `configs/`
3. See [TRANSFORMATION_SUMMARY.md](../TRANSFORMATION_SUMMARY.md) for project details
4. Customize model architecture in config files
5. Add your own datasets with new YAML configs

---

[+] Happy Segmenting!
