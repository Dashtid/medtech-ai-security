# Medical Image Segmentation with U-Net

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![UV Package Manager](https://img.shields.io/badge/package%20manager-UV-orange.svg)](https://github.com/astral-sh/uv)

A production-ready implementation of U-Net architecture for medical image segmentation. This package provides clean, well-documented code for training segmentation models on various medical imaging datasets.

## Project Status

**Currently:** Basic U-Net implementation with data loaders and preprocessing utilities.

**Working:** Model architecture, data loading, preprocessing
**In Progress:** Training scripts, loss functions, evaluation metrics
**Planned:** Advanced U-Net variants (Deep U-Net, Attention U-Net), 3D support

## Features

- **U-Net Architecture**: Classic U-Net implementation with configurable depth and filters
- **Data Loading**: Support for 2D/3D medical images (NIfTI, DICOM, PNG, JPG)
- **Preprocessing**: Normalization, resizing, intensity windowing
- **Dataset Integration**: Easy download scripts for public datasets (MedMNIST, MSD)
- **Modern Python**: Type hints, clean architecture, professional package structure
- **UV Package Manager**: Fast dependency resolution and virtual environment management

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/Dashtid/biomedical-ai.git
cd biomedical-ai

# Install dependencies (includes dev tools)
UV_LINK_MODE=copy uv sync --extra dev
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### 1. Download a Dataset

Start with MedMNIST for quick experimentation:

```bash
# Download PathMNIST (small histopathology dataset)
python scripts/download_data.py --dataset medmnist --task pathmnist --output data/

# Or download Medical Segmentation Decathlon task
python scripts/download_data.py --dataset msd --task liver --output data/
```

See [docs/DATASETS.md](docs/DATASETS.md) for complete dataset information.

### 2. Use the Package

```python
from med_seg.models import UNet
from med_seg.data import MedicalImageLoader, MedicalImagePreprocessor

# Build a U-Net model
model_builder = UNet(
    input_size=256,
    input_channels=1,
    num_classes=1,
    base_filters=64,
    depth=4,
    use_batch_norm=True
)
model = model_builder.build()
model.summary()

# Load data
loader = MedicalImageLoader(
    data_dir="data/images",
    mask_dir="data/masks",
    image_extension=".png"
)

# Preprocess
preprocessor = MedicalImagePreprocessor(
    target_size=(256, 256),
    normalization_method="min-max"
)

images, masks = loader.load_dataset_2d(max_samples=100)
images, masks = preprocessor.preprocess_batch(images, masks)
```

## Available Datasets

### Beginner-Friendly
- **MedMNIST v2**: Lightweight 2D datasets (28×28 images, <1 GB)
  - PathMNIST, ChestMNIST, DermaMNIST, and more
  - Perfect for prototyping and learning

### Production
- **Medical Segmentation Decathlon**: 10 tasks, 2,633 3D volumes
  - Liver, brain, heart, prostate, lung, and more
  - CC-BY-SA license (commercial use allowed)

### Advanced
- **AMOS**: 500 CT + 100 MRI scans, 15 organs (~100 GB)
- **BraTS**: 4,500+ brain tumor MRI scans (~50 GB)

See [docs/DATASETS.md](docs/DATASETS.md) for detailed information and download instructions.

## Architecture

### U-Net Configuration Options

```python
UNet(
    input_size=256,          # Input image size
    input_channels=1,        # 1 for grayscale, 3 for RGB
    num_classes=1,           # 1 for binary, >1 for multi-class
    base_filters=64,         # Number of filters in first layer
    depth=4,                 # Number of downsampling levels
    use_batch_norm=True,     # Batch normalization
    use_dropout=False,       # Dropout for regularization
    dropout_rate=0.5         # Dropout rate if enabled
)
```

### Model Architecture

```
Input (256x256x1)
    ↓
[Encoder Block 1] ──→ Skip Connection 1
    ↓ MaxPool
[Encoder Block 2] ──→ Skip Connection 2
    ↓ MaxPool
[Encoder Block 3] ──→ Skip Connection 3
    ↓ MaxPool
[Encoder Block 4] ──→ Skip Connection 4
    ↓ MaxPool
[Bottleneck]
    ↓ UpSample + Concat Skip 4
[Decoder Block 4]
    ↓ UpSample + Concat Skip 3
[Decoder Block 3]
    ↓ UpSample + Concat Skip 2
[Decoder Block 2]
    ↓ UpSample + Concat Skip 1
[Decoder Block 1]
    ↓
Output (256x256x1)
```

## Project Structure

```
biomedical-ai/
├── src/med_seg/
│   ├── models/           # U-Net architectures
│   │   └── unet.py       # Standard U-Net
│   ├── data/             # Data loading & preprocessing
│   │   ├── loader.py     # Medical image loader
│   │   └── preprocessor.py
│   ├── training/         # Training utilities
│   │   ├── losses.py     # Loss functions (DICE, etc.)
│   │   ├── metrics.py    # Evaluation metrics
│   │   └── trainer.py    # Training orchestration
│   ├── evaluation/       # Evaluation & ensembling
│   └── utils/            # Config, visualization
├── scripts/
│   ├── download_data.py  # Dataset download script
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── inference.py      # Inference script
├── configs/              # YAML configuration files
├── docs/                 # Documentation
│   └── DATASETS.md       # Dataset information
├── tests/                # Unit tests
└── notebooks/            # Jupyter notebooks (examples)
```

## Development

### Run Tests

```bash
UV_LINK_MODE=copy uv run pytest -v
```

### Code Formatting

```bash
UV_LINK_MODE=copy uv run black src/ tests/
UV_LINK_MODE=copy uv run ruff check src/ tests/
```

### Type Checking

```bash
UV_LINK_MODE=copy uv run mypy src/
```

## Roadmap

- [x] Basic U-Net implementation
- [x] Data loading utilities
- [x] Preprocessing pipeline
- [x] Dataset download scripts
- [ ] Training script with DICE loss
- [ ] Evaluation metrics (IoU, precision, recall)
- [ ] Example notebooks
- [ ] Deep U-Net variant
- [ ] Attention U-Net
- [ ] 3D U-Net support
- [ ] Pre-trained weights
- [ ] Web interface (Gradio)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Citation

If you use this code in your research, please cite:

**U-Net Architecture:**
```bibtex
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
```

**Datasets:** See [docs/DATASETS.md](docs/DATASETS.md) for dataset-specific citations.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contact

**David Dashti**
Email: david.dashti@hermesmedical.com
GitHub: [@Dashtid](https://github.com/Dashtid)

---

**Note:** This project is under active development. The API may change as new features are added.
