# Nuclear Medicine AI - PET/CT Tumor Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![UV Package Manager](https://img.shields.io/badge/package%20manager-UV-orange.svg)](https://github.com/astral-sh/uv)

Deep learning for automated tumor segmentation in PET/CT nuclear medicine imaging. Production-ready implementation with 3D U-Net architecture for whole-body tumor detection and segmentation.

## Project Focus

**Nuclear Medicine Applications:**
- FDG-PET/CT whole-body tumor segmentation
- Multi-modal fusion (PET + CT)
- Automated lesion detection for oncology
- Explainable AI for clinical workflows

## Project Status

**Current Focus:** Nuclear medicine PET/CT tumor segmentation

**Completed:**
- 2D U-Net architecture with configurable depth and filters
- Medical image data loader (2D/3D support for NIfTI, DICOM, PNG)
- Training infrastructure (DICE loss, IoU metrics, trainer)
- TCIA dataset integration script (ACRIN-NSCLC-FDG-PET collection)

**In Progress:**
- NBIA Data Retriever integration for automated PET/CT downloads
- 3D U-Net implementation for volumetric tumor segmentation
- NIfTI data loader for 3D PET/CT volumes

**Planned:**
- AutoPET challenge entry with whole-body tumor segmentation
- Explainable AI (Grad-CAM for 3D volumes)
- Uncertainty quantification for clinical decision support

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

### 1. Download PET/CT Dataset

Download ACRIN-NSCLC-FDG-PET dataset from TCIA:

```bash
# Download 10 patients from ACRIN-NSCLC-FDG-PET collection
# Requires NBIA Data Retriever CLI (see Nuclear Medicine Datasets section)
python scripts/download_tcia_pet.py \
    --collection "ACRIN-NSCLC-FDG-PET" \
    --max-patients 10 \
    --output data/tcia
```

See [docs/DATASETS.md](docs/DATASETS.md) for complete dataset information and alternative download methods.

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

## Nuclear Medicine Datasets

### Primary Focus

**ACRIN-NSCLC-FDG-PET** (Currently Integrated)
- 242 patients with non-small cell lung cancer (NSCLC)
- Whole-body FDG-PET/CT scans (PET + CT modalities)
- 4 PET series per patient (~195 slices each, ~7MB per series)
- Publicly accessible via TCIA (The Cancer Imaging Archive)
- Modalities: PET (with/without attenuation correction), CT (multiple series)
- Download via NBIA Data Retriever CLI (integration in progress)

**AutoPET Challenge** (Planned)
- 900 FDG-PET/CT whole-body scans
- Melanoma, lymphoma, and lung cancer patients
- Expert-annotated tumor lesions
- Competition-ready benchmark dataset

### Additional Resources
- **HECKTOR**: Head & neck cancer PET/CT (882 patients)
- **Lung-PET-CT-Dx**: General lung cancer PET/CT imaging
- **Medical Segmentation Decathlon**: Various organ segmentation tasks

### Download PET/CT Data

```bash
# Option 1: Using our TCIA script (requires NBIA Data Retriever CLI)
python scripts/download_tcia_pet.py \
    --collection "ACRIN-NSCLC-FDG-PET" \
    --max-patients 10 \
    --output data/tcia

# Option 2: Direct NBIA Data Retriever (manual)
# 1. Download NBIA Data Retriever from https://wiki.cancerimagingarchive.net/
# 2. Generate manifest file for desired collection
# 3. Use NBIA Data Retriever to download DICOM files
```

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
│   │   └── unet.py       # Standard 2D U-Net
│   ├── data/             # Data loading & preprocessing
│   │   ├── loader.py     # Medical image loader (2D/3D, NIfTI, DICOM)
│   │   └── preprocessor.py
│   ├── training/         # Training utilities
│   │   ├── losses.py     # Loss functions (DICE, etc.)
│   │   ├── metrics.py    # Evaluation metrics
│   │   └── trainer.py    # Training orchestration
│   ├── evaluation/       # Evaluation & ensembling
│   └── utils/            # Config, visualization
├── scripts/
│   ├── download_tcia_pet.py  # TCIA PET/CT download script
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── inference.py          # Inference script
├── examples/
│   └── train_simple.py   # Simple training example
├── configs/              # YAML configuration files
├── docs/                 # Documentation
│   └── DATASETS.md       # Dataset information
├── tests/                # Unit tests
└── data/                 # Downloaded datasets (gitignored)
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
