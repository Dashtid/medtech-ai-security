# MedTech AI Security - AI-Powered Medical Device Cybersecurity

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![UV Package Manager](https://img.shields.io/badge/package%20manager-UV-orange.svg)](https://github.com/astral-sh/uv)

AI and machine learning for medical device cybersecurity, combining NLP, anomaly detection, and adversarial ML to secure healthcare technology. Built on production experience with post-market surveillance scanning and vulnerability intelligence.

## Project Transition Notice

**November 2025 - Strategic Pivot:**

This repository has transitioned from pure biomedical imaging AI to **AI for medical device cybersecurity**. The completed multi-task PET/CT learning system (30 epochs, DICE 0.340, C-index 0.669) serves as the foundation for understanding AI's role in medical technology, but the focus has shifted to where **cybersecurity intersects with medical AI**.

**Why This Direction:**
- Aligns with real-world role as medical device cybersecurity engineer at Hermes Medical Solutions
- Combines domain expertise in medical technology, AI/ML, and cybersecurity
- Addresses critical healthcare security challenges (FDA, EU MDR, ICS-CERT advisories)
- Leverages existing production infrastructure (PMS-scanner, DefectDojo, RAG backend, K3s cluster)
- Unique niche: few practitioners combine medical domain knowledge + AI + security

The biomedical imaging work remains in this repository as a learning foundation and demonstration of multi-task learning, uncertainty quantification, and production ML deployment.

## New Project Focus

**Medical Device Security AI System:**
- NLP-based threat intelligence extraction from CVE/ICS-CERT advisories
- ML-powered vulnerability detection for medical devices
- Anomaly detection for medical device network traffic (DICOM, HL7)
- Adversarial ML attacks on medical AI models and defenses
- SBOM (Software Bill of Materials) analysis with graph neural networks
- Automated compliance checking (FDA, EU MDR, IEC 62304)
- Integration with existing DefectDojo vulnerability management

**Infrastructure Ready:**
- Production K3s cluster on lab server (10.143.31.18)
- Existing PMS-scanner for medical device vulnerability scanning
- RAG backend for document processing (QMS, technical files)
- DefectDojo for vulnerability tracking
- Gitea CI/CD for automated pipelines

See [PIVOT_PLAN.md](PIVOT_PLAN.md) for detailed transition roadmap and technical approach.

## Project Status

**Current Phase:** Foundation established, transitioning to medical device cybersecurity AI (2025-11-06)

**Completed:**
- Multi-task U-Net architecture (shared encoder + dual decoders)
- Survival prediction with Cox proportional hazards loss
- Monte Carlo Dropout uncertainty quantification
- Focal Tversky loss for severe class imbalance
- Production optimization (quantization, pruning, TFLite conversion)
- Comprehensive evaluation framework (segmentation + survival + uncertainty)
- Synthetic PET/CT data generation with realistic survival outcomes
- Model comparison and benchmarking tools
- Real-time training monitoring

**Ready to Train:**
- Multi-task model (segmentation + survival): See [QUICKSTART_MULTITASK.md](QUICKSTART_MULTITASK.md)
- Expected performance: DICE 0.65-0.75, C-index 0.65-0.80
- Estimated training time: 30 minutes (30 epochs)

**Documentation:**
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - One-page cheat sheet for demos and interviews
- [PROJECT_SHOWCASE.md](PROJECT_SHOWCASE.md) - Portfolio-ready technical overview
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) - Step-by-step evaluation workflow
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [scripts/README_DEMO.md](scripts/README_DEMO.md) - Professional demo script guide
- [MULTITASK_GUIDE.md](MULTITASK_GUIDE.md) - Comprehensive implementation guide
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Technical deep dive
- [QUICKSTART_MULTITASK.md](QUICKSTART_MULTITASK.md) - 2-hour quick start guide

## Biomedical AI Foundation (Completed)

The initial multi-task PET/CT learning system serves as a strong foundation demonstrating:

**Training Results (30 epochs completed):**
- Final training DICE: 0.340 (34.0%)
- Final validation DICE: 0.206 (20.6%)
- Training survival C-index: 0.669 (66.9%)
- Model size: 363 MB (31.6M parameters)

**Skills Demonstrated:**
- Multi-task deep learning architecture
- Bayesian uncertainty quantification
- Production model optimization
- Medical imaging preprocessing
- Loss function design for class imbalance
- Survival analysis with Cox proportional hazards

**Portfolio Value:**
- Comprehensive documentation (1,500+ lines)
- Professional demo script with portfolio-ready output
- Complete evaluation framework
- Production-ready quantization pipeline

See sections below for technical details on the U-Net architecture, loss functions, and training pipeline. This work demonstrates the AI/ML foundation now being applied to medical device cybersecurity.

## Features

### Multi-Task Learning
- **Shared Encoder Architecture**: Learn features useful for both segmentation and survival
- **Dual Task Optimization**: Simultaneous tumor segmentation + survival prediction
- **Cox Proportional Hazards**: Standard clinical survival analysis method
- **Concordance Index (C-index)**: Industry-standard survival metric

### Uncertainty Quantification
- **Monte Carlo Dropout**: Bayesian uncertainty via 30 forward passes
- **Calibrated Predictions**: Expected Calibration Error (ECE) < 0.10
- **Clinical Trust**: Flag uncertain predictions for expert review
- **Confidence Intervals**: Mean prediction with standard deviation

### Advanced Loss Functions
- **Focal Tversky Loss**: State-of-the-art for severe class imbalance (alpha=0.3, beta=0.7, gamma=0.75)
- **Dice-Focal Combined**: MICCAI 2020 HECKTOR Challenge winner approach
- **Cox PH Loss**: Negative partial log-likelihood for survival

### Production Ready
- **INT8 Quantization**: 8x size reduction with minimal accuracy loss
- **Weight Pruning**: 50% sparsity for further compression
- **TFLite Conversion**: Deploy on mobile/edge devices
- **Benchmarking Tools**: Comprehensive speed and size comparisons

### Core Capabilities
- **U-Net Architecture**: Configurable depth and filters with batch normalization
- **Data Loading**: Support for 2D/3D medical images (NIfTI, DICOM, PNG, JPG)
- **Preprocessing**: Normalization, resizing, intensity windowing
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

### Option 1: Multi-Task System (Recommended for Resume)

Complete multi-task learning system in 2 hours:

```bash
# 0. Quick Demo (10 seconds - Portfolio Screenshot!)
UV_LINK_MODE=copy uv run python scripts/demo.py --patient patient_001

# 1. Train multi-task model (30 min)
UV_LINK_MODE=copy uv run python scripts/train_multitask.py \
    --data-dir data/synthetic_v2_survival \
    --epochs 30

# 2. Evaluate with uncertainty (5 min)
UV_LINK_MODE=copy uv run python scripts/evaluate_multitask.py \
    --model models/multitask_unet/best_model.keras \
    --data-dir data/synthetic_v2_survival

# 3. Uncertainty visualization (2 min)
UV_LINK_MODE=copy uv run python scripts/inference_with_uncertainty.py \
    --model models/multitask_unet/best_model.keras \
    --data-dir data/synthetic_v2_survival \
    --patient patient_001

# 4. Optimize for deployment (10 min)
UV_LINK_MODE=copy uv run python scripts/optimize_model.py \
    --model models/multitask_unet/best_model.keras \
    --data-dir data/synthetic_v2_survival \
    --quantize \
    --benchmark
```

**New**: Run [scripts/demo.py](scripts/demo.py) for portfolio-ready terminal output showing segmentation metrics, survival risk prediction, and uncertainty quantification in a professional format (perfect for screenshots)

See [QUICKSTART_MULTITASK.md](QUICKSTART_MULTITASK.md) for complete guide.

### Option 2: Basic Segmentation

Use the core U-Net package:

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
│   ├── models/
│   │   ├── unet.py               # Standard 2D U-Net
│   │   └── multitask_unet.py     # Multi-task U-Net (segmentation + survival)
│   ├── data/
│   │   ├── loader.py             # Medical image loader (2D/3D, NIfTI, DICOM)
│   │   ├── preprocessor.py       # Image preprocessing
│   │   └── survival_generator.py # Data generator for multi-task learning
│   ├── training/
│   │   ├── losses.py             # Focal Tversky, DICE, Cox PH loss
│   │   ├── survival_losses.py    # Cox proportional hazards loss
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── trainer.py            # Training orchestration
│   ├── evaluation/               # Evaluation & ensembling
│   └── utils/                    # Config, visualization
├── scripts/
│   ├── train_multitask.py        # Multi-task training (segmentation + survival)
│   ├── evaluate_multitask.py     # Comprehensive evaluation with uncertainty
│   ├── inference_with_uncertainty.py  # MC Dropout inference demo
│   ├── optimize_model.py         # Quantization & pruning
│   ├── generate_survival_data.py # Generate survival outcomes
│   ├── monitor_training.py       # Real-time training visualization
│   ├── compare_models.py         # Model comparison framework
│   └── train_petct_unet.py       # Basic segmentation training
├── docs/
│   ├── QUICKSTART_MULTITASK.md   # 2-hour quick start guide
│   ├── MULTITASK_GUIDE.md        # Comprehensive implementation guide
│   ├── PROJECT_SUMMARY.md        # Technical deep dive
│   ├── NEXT_SESSION_CHECKLIST.md # Step-by-step execution checklist
│   └── DATASETS.md               # Dataset information
├── data/
│   ├── synthetic_v2/             # 10 synthetic PET/CT patients (218 slices)
│   └── synthetic_v2_survival/    # Same patients with survival data
├── tests/                        # Unit tests
└── models/                       # Trained models (gitignored)
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

### Completed
- [x] Basic U-Net implementation
- [x] Data loading utilities
- [x] Preprocessing pipeline
- [x] Multi-task U-Net architecture
- [x] Survival prediction with Cox PH loss
- [x] Monte Carlo Dropout uncertainty
- [x] Focal Tversky loss for class imbalance
- [x] Production optimization (quantization, pruning)
- [x] Comprehensive evaluation framework
- [x] Model comparison and benchmarking
- [x] Synthetic PET/CT data generation

### In Progress
- [ ] Training on real datasets (AutoPET, HECKTOR)
- [ ] 3D U-Net for volumetric segmentation

### Planned
- [ ] Attention mechanisms (Attention U-Net)
- [ ] Explainable AI (Grad-CAM for 3D)
- [ ] Web interface (Gradio/FastAPI)
- [ ] DICOM support for clinical deployment
- [ ] Multi-site federated learning
- [ ] Active learning for annotation

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

**Focal Tversky Loss:**
```bibtex
@inproceedings{abraham2019focal,
  title={A novel focal tversky loss function with improved attention u-net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  booktitle={IEEE ISBI},
  year={2019}
}
```

**Monte Carlo Dropout:**
```bibtex
@inproceedings{gal2016dropout,
  title={Dropout as a bayesian approximation: Representing model uncertainty in deep learning},
  author={Gal, Yarin and Ghahramani, Zoubin},
  booktitle={ICML},
  year={2016}
}
```

**Cox Proportional Hazards:**
```bibtex
@article{cox1972regression,
  title={Regression models and life-tables},
  author={Cox, David R},
  journal={Journal of the Royal Statistical Society: Series B},
  year={1972}
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
