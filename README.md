# Medical Image Segmentation with U-Net Architectures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional implementation of U-Net architectures for medical image segmentation with multi-expert ensemble support, developed for the QUBIQ Challenge.

## Overview

This project implements state-of-the-art deep learning models for medical image segmentation, specifically designed to handle multi-expert annotations and quantify inter-rater variability. The implementation includes multiple U-Net variants with advanced features such as spatial dropout, LSTM-enhanced decoders, and deep architectures.

**Key Features:**
- Multiple U-Net architecture variants (standard, deep, LSTM-enhanced)
- Multi-expert ensemble evaluation methodology
- Comprehensive data augmentation pipeline
- DICE loss with additional metrics (IoU, precision, recall)
- Configurable training via YAML files
- Professional package structure with UV dependency management
- Extensive documentation and type hints

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-image-segmentation.git
cd medical-image-segmentation

# Install dependencies
uv sync
```

## Quick Start

```bash
# Train models for all experts
python scripts/train.py --config configs/brain_growth.yaml

# Train for specific expert
python scripts/train.py --config configs/kidney.yaml --expert 1
```

## Results

Performance on QUBIQ Challenge datasets:

| Dataset | DICE Score | Top Leaderboard |
|---------|------------|----------------|
| Brain Growth | 0.9034 | 0.5548 |
| Kidney | 0.9181 | 0.8532 |
| **Average** | **0.864** | **0.7778** |

## Architecture Variants

1. **Standard U-Net** - Classic architecture with spatial dropout
2. **Deep U-Net** - Extended with 5 encoding levels
3. **U-Net with LSTM** - ConvLSTM2D-enhanced decoder
4. **Deep U-Net with LSTM** - Combines depth and LSTM

## License

MIT License - See LICENSE file for details.

## Contact

David Dashti - david.dashti@hermesmedical.com