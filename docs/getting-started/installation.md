# Installation

## Requirements

- Python 3.10 or higher
- [UV](https://github.com/astral-sh/uv) package manager (recommended) or pip
- 4GB RAM minimum (8GB recommended for ML models)
- Docker (optional, for containerized deployment)

## Installation Methods

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/Dashtid/medtech-ai-security.git
cd medtech-ai-security

# Install with UV
uv sync

# Install with development dependencies
uv sync --extra dev
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/Dashtid/medtech-ai-security.git
cd medtech-ai-security

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Using Docker

```bash
# Build the image
docker build -t medtech-ai-security .

# Run the container
docker run -it medtech-ai-security
```

## Verify Installation

```bash
# Check CLI tools are available
medsec-nvd --help
medsec-sbom --help
medsec-adversarial --help

# Run the test suite
uv run pytest tests/ -v
```

## Optional Dependencies

### For GPU Acceleration

```bash
# Install TensorFlow with GPU support
pip install tensorflow[cuda]
```

### For Live Traffic Capture

```bash
# Requires root/admin privileges
pip install scapy
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first analysis
- [Configuration](configuration.md) - Configure API keys and settings
