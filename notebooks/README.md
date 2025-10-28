# Notebooks

Example Jupyter notebooks for medical image segmentation.

## Available Notebooks

### Coming Soon

- `01_getting_started.ipynb` - Introduction to the package and basic usage
- `02_unet_architecture.ipynb` - Exploring U-Net architecture
- `03_data_loading.ipynb` - Loading and preprocessing medical images
- `04_training_simple_model.ipynb` - Training on MedMNIST
- `05_evaluation.ipynb` - Evaluating model performance

## Setup

Install Jupyter dependencies:

```bash
UV_LINK_MODE=copy uv sync --extra notebooks
```

Start Jupyter Lab:

```bash
UV_LINK_MODE=copy uv run jupyter lab
```

## Note

Jupyter notebooks are gitignored by default. To share a notebook, explicitly add it with:

```bash
git add -f notebooks/example.ipynb
```
