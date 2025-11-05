# Medical Imaging Datasets

This document provides information about publicly available medical imaging datasets for segmentation tasks.

---

## Nuclear Medicine / PET-CT Datasets

### FDG-PET-CT-Lesions (Recommended for PET/CT Segmentation)

**Best for:** Whole-body tumor segmentation, nuclear medicine, multi-modal learning

**Description:** 1,014 whole-body FDG-PET/CT scans with manually annotated tumor lesions from 900 patients with melanoma, lymphoma, and non-small cell lung cancer (NSCLC), plus 513 negative controls.

**What you get:**
- CT volumes (high resolution, ~512×512 × 200-300 slices)
- PET volumes (lower resolution, ~200×200 × 150-200 slices)
- Binary segmentation masks (FDG-avid tumor lesions)
- Preprocessed NIfTI format available (SUV.nii.gz, CTres.nii.gz, SEG.nii.gz)

**Access:**
- Requires TCIA Restricted License Agreement (facial reconstruction concerns)
- Available on TCIA: https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/
- Also part of AutoPET challenge dataset

**Preprocessing:**
- TCIA preprocessing scripts: https://github.com/lab-midas/TCIA_processing
- Converts DICOM → NIfTI with SUV conversion
- Resamples CT to match PET resolution

**Size:** 418.9 GB (full DICOM), smaller for preprocessed NIfTI
**License:** CC BY-NC 4.0 (requires license agreement)
**Citation:** Gatidis S, Kuestner T. (2022). DOI: 10.7937/gkr0-xv29

**Typical Data Characteristics:**
```
Volume shape: 200×200×150 (preprocessed)
Spacing: 2×2×3 mm (PET resolution)
CT range: -1024 to +3071 HU
SUV range: 0-15 (typical tumors: 4-12)
Tumor prevalence: 0.02-0.10% of voxels (highly imbalanced)
```

---

### ACRIN-NSCLC-FDG-PET

**Best for:** Lung cancer, treatment response prediction

**Description:** 242 patients with non-small cell lung cancer (NSCLC) from clinical trial ACRIN 6668. Includes pre- and post-treatment scans for studying treatment response.

**What you get:**
- Multiple PET series per patient (~4 series, 195 slices each)
- Multiple CT series
- Clinical outcome data (survival, treatment response)
- **Note:** No segmentation masks included

**Access:**
- Publicly accessible via TCIA API (no special license required)
- Collection name: "ACRIN-NSCLC-FDG-PET"
- Can be downloaded via tcia-utils Python package

**Limitations:**
- No tumor annotations (would need manual segmentation)
- Best suited for research requiring clinical outcomes
- tcia-utils API download may have reliability issues

**Size:** Variable (per patient ~30-50 MB)
**License:** Public access via TCIA
**Website:** https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/

---

### AutoPET Challenge Dataset

**Best for:** Competition benchmarking, state-of-the-art comparisons

**Description:** Competition dataset based on FDG-PET-CT-Lesions with standardized train/validation/test splits. Multiple versions (AutoPET-I, II, III, IV).

**What you get:**
- Training: 1,014 studies (preprocessed NIfTI)
- Validation: 200 studies
- Test: Hidden (for competition evaluation)
- nnUNet-compatible format

**Access:**
- https://autopet.grand-challenge.org/
- Requires registration for competition
- Training data publicly available after competition

**Preprocessing:**
- Already converted to SUV.nii.gz, CTres.nii.gz, SEG.nii.gz
- Ready for training (no additional preprocessing needed)

**Size:** ~120 GB (preprocessed NIfTI)
**License:** CC BY-NC 4.0

---

### Synthetic PET/CT Data (Development/Testing)

**Best for:** Pipeline development, testing code before downloading real data

**Description:** Script-generated synthetic PET/CT data that mimics real structure with realistic Hounsfield Units (CT) and SUV values (PET).

**Generation:**
```bash
python scripts/create_synthetic_petct.py --output data/synthetic --num-patients 10
```

**What you get:**
- CT.nii.gz: High-resolution CT with organs (lungs, liver, bones)
- CTres.nii.gz: CT resampled to PET resolution
- SUV.nii.gz: PET with 2-4 random tumor lesions per patient
- SEG.nii.gz: Binary tumor masks

**Characteristics:**
- Realistic CT values (-1000 HU for air, 0 HU for soft tissue, 300 HU for bone)
- Realistic SUV values (1-2 background, 5-12 for tumors, 6-8 for brain)
- Tumors: Spherical lesions with Gaussian intensity profiles
- Class imbalance: ~0.02-0.10% tumor voxels (matches real data)

**Use cases:**
- Test data loaders and preprocessing
- Verify visualization tools
- Debug training pipelines
- Develop models before accessing real patient data

**Size:** ~10 MB per patient (compressed NIfTI)
**License:** No restrictions (synthetic data)

---

## Quick Start Datasets

### MedMNIST v2 (Recommended for Beginners)

**Best for:** Quick prototyping, learning, testing pipelines

**Description:** Lightweight 2D medical image classification datasets with preprocessed 28×28 images.

**Download:**
```bash
python scripts/download_data.py --dataset medmnist --task pathmnist --output data/
```

**Available Tasks:**
- `pathmnist`: Pathology images (colon histology)
- `chestmnist`: Chest X-ray images
- `dermamnist`: Dermatoscopy images
- `octmnist`: Optical Coherence Tomography
- `pneumoniamnist`: Pneumonia chest X-rays
- `retinamnist`: Retinal fundus images
- `breastmnist`: Breast ultrasound
- `bloodmnist`: Blood cell microscopy
- `tissuemnist`: Kidney tissue microscopy
- `organamnist`, `organcmnist`, `organsmnist`: 3D organ CT scans

**List all tasks:**
```bash
python scripts/download_data.py --dataset medmnist --list
```

**Size:** Small (few MB per task)
**License:** Creative Commons
**Website:** https://medmnist.com/

---

## Production Datasets

### Medical Segmentation Decathlon (MSD)

**Best for:** Benchmarking, production models, research

**Description:** 10 segmentation tasks covering various organs and modalities (CT, MRI). Contains 2,633 3D medical images with expert annotations.

**Download:**
```bash
# Download specific task
python scripts/download_data.py --dataset msd --task liver --output data/

# List all available tasks
python scripts/download_data.py --dataset msd --list
```

**Available Tasks:**

| Task | Organ | Modality | Train Samples | Test Samples | Size |
|------|-------|----------|---------------|--------------|------|
| `brain` | Brain tumor | Multimodal MRI | 484 | 266 | ~4.5 GB |
| `heart` | Left atrium | MRI | 20 | 10 | ~500 MB |
| `liver` | Liver & tumor | Portal venous CT | 131 | 70 | ~2.5 GB |
| `hippocampus` | Hippocampus | MRI | 263 | 131 | ~1.2 GB |
| `prostate` | Prostate | Multimodal MRI | 32 | 16 | ~800 MB |
| `lung` | Lung tumor | CT | 64 | 32 | ~1.8 GB |
| `pancreas` | Pancreas & tumor | Portal venous CT | 281 | 139 | ~3.2 GB |
| `hepatic` | Hepatic vessels | CT | 303 | 140 | ~3.5 GB |
| `spleen` | Spleen | CT | 41 | 20 | ~900 MB |
| `colon` | Colon cancer | CT | 126 | 64 | ~2.1 GB |

**License:** CC-BY-SA 4.0 (permissive, allows commercial use)
**Website:** http://medicaldecathlon.com/
**Paper:** https://www.nature.com/articles/s41467-022-30695-9

---

## Advanced Datasets

### AMOS (Abdominal Multi-Organ Segmentation)

**Best for:** Multi-organ segmentation, cross-modality learning

**Description:** 500 CT + 100 MRI scans with 15 abdominal organs annotated. Multi-center, multi-vendor data.

**Organs:** Spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, pancreas, right adrenal gland, left adrenal gland, duodenum, bladder, prostate/uterus

**Download:** https://amos22.grand-challenge.org/ (requires registration)

**Size:** ~100 GB
**License:** Research use
**Paper:** https://arxiv.org/abs/2206.08023

---

### BraTS (Brain Tumor Segmentation)

**Best for:** Brain tumor research, state-of-the-art benchmarks

**Description:** Multi-institutional brain tumor MRI dataset with 4,500+ cases (as of 2024). Includes pre-operative multimodal MRI scans.

**Modalities:** T1, T1-contrast, T2, FLAIR

**Download:** http://braintumorsegmentation.org/ (requires registration)

**Size:** Variable by year (10-50 GB)
**License:** Research use, cite competition paper
**Note:** BraTS 2025 dataset not yet released (as of October 2025)

---

## Dataset Benchmarks

### MedSegBench

**Description:** Comprehensive benchmark with 35 medical image segmentation datasets covering ultrasound, dermoscopy, MRI, X-ray, OCT, and more.

**Total:** 60,000+ images across multiple modalities

**Website:** https://www.nature.com/articles/s41597-024-04159-2

---

## Data Format Notes

### 2D Images
- **Supported formats:** PNG, JPG, TIFF
- **Typical use:** Histopathology, X-rays, dermoscopy
- **Preprocessing:** Direct loading with PIL/OpenCV

### 3D Volumes
- **Supported formats:** NIfTI (.nii, .nii.gz), DICOM, MHA/MHD
- **Typical use:** CT, MRI scans
- **Preprocessing:** SimpleITK, nibabel
- **Note:** Can extract 2D slices for 2D models

---

## Recommended Workflow

### 1. Start with MedMNIST
```bash
# Quick test with small dataset
python scripts/download_data.py --dataset medmnist --task pathmnist --output data/

# Train simple model (fast iteration)
python scripts/train.py --config configs/medmnist_example.yaml
```

### 2. Move to MSD
```bash
# Production-quality dataset
python scripts/download_data.py --dataset msd --task liver --output data/

# Train on full task
python scripts/train.py --config configs/msd_liver.yaml
```

### 3. Advanced: AMOS or BraTS
- Register for access
- Download manually
- Use provided data loaders
- Train on high-performance hardware (GPU recommended)

---

## Dataset Statistics Comparison

| Dataset | Images | Modalities | 2D/3D | Size | License | Difficulty |
|---------|--------|------------|-------|------|---------|------------|
| **MedMNIST** | 700K+ | 10+ | 2D | <1 GB | CC | Beginner |
| **MSD** | 2,633 | 2 | 3D | 25 GB | CC-BY-SA | Intermediate |
| **AMOS** | 600 | 2 | 3D | 100 GB | Research | Advanced |
| **BraTS** | 4,500+ | 1 | 3D | 50 GB | Research | Advanced |

---

## Citation Requirements

### Medical Segmentation Decathlon
```bibtex
@article{antonelli2022medical,
  title={The medical segmentation decathlon},
  author={Antonelli, Michela and others},
  journal={Nature communications},
  volume={13},
  number={1},
  pages={4128},
  year={2022}
}
```

### MedMNIST v2
```bibtex
@article{yang2023medmnist,
  title={MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={41},
  year={2023}
}
```

### AMOS
```bibtex
@inproceedings{ji2022amos,
  title={AMOS: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation},
  author={Ji, Yuanfeng and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2022}
}
```

---

## Additional Resources

- **The Cancer Imaging Archive (TCIA):** https://www.cancerimagingarchive.net/
- **Stanford AIMI Datasets:** https://aimi.stanford.edu/shared-datasets
- **Grand Challenge:** https://grand-challenge.org/challenges/
- **OpenNeuro (neuroimaging):** https://openneuro.org/
- **Papers with Code (datasets):** https://paperswithcode.com/datasets?task=medical-image-segmentation

---

## Need Help?

1. **MedMNIST not installing?**
   ```bash
   pip install medmnist
   ```

2. **MSD download failing?**
   - Check internet connection
   - Try manual download from AWS S3
   - Use VPN if region-blocked

3. **Out of disk space?**
   - Start with MedMNIST (small)
   - Download one MSD task at a time
   - Use external drive for large datasets

4. **Need GPU?**
   - MedMNIST: CPU is fine
   - MSD: GPU recommended (NVIDIA GTX 1060+ or better)
   - AMOS/BraTS: GPU required (NVIDIA RTX 2060+ or better)
