# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Unit test suite for all 5 phases

### Changed

- Removed legacy biomedical AI code and data (~1.6GB cleanup)
  - Removed `src/med_seg/` and `src/medical_image_segmentation/` packages
  - Removed legacy training/evaluation scripts (19 files)
  - Removed legacy configs (brain_tumor, kidney, prostate)
  - Removed `requirements.txt` (use pyproject.toml)
  - Removed legacy `medseg-*` CLI entry points from pyproject.toml
  - Archived `docs/PIVOT_PLAN.md` to `docs/archive/`

### Fixed

- Resolved all mypy type checking errors (0 errors in 36 source files)
  - Fixed numpy type annotations with `np.asarray()` and `float()` wrappers
  - Added explicit type annotations for `**kwargs: Any` parameters
  - Fixed `AsyncIterator[None]` return types for FastAPI lifespan functions
  - Added assertions for Optional attribute access
  - Fixed variable shadowing issues causing type conflicts
  - Added mypy override for `requests.*` module in pyproject.toml

## [1.0.0] - 2025-11-30

### Added

- **Phase 5: SBOM Supply Chain Analysis with GNNs**
  - CycloneDX and SPDX SBOM parsing
  - Graph Neural Network (GCN/GAT) for vulnerability propagation
  - Supply chain risk scoring with FDA/EU MDR compliance notes
  - Interactive HTML visualization with D3.js
  - CLI tool: `medsec-sbom`

- **Phase 4: Adversarial ML Testing**
  - FGSM, PGD, and Carlini & Wagner attack implementations
  - Defense methods: Gaussian blur, JPEG compression, feature squeezing, adversarial training
  - Clinical impact assessment for medical AI models
  - CLI tool: `medsec-adversarial`

- **Phase 3: Anomaly Detection for Medical Device Traffic**
  - Synthetic DICOM/HL7 traffic generator with 10 attack types
  - Autoencoder-based anomaly detection (92.5% accuracy)
  - 16-dimensional feature engineering
  - CLI tools: `medsec-traffic-gen`, `medsec-anomaly`

- **Phase 2: ML Vulnerability Risk Scoring**
  - Naive Bayes + KNN ensemble classifier
  - Feature engineering: CVSS scores, CWE domains, device type, clinical impact
  - 75% test accuracy on medical device CVE dataset
  - CLI tool: `medsec-risk`

- **Phase 1: NLP Threat Intelligence**
  - NVD API scraper for medical device CVEs
  - CISA ICS-CERT advisory parser
  - Claude.ai integration for vulnerability enrichment
  - CLI tools: `medsec-nvd`, `medsec-cisa`, `medsec-enrich`

- Comprehensive demo script (`scripts/demo_security.py`)
- Project documentation (README.md, CONTRIBUTING.md)

### Changed

- Strategic pivot from biomedical AI (PET/CT segmentation) to medical device cybersecurity
- Complete project restructure under `src/medtech_ai_security/`

### Removed

- Legacy biomedical AI code (`med_seg` package)
- Obsolete documentation files (15 markdown files from previous project)

## [0.1.0] - 2025-11-01

### Added

- Initial project setup
- Multi-task PET/CT learning system (deprecated)
- U-Net architecture with shared encoder + dual decoders
- Monte Carlo Dropout uncertainty quantification

---

[Unreleased]: https://github.com/Dashtid/medtech-ai-security/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Dashtid/medtech-ai-security/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/Dashtid/medtech-ai-security/releases/tag/v0.1.0
