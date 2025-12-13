# Changelog

All notable changes to MedTech AI Security will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MkDocs Material documentation site
- OpenSSF Scorecard security workflow
- Container security scanning with Trivy
- Semgrep SAST integration
- CITATION.cff for academic references
- CODEOWNERS for automatic review assignment
- Comprehensive API documentation

### Changed
- Updated Mermaid diagrams in ARCHITECTURE.md
- Enhanced pre-commit hook configuration
- Improved test coverage configuration

### Fixed
- Suppressed third-party library warnings in pytest
- Resolved Keras/NumPy 2.0 compatibility warnings

## [0.2.0] - 2025-12-13

### Added
- Phase 5: SBOM Analysis module with GNN-based risk scoring
- CycloneDX and SPDX SBOM format support
- DefectDojo integration for vulnerability management
- Graph visualization for dependency analysis
- FDA 510(k) export format

### Changed
- Refactored CLI to use click groups
- Improved error handling across all modules
- Enhanced documentation structure

### Security
- Added dependency vulnerability scanning
- Implemented license compliance checking

## [0.1.0] - 2025-11-15

### Added
- Phase 1: Threat Intelligence module
  - NVD API integration for CVE collection
  - CISA ICS-CERT advisory parser
  - Claude AI-powered vulnerability enrichment
  - Medical device keyword filtering

- Phase 2: ML Risk Scoring module
  - Ensemble model (XGBoost, RandomForest, LightGBM)
  - Feature engineering for medical device context
  - CVSS-based priority scoring

- Phase 3: Anomaly Detection module
  - Autoencoder-based traffic analysis
  - Protocol-aware detection (DICOM, HL7, FHIR)
  - Real-time monitoring support

- Phase 4: Adversarial ML module
  - FGSM and PGD attack implementations
  - C&W optimization-based attack
  - JPEG compression and Gaussian blur defenses
  - Robustness evaluation framework

- Initial CLI tools for all modules
- REST API with FastAPI
- Docker containerization
- Kubernetes deployment manifests
- Comprehensive test suite (672 tests)

### Documentation
- README with quick start guide
- ARCHITECTURE.md with system design
- CONTRIBUTING.md with guidelines
- SECURITY.md with vulnerability reporting

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.2.0 | 2025-12-13 | SBOM Analysis, DefectDojo integration |
| 0.1.0 | 2025-11-15 | Initial release with 4 core modules |

## Upgrade Notes

### 0.1.x to 0.2.0

- New dependency: `torch-geometric` for GNN models
- Configuration change: SBOM settings added to config.yaml
- CLI change: New `medsec-sbom` command group

```bash
# Update dependencies
uv sync

# Run database migrations (if applicable)
medsec-db migrate

# Verify installation
medsec-sbom --version
```

## Links

- [Full Changelog on GitHub](https://github.com/Dashtid/medtech-ai-security/blob/main/CHANGELOG.md)
- [Release Notes](https://github.com/Dashtid/medtech-ai-security/releases)
- [Upgrade Guide](https://github.com/Dashtid/medtech-ai-security/blob/main/docs/upgrading.md)
