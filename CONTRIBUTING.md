# Contributing to MedTech AI Security

Thank you for your interest in contributing to this AI-powered medical device cybersecurity platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/medtech-ai-security.git
   cd medtech-ai-security
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Dashtid/medtech-ai-security.git
   ```

## Development Setup

### Using UV (Recommended)

```bash
# Install dependencies
uv sync --extra dev

# Install pre-commit hooks (if configured)
uv run pre-commit install
```

### Traditional Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Project Structure

```text
medtech-ai-security/
  src/medtech_ai_security/
    threat_intel/          # Phase 1: NVD/CISA scrapers, Claude enrichment
    ml/                    # Phase 2: Vulnerability risk scoring
    anomaly/               # Phase 3: Traffic anomaly detection
    adversarial/           # Phase 4: Adversarial ML attacks/defenses
    sbom_analysis/         # Phase 5: SBOM GNN analysis
  scripts/                 # Demo and utility scripts
  data/                    # Sample data and outputs
  models/                  # Trained models
  docs/                    # Documentation
  tests/                   # Unit tests
```

## Making Changes

### 1. Create a Branch

Create a descriptive branch name:

```bash
# Feature branch
git checkout -b feature/add-new-attack-method

# Bug fix branch
git checkout -b fix/sbom-parser-edge-case

# Documentation branch
git checkout -b docs/update-threat-intel-usage
```

### 2. Make Your Changes

- Write clean, documented code
- Add type hints to all functions
- Include docstrings (Google style)
- Update tests for new features
- Update documentation

### 3. Follow Code Style

We use several tools to maintain code quality:

- **Black** - Code formatting
- **Ruff** - Fast linting
- **mypy** - Type checking

Run all checks:

```bash
# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/ tests/ --fix

# Type check
uv run mypy src/
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=medtech_ai_security --cov-report=html

# Run specific test file
uv run pytest tests/test_sbom_analysis.py
```

### Running the Demo

```bash
# Comprehensive demo of all 5 phases
uv run python scripts/demo_security.py
```

### CLI Tools

Test individual modules via CLI:

```bash
# Phase 1: Threat Intelligence
medsec-nvd --days 7 --output data/cves.json

# Phase 2: Risk Scoring
medsec-risk train --data data/enriched.json

# Phase 3: Anomaly Detection
medsec-traffic-gen --normal 100 --attack 20

# Phase 4: Adversarial ML
medsec-adversarial evaluate --method fgsm

# Phase 5: SBOM Analysis
medsec-sbom demo --html report.html
```

## Documentation

### Code Documentation

Use Google-style docstrings:

```python
def analyze_sbom(sbom_path: str, output_format: str = "json") -> dict:
    """Analyze an SBOM file for supply chain risks.

    Args:
        sbom_path: Path to the SBOM file (CycloneDX or SPDX)
        output_format: Output format ("json" or "html")

    Returns:
        Risk analysis report as a dictionary

    Raises:
        ValueError: If SBOM format is not supported

    Example:
        >>> report = analyze_sbom("sbom.json", output_format="html")
        >>> print(report["risk_score"])
    """
    ...
```

### Documentation Files

- Update `README.md` for user-facing changes
- Update `docs/THREAT_INTEL_USAGE.md` for threat intel changes
- Add CLI examples to module docstrings

## Submitting Changes

### 1. Commit Your Changes

Use conventional commit messages:

```bash
# Feature
git commit -m "feat: add new SBOM vulnerability detection method"

# Bug fix
git commit -m "fix: correct GNN layer weight initialization"

# Documentation
git commit -m "docs: update CLI examples in README"

# Tests
git commit -m "test: add unit tests for adversarial attacks"

# Refactoring
git commit -m "refactor: simplify traffic generator logic"
```

### 2. Push to Your Fork

```bash
git push origin your-branch-name
```

### 3. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:
   - **Title**: Clear, descriptive title
   - **Description**: What changes were made and why
   - **Related Issues**: Link any related issues
   - **Testing**: How were the changes tested

## Code Style

### Python Style

- Follow PEP 8
- Use type hints (Python 3.10+ syntax)
- Maximum line length: 100 characters
- Use descriptive variable names

### Type Hints

```python
from typing import Optional
import numpy as np

def process_traffic(
    packets: list[dict],
    threshold: float = 0.5,
    labels: Optional[np.ndarray] = None
) -> tuple[np.ndarray, list[str]]:
    """Process network traffic for anomaly detection."""
    ...
```

### Imports

Organize imports in this order:

1. Standard library
2. Third-party packages
3. Local modules

```python
import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from medtech_ai_security.sbom_analysis import SBOMParser
from medtech_ai_security.ml import RiskScorer
```

## Project-Specific Guidelines

### Adding New Attack Methods (Phase 4)

1. Add to `src/medtech_ai_security/adversarial/attacks.py`
2. Implement attack class following existing patterns
3. Add to evaluator integration
4. Create unit tests
5. Update CLI help text

### Adding New SBOM Formats (Phase 5)

1. Add parser in `src/medtech_ai_security/sbom_analysis/parser.py`
2. Implement format detection logic
3. Map to common `Package` and `Dependency` structures
4. Add test SBOM files to `data/`
5. Update documentation

### Adding New Risk Factors

1. Update risk calculation in relevant scorer
2. Document weights and rationale
3. Update compliance notes if FDA/EU MDR related
4. Add unit tests for edge cases

## Questions?

If you have questions:

1. Check existing issues and discussions
2. Read the documentation in `docs/`
3. Open a new issue with the "question" label

## Recognition

Contributors will be acknowledged in:
- `README.md` contributors section
- Release notes
- Project documentation

Thank you for contributing to medical device cybersecurity!
