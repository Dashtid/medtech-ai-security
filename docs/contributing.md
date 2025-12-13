# Contributing

Thank you for your interest in contributing to MedTech AI Security!

## Quick Links

- [Full Contributing Guide](https://github.com/Dashtid/medtech-ai-security/blob/main/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/Dashtid/medtech-ai-security/blob/main/CODE_OF_CONDUCT.md)
- [Issue Tracker](https://github.com/Dashtid/medtech-ai-security/issues)
- [Pull Requests](https://github.com/Dashtid/medtech-ai-security/pulls)

## Getting Started

### Prerequisites

- Python 3.11 or later
- UV package manager (recommended)
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Dashtid/medtech-ai-security.git
cd medtech-ai-security

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install development dependencies
uv sync --all-extras

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/medtech_ai_security --cov-report=html

# Run specific test file
pytest tests/test_threat_intel.py

# Run tests matching pattern
pytest -k "test_nvd"
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch** for your feature or fix
3. **Make changes** following our coding standards
4. **Write tests** for new functionality
5. **Run the test suite** to ensure nothing breaks
6. **Submit a pull request**

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new SBOM parser for SPDX 2.3
fix: correct CVSS score calculation edge case
docs: update API documentation for threat intel
test: add unit tests for anomaly detector
```

## Code Standards

### Style Guide

- Follow PEP 8 for Python code
- Use type hints for all functions
- Maximum line length: 100 characters
- Use docstrings for public APIs

### Pre-commit Checks

The following checks run automatically:

- **ruff** - Linting and formatting
- **mypy** - Type checking
- **bandit** - Security scanning
- **pytest** - Unit tests

### Documentation

- Update docstrings when modifying functions
- Update relevant documentation files
- Include examples for new features

## Types of Contributions

### Bug Reports

- Use the bug report template
- Include reproduction steps
- Attach relevant logs or screenshots

### Feature Requests

- Use the feature request template
- Describe the use case
- Explain expected behavior

### Pull Requests

- Reference related issues
- Include tests for changes
- Update documentation
- Follow the PR template

## Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. All comments addressed
4. Squash merge to main branch

## Questions?

- Open a [Discussion](https://github.com/Dashtid/medtech-ai-security/discussions)
- Check [existing issues](https://github.com/Dashtid/medtech-ai-security/issues)
