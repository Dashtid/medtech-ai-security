# Configuration

## Environment Variables

Create a `.env` file in the project root:

```bash
# Claude AI API (for threat intelligence enrichment)
ANTHROPIC_API_KEY=your_api_key_here

# DefectDojo Integration
DEFECTDOJO_URL=https://defectdojo.example.com
DEFECTDOJO_API_KEY=your_api_key_here

# NVD API (optional, increases rate limit)
NVD_API_KEY=your_api_key_here

# Logging
LOG_LEVEL=INFO
```

## API Keys

### Claude AI (Anthropic)

Required for Phase 1 threat intelligence enrichment.

1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Create an API key
3. Set `ANTHROPIC_API_KEY` environment variable

### NVD API

Optional but recommended. Without an API key, NVD limits requests to 5 per 30 seconds.

1. Request a key at [nvd.nist.gov](https://nvd.nist.gov/developers/request-an-api-key)
2. Set `NVD_API_KEY` environment variable

### DefectDojo

Required for vulnerability management integration.

1. Log into your DefectDojo instance
2. Go to API v2 Key under your profile
3. Set `DEFECTDOJO_URL` and `DEFECTDOJO_API_KEY`

## Configuration Files

### pyproject.toml

Project configuration including:

- Dependencies
- Tool settings (black, ruff, mypy)
- Build configuration

### pytest.ini

Test configuration:

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning
```

### .pre-commit-config.yaml

Pre-commit hooks for code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Model Configuration

### Anomaly Detection

```python
from medtech_ai_security.anomaly import AnomalyDetector

detector = AnomalyDetector(
    encoding_dim=16,
    hidden_layers=[64, 32],
    threshold_percentile=95,
    epochs=50
)
```

### SBOM GNN

```python
from medtech_ai_security.sbom_analyzer import VulnerabilityGNN

model = VulnerabilityGNN(
    input_dim=88,
    hidden_dim=64,
    output_dim=3,
    num_heads=4,
    dropout=0.3
)
```

## Kubernetes Configuration

See [Deployment Guide](../deployment/kubernetes.md) for:

- ConfigMaps and Secrets
- Resource limits
- Horizontal Pod Autoscaling
- Network policies
