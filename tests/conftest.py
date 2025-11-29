"""Shared pytest fixtures for medtech-ai-security tests."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src directory to path for imports without installation
_src_path = str(Path(__file__).parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


@pytest.fixture
def sample_cve_data():
    """Sample CVE data for testing."""
    return [
        {
            "cve_id": "CVE-2021-44228",
            "description": "Apache Log4j2 remote code execution vulnerability",
            "cvss_score": 10.0,
            "severity": "CRITICAL",
            "published_date": "2021-12-10",
            "cwe_id": "CWE-502",
            "affected_products": ["log4j-core"],
        },
        {
            "cve_id": "CVE-2023-44487",
            "description": "HTTP/2 Rapid Reset Attack",
            "cvss_score": 7.5,
            "severity": "HIGH",
            "published_date": "2023-10-10",
            "cwe_id": "CWE-400",
            "affected_products": ["nginx", "apache"],
        },
        {
            "cve_id": "CVE-2024-1234",
            "description": "Sample medium severity vulnerability",
            "cvss_score": 4.2,
            "severity": "MEDIUM",
            "published_date": "2024-01-15",
            "cwe_id": "CWE-79",
            "affected_products": ["sample-app"],
        },
    ]


@pytest.fixture
def sample_sbom_cyclonedx():
    """Sample CycloneDX SBOM for testing."""
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "name": "test-medical-app",
                "version": "1.0.0",
                "type": "application",
            }
        },
        "components": [
            {
                "name": "express",
                "version": "4.17.1",
                "type": "library",
                "purl": "pkg:npm/express@4.17.1",
            },
            {
                "name": "lodash",
                "version": "4.17.20",
                "type": "library",
                "purl": "pkg:npm/lodash@4.17.20",
            },
        ],
        "dependencies": [
            {"ref": "test-medical-app@1.0.0", "dependsOn": ["express@4.17.1"]},
            {"ref": "express@4.17.1", "dependsOn": ["lodash@4.17.20"]},
        ],
        "vulnerabilities": [
            {
                "id": "CVE-2021-23337",
                "source": {"name": "NVD"},
                "ratings": [{"score": 7.2, "severity": "high"}],
                "affects": [{"ref": "lodash@4.17.20"}],
            }
        ],
    }


@pytest.fixture
def temp_sbom_file(sample_sbom_cyclonedx):
    """Create a temporary SBOM file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(sample_sbom_cyclonedx, f)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_traffic_features():
    """Sample traffic features for anomaly detection testing."""
    np.random.seed(42)
    # Normal traffic features (16 dimensions)
    normal = np.random.normal(loc=0.5, scale=0.1, size=(100, 16))
    normal = np.clip(normal, 0, 1)

    # Attack traffic features (shifted distribution)
    attack = np.random.normal(loc=0.7, scale=0.2, size=(20, 16))
    attack = np.clip(attack, 0, 1)

    return {
        "normal": normal.astype(np.float32),
        "attack": attack.astype(np.float32),
    }


@pytest.fixture
def sample_image_batch():
    """Sample image batch for adversarial testing."""
    np.random.seed(42)
    # Batch of 10 grayscale images (28x28)
    images = np.random.rand(10, 28, 28, 1).astype(np.float32)
    # Binary labels
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    return images, labels


@pytest.fixture
def sample_risk_training_data():
    """Sample training data for risk scoring model."""
    np.random.seed(42)
    n_samples = 100

    # Features: [cvss, exploitability, impact, age_days]
    features = np.column_stack([
        np.random.uniform(0, 10, n_samples),  # CVSS score
        np.random.uniform(0, 4, n_samples),   # Exploitability
        np.random.uniform(0, 6, n_samples),   # Impact
        np.random.randint(0, 365, n_samples), # Age in days
    ])

    # Labels based on CVSS threshold
    labels = np.where(
        features[:, 0] >= 9.0, "critical",
        np.where(
            features[:, 0] >= 7.0, "high",
            np.where(features[:, 0] >= 4.0, "medium", "low")
        )
    )

    return features.astype(np.float32), labels
