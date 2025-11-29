"""Unit tests for Phase 2: ML Vulnerability Risk Scoring."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from medtech_ai_security.ml.risk_scorer import (
    CWE_DOMAINS,
    KNNClassifier,
    NaiveBayesClassifier,
    RiskPrediction,
    VulnerabilityRiskScorer,
)


class TestRiskPrediction:
    """Test RiskPrediction dataclass."""

    def test_risk_prediction_creation(self):
        """Test creating a RiskPrediction."""
        prediction = RiskPrediction(
            cve_id="CVE-2024-1234",
            risk_score=85.5,
            priority="CRITICAL",
            confidence=0.92,
        )

        assert prediction.cve_id == "CVE-2024-1234"
        assert prediction.risk_score == 85.5
        assert prediction.priority == "CRITICAL"
        assert prediction.confidence == 0.92

    def test_risk_prediction_to_dict(self):
        """Test converting RiskPrediction to dictionary."""
        prediction = RiskPrediction(
            cve_id="CVE-2024-1234",
            risk_score=85.567,
            priority="HIGH",
            confidence=0.8765,
            contributing_factors={"cvss_score": 0.5},
            recommendation="Patch immediately",
        )

        result = prediction.to_dict()

        assert result["cve_id"] == "CVE-2024-1234"
        assert result["risk_score"] == 85.57  # Rounded to 2 decimals
        assert result["confidence"] == 0.877  # Rounded to 3 decimals
        assert result["contributing_factors"]["cvss_score"] == 0.5


class TestNaiveBayesClassifier:
    """Test custom Naive Bayes classifier."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        # Class 0: centered at (0, 0)
        X0 = np.random.randn(50, 2) + np.array([0, 0])
        # Class 1: centered at (3, 3)
        X1 = np.random.randn(50, 2) + np.array([3, 3])

        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        return X, y

    def test_naive_bayes_fit(self, sample_data):
        """Test fitting Naive Bayes classifier."""
        X, y = sample_data
        clf = NaiveBayesClassifier()

        result = clf.fit(X, y)

        assert result is clf  # Returns self
        assert len(clf.classes) == 2
        assert 0 in clf.class_priors
        assert 1 in clf.class_priors

    def test_naive_bayes_predict(self, sample_data):
        """Test Naive Bayes prediction."""
        X, y = sample_data
        clf = NaiveBayesClassifier().fit(X, y)

        # Test prediction on training data
        predictions = clf.predict(X)

        assert predictions.shape == y.shape
        assert set(predictions).issubset({0, 1})

    def test_naive_bayes_predict_proba(self, sample_data):
        """Test probability predictions."""
        X, y = sample_data
        clf = NaiveBayesClassifier().fit(X, y)

        probs = clf.predict_proba(X)

        assert probs.shape == (100, 2)
        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)
        # Probabilities should be between 0 and 1
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_naive_bayes_score(self, sample_data):
        """Test accuracy scoring."""
        X, y = sample_data
        clf = NaiveBayesClassifier().fit(X, y)

        score = clf.score(X, y)

        # Should have reasonable accuracy on separable data
        assert 0.8 <= score <= 1.0


class TestKNNClassifier:
    """Test custom KNN classifier."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        X0 = np.random.randn(30, 2) + np.array([0, 0])
        X1 = np.random.randn(30, 2) + np.array([4, 4])

        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        return X, y

    def test_knn_fit(self, sample_data):
        """Test fitting KNN classifier."""
        X, y = sample_data
        clf = KNNClassifier(k=3)

        result = clf.fit(X, y)

        assert result is clf
        assert clf.X_train is not None
        assert clf.y_train is not None
        assert len(clf.classes) == 2

    def test_knn_predict(self, sample_data):
        """Test KNN prediction."""
        X, y = sample_data
        clf = KNNClassifier(k=5).fit(X, y)

        predictions = clf.predict(X)

        assert predictions.shape == y.shape
        assert set(predictions).issubset({0, 1})

    def test_knn_predict_proba(self, sample_data):
        """Test KNN probability predictions."""
        X, y = sample_data
        clf = KNNClassifier(k=5).fit(X, y)

        probs = clf.predict_proba(X)

        assert probs.shape == (60, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_knn_different_k_values(self, sample_data):
        """Test KNN with different k values."""
        X, y = sample_data

        for k in [1, 3, 5, 10]:
            clf = KNNClassifier(k=k).fit(X, y)
            score = clf.score(X, y)
            assert 0.7 <= score <= 1.0


class TestVulnerabilityRiskScorer:
    """Test VulnerabilityRiskScorer."""

    @pytest.fixture
    def sample_cve(self):
        """Sample CVE for testing."""
        return {
            "cve_id": "CVE-2024-1234",
            "cvss_v3_score": 9.8,
            "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "cwe_ids": ["CWE-798"],
            "device_type": "imaging",
            "clinical_impact": "HIGH",
            "exploitability": "EASY",
            "published_date": "2024-01-15T10:00:00.000Z",
            "references": ["https://example.com/exploit"],
        }

    @pytest.fixture
    def training_data_file(self, tmp_path):
        """Create temporary training data file."""
        cves = [
            {
                "cve_id": f"CVE-2024-{i:04d}",
                "cvss_v3_score": np.random.uniform(0, 10),
                "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "cwe_ids": ["CWE-798"],
                "device_type": np.random.choice(["imaging", "monitoring", "infusion"]),
                "clinical_impact": np.random.choice(["HIGH", "MEDIUM", "LOW"]),
                "exploitability": np.random.choice(["EASY", "MODERATE", "HARD"]),
                "published_date": "2024-01-15T10:00:00.000Z",
                "references": [],
            }
            for i in range(100)
        ]

        data = {"cves": cves}
        file_path = tmp_path / "training_data.json"

        with open(file_path, "w") as f:
            json.dump(data, f)

        return file_path

    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        scorer = VulnerabilityRiskScorer()

        assert scorer.is_trained is False
        assert scorer.feature_names == []

    def test_parse_cvss_vector(self):
        """Test CVSS vector parsing."""
        scorer = VulnerabilityRiskScorer()

        result = scorer._parse_cvss_vector(
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        )

        assert result["av"] == 0.85  # Network
        assert result["ac"] == 0.77  # Low
        assert result["pr"] == 0.85  # None
        assert result["c"] == 0.56  # High

    def test_parse_cvss_vector_none(self):
        """Test CVSS vector parsing with None input."""
        scorer = VulnerabilityRiskScorer()

        result = scorer._parse_cvss_vector(None)

        # Should return defaults
        assert "av" in result
        assert result["av"] == 0.5

    def test_map_cwe_to_domain(self):
        """Test CWE to domain mapping."""
        scorer = VulnerabilityRiskScorer()

        # Memory safety CWE
        domain = scorer._map_cwe_to_domain(["CWE-119"])
        assert domain == "memory_safety"

        # Authentication CWE
        domain = scorer._map_cwe_to_domain(["CWE-798"])
        assert domain == "authentication"

        # Unknown CWE
        domain = scorer._map_cwe_to_domain(["CWE-9999"])
        assert domain == "other"

        # Empty list
        domain = scorer._map_cwe_to_domain([])
        assert domain == "other"

    def test_encode_device_type(self):
        """Test device type encoding."""
        scorer = VulnerabilityRiskScorer()

        result = scorer._encode_device_type("imaging")

        assert result["device_imaging"] == 1
        assert result["device_monitoring"] == 0

    def test_encode_clinical_impact(self):
        """Test clinical impact encoding."""
        scorer = VulnerabilityRiskScorer()

        assert scorer._encode_clinical_impact("HIGH") == 1.0
        assert scorer._encode_clinical_impact("MEDIUM") == 0.6
        assert scorer._encode_clinical_impact("LOW") == 0.3
        assert scorer._encode_clinical_impact(None) == 0.5

    def test_extract_features(self, sample_cve):
        """Test feature extraction from CVE."""
        scorer = VulnerabilityRiskScorer()

        features = scorer._extract_features(sample_cve)

        assert "cvss_score" in features
        assert features["cvss_score"] == 9.8
        assert "clinical_impact" in features
        assert "device_imaging" in features
        assert features["device_imaging"] == 1

    def test_compute_target(self):
        """Test target priority computation."""
        scorer = VulnerabilityRiskScorer()

        # Critical case
        cve_critical = {
            "cvss_v3_score": 10.0,
            "clinical_impact": "HIGH",
            "exploitability": "EASY",
            "device_type": "monitoring",
        }
        assert scorer._compute_target(cve_critical) == "CRITICAL"

        # Low case
        cve_low = {
            "cvss_v3_score": 2.0,
            "clinical_impact": "LOW",
            "exploitability": "HARD",
            "device_type": "other",
        }
        assert scorer._compute_target(cve_low) == "LOW"

    def test_load_training_data(self, training_data_file):
        """Test loading training data."""
        scorer = VulnerabilityRiskScorer()

        count = scorer.load_training_data(training_data_file)

        assert count == 100
        assert "X" in scorer.training_data
        assert "y" in scorer.training_data
        assert len(scorer.feature_names) > 0

    def test_train(self, training_data_file):
        """Test model training."""
        scorer = VulnerabilityRiskScorer()
        scorer.load_training_data(training_data_file)

        metrics = scorer.train(test_size=0.2)

        assert scorer.is_trained is True
        assert "nb_test_accuracy" in metrics
        assert "knn_test_accuracy" in metrics
        assert 0 <= metrics["nb_test_accuracy"] <= 1
        assert 0 <= metrics["knn_test_accuracy"] <= 1

    def test_predict(self, training_data_file, sample_cve):
        """Test prediction on single CVE."""
        scorer = VulnerabilityRiskScorer()
        scorer.load_training_data(training_data_file)
        scorer.train()

        prediction = scorer.predict(sample_cve)

        assert isinstance(prediction, RiskPrediction)
        assert prediction.cve_id == "CVE-2024-1234"
        assert 0 <= prediction.risk_score <= 100
        assert prediction.priority in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert 0 <= prediction.confidence <= 1

    def test_predict_batch(self, training_data_file):
        """Test batch prediction."""
        scorer = VulnerabilityRiskScorer()
        scorer.load_training_data(training_data_file)
        scorer.train()

        cves = [
            {"cve_id": "CVE-2024-0001", "cvss_v3_score": 9.0},
            {"cve_id": "CVE-2024-0002", "cvss_v3_score": 5.0},
            {"cve_id": "CVE-2024-0003", "cvss_v3_score": 2.0},
        ]

        predictions = scorer.predict_batch(cves)

        assert len(predictions) == 3
        assert all(isinstance(p, RiskPrediction) for p in predictions)

    def test_save_and_load_model(self, training_data_file, tmp_path, sample_cve):
        """Test saving and loading model."""
        scorer = VulnerabilityRiskScorer()
        scorer.load_training_data(training_data_file)
        scorer.train()

        # Get prediction before save
        pred_before = scorer.predict(sample_cve)

        # Save model
        model_path = tmp_path / "model"
        scorer.save_model(model_path)

        # Load into new scorer
        new_scorer = VulnerabilityRiskScorer()
        new_scorer.load_model(model_path)

        # Predictions should be the same
        pred_after = new_scorer.predict(sample_cve)

        assert pred_before.risk_score == pytest.approx(pred_after.risk_score, rel=0.01)
        assert pred_before.priority == pred_after.priority


class TestCWEDomains:
    """Test CWE domain mappings."""

    def test_cwe_domains_not_empty(self):
        """Test CWE domains dictionary is populated."""
        assert len(CWE_DOMAINS) > 0

    def test_cwe_domains_valid_domains(self):
        """Test all CWE domains map to valid domain names."""
        valid_domains = {
            "memory_safety",
            "authentication",
            "injection",
            "cryptography",
            "access_control",
            "input_validation",
            "resource_management",
        }

        for cwe, domain in CWE_DOMAINS.items():
            assert domain in valid_domains, f"{cwe} maps to invalid domain {domain}"

    def test_common_cwes_mapped(self):
        """Test common CWEs are mapped."""
        common_cwes = ["CWE-79", "CWE-89", "CWE-119", "CWE-287", "CWE-798"]

        for cwe in common_cwes:
            assert cwe in CWE_DOMAINS, f"Common {cwe} not mapped"
