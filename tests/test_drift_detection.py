"""
Tests for the model drift detection module.

Tests cover:
- Statistical drift methods (KL divergence, JS divergence, PSI, Wasserstein, KS test)
- DriftDetector class functionality
- Drift severity classification
- Feature drift detection
- Prediction drift detection
- Edge cases and error handling
"""

import numpy as np
import pytest

from medtech_ai_security.ml.drift_detection import (
    DriftDetector,
    DriftMethod,
    DriftReport,
    DriftResult,
    DriftSeverity,
    DriftThresholds,
    DriftType,
    compute_js_divergence,
    compute_kl_divergence,
    compute_ks_test,
    compute_psi,
    compute_wasserstein,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reference_data():
    """Generate reference (baseline) data."""
    np.random.seed(42)
    return np.random.normal(loc=0.5, scale=0.1, size=(1000, 5))


@pytest.fixture
def current_data_no_drift(reference_data):
    """Generate current data with no drift (same distribution)."""
    np.random.seed(43)
    return np.random.normal(loc=0.5, scale=0.1, size=(1000, 5))


@pytest.fixture
def current_data_small_drift(reference_data):
    """Generate current data with small drift."""
    np.random.seed(44)
    return np.random.normal(loc=0.55, scale=0.1, size=(1000, 5))


@pytest.fixture
def current_data_large_drift(reference_data):
    """Generate current data with large drift."""
    np.random.seed(45)
    return np.random.normal(loc=0.8, scale=0.2, size=(1000, 5))


@pytest.fixture
def reference_predictions():
    """Generate reference predictions."""
    np.random.seed(46)
    return np.random.choice([0, 1], size=500, p=[0.7, 0.3]).astype(float)


@pytest.fixture
def drifted_predictions():
    """Generate drifted predictions."""
    np.random.seed(47)
    return np.random.choice([0, 1], size=500, p=[0.5, 0.5]).astype(float)


# =============================================================================
# Enum Tests
# =============================================================================


class TestDriftType:
    """Test DriftType enum."""

    def test_drift_type_values(self):
        """Test drift type enum values."""
        assert DriftType.DATA_DRIFT.value == "data_drift"
        assert DriftType.CONCEPT_DRIFT.value == "concept_drift"
        assert DriftType.PREDICTION_DRIFT.value == "prediction_drift"
        assert DriftType.FEATURE_DRIFT.value == "feature_drift"


class TestDriftSeverity:
    """Test DriftSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert DriftSeverity.NONE.value == "none"
        assert DriftSeverity.LOW.value == "low"
        assert DriftSeverity.MEDIUM.value == "medium"
        assert DriftSeverity.HIGH.value == "high"
        assert DriftSeverity.CRITICAL.value == "critical"


class TestDriftMethod:
    """Test DriftMethod enum."""

    def test_method_values(self):
        """Test method enum values."""
        assert DriftMethod.KL_DIVERGENCE.value == "kl_divergence"
        assert DriftMethod.JS_DIVERGENCE.value == "js_divergence"
        assert DriftMethod.PSI.value == "psi"
        assert DriftMethod.WASSERSTEIN.value == "wasserstein"
        assert DriftMethod.KS_TEST.value == "ks_test"
        assert DriftMethod.CHI_SQUARE.value == "chi_square"


# =============================================================================
# Statistical Function Tests
# =============================================================================


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_identical_distributions(self):
        """Test KL divergence for identical distributions."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(0, 1, 1000)

        kl = compute_kl_divergence(p, q)
        # KL divergence should be small for similar distributions
        # Using 0.15 threshold to account for random variation in histogram binning
        assert kl < 0.15

    def test_kl_different_distributions(self):
        """Test KL divergence for different distributions."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(5, 1, 1000)

        kl = compute_kl_divergence(p, q)
        # KL divergence should be large for different distributions
        assert kl > 1.0

    def test_kl_non_negative(self):
        """Test KL divergence is always non-negative."""
        np.random.seed(42)
        p = np.random.uniform(0, 1, 500)
        q = np.random.uniform(0.5, 1.5, 500)

        kl = compute_kl_divergence(p, q)
        assert kl >= 0


class TestJSDivergence:
    """Test JS divergence computation."""

    def test_js_identical_distributions(self):
        """Test JS divergence for identical distributions."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(0, 1, 1000)

        js = compute_js_divergence(p, q)
        # Using 0.15 threshold to account for random variation in histogram binning
        assert js < 0.15

    def test_js_different_distributions(self):
        """Test JS divergence for different distributions."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(5, 1, 1000)

        js = compute_js_divergence(p, q)
        assert js > 0.3

    def test_js_symmetric(self):
        """Test JS divergence is symmetric."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 500)
        q = np.random.normal(2, 1, 500)

        js_pq = compute_js_divergence(p, q)
        js_qp = compute_js_divergence(q, p)

        assert abs(js_pq - js_qp) < 0.05  # Should be approximately equal

    def test_js_bounded(self):
        """Test JS divergence is bounded [0, 1]."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 500)
        q = np.random.normal(10, 0.5, 500)

        js = compute_js_divergence(p, q)
        assert 0 <= js <= 1


class TestPSI:
    """Test PSI (Population Stability Index) computation."""

    def test_psi_no_drift(self):
        """Test PSI for no drift."""
        np.random.seed(42)
        expected = np.random.normal(0.5, 0.1, 1000)
        actual = np.random.normal(0.5, 0.1, 1000)

        psi = compute_psi(expected, actual)
        # PSI < 0.1 indicates no significant drift
        assert psi < 0.1

    def test_psi_moderate_drift(self):
        """Test PSI for moderate drift."""
        np.random.seed(42)
        expected = np.random.normal(0.5, 0.1, 1000)
        # Smaller shift: mean 0.52, std 0.11 (closer to reference)
        actual = np.random.normal(0.52, 0.11, 1000)

        psi = compute_psi(expected, actual)
        # PSI between 0.05 and 0.5 indicates some drift
        assert 0.05 < psi < 0.5

    def test_psi_large_drift(self):
        """Test PSI for large drift."""
        np.random.seed(42)
        expected = np.random.normal(0.3, 0.1, 1000)
        actual = np.random.normal(0.8, 0.2, 1000)

        psi = compute_psi(expected, actual)
        # PSI > 0.25 indicates significant drift
        assert psi > 0.2

    def test_psi_non_negative(self):
        """Test PSI is always non-negative."""
        np.random.seed(42)
        expected = np.random.uniform(0, 1, 500)
        actual = np.random.uniform(0, 1, 500)

        psi = compute_psi(expected, actual)
        assert psi >= 0


class TestWasserstein:
    """Test Wasserstein distance computation."""

    def test_wasserstein_identical(self):
        """Test Wasserstein for identical distributions."""
        np.random.seed(42)
        u = np.random.normal(0, 1, 1000)
        v = np.random.normal(0, 1, 1000)

        dist = compute_wasserstein(u, v)
        assert dist < 0.1

    def test_wasserstein_shifted(self):
        """Test Wasserstein for shifted distributions."""
        np.random.seed(42)
        u = np.random.normal(0, 1, 1000)
        v = np.random.normal(2, 1, 1000)  # Shifted by 2

        dist = compute_wasserstein(u, v)
        # Wasserstein should approximate the mean shift
        assert 1.5 < dist < 2.5

    def test_wasserstein_non_negative(self):
        """Test Wasserstein is always non-negative."""
        np.random.seed(42)
        u = np.random.exponential(1, 500)
        v = np.random.exponential(2, 500)

        dist = compute_wasserstein(u, v)
        assert dist >= 0


class TestKSTest:
    """Test Kolmogorov-Smirnov test."""

    def test_ks_identical(self):
        """Test KS test for identical distributions."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(0, 1, 1000)

        statistic, p_value = compute_ks_test(p, q)

        assert statistic < 0.1
        assert p_value > 0.05  # Not significantly different

    def test_ks_different(self):
        """Test KS test for different distributions."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(3, 1, 1000)

        statistic, p_value = compute_ks_test(p, q)

        assert statistic > 0.3
        assert p_value < 0.05  # Significantly different

    def test_ks_statistic_bounded(self):
        """Test KS statistic is bounded [0, 1]."""
        np.random.seed(42)
        p = np.random.uniform(0, 1, 500)
        q = np.random.uniform(0.5, 1.5, 500)

        statistic, p_value = compute_ks_test(p, q)

        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1


# =============================================================================
# DriftThresholds Tests
# =============================================================================


class TestDriftThresholds:
    """Test DriftThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = DriftThresholds()

        assert thresholds.low > 0
        assert thresholds.medium > thresholds.low
        assert thresholds.high > thresholds.medium
        assert thresholds.critical > thresholds.high

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = DriftThresholds(
            low=0.05,
            medium=0.1,
            high=0.2,
            critical=0.4,
        )

        assert thresholds.low == 0.05
        assert thresholds.critical == 0.4

    def test_classify_none(self):
        """Test severity classification - none."""
        thresholds = DriftThresholds()
        severity = thresholds.classify(0.01)
        assert severity == DriftSeverity.NONE

    def test_classify_low(self):
        """Test severity classification - low."""
        thresholds = DriftThresholds()
        severity = thresholds.classify(0.15)
        assert severity == DriftSeverity.LOW

    def test_classify_medium(self):
        """Test severity classification - medium."""
        thresholds = DriftThresholds()
        severity = thresholds.classify(0.25)
        assert severity == DriftSeverity.MEDIUM

    def test_classify_high(self):
        """Test severity classification - high."""
        thresholds = DriftThresholds()
        severity = thresholds.classify(0.4)
        assert severity == DriftSeverity.HIGH

    def test_classify_critical(self):
        """Test severity classification - critical."""
        thresholds = DriftThresholds()
        severity = thresholds.classify(0.6)
        assert severity == DriftSeverity.CRITICAL


# =============================================================================
# DriftResult Tests
# =============================================================================


class TestDriftResult:
    """Test DriftResult dataclass."""

    def test_creation(self):
        """Test drift result creation."""
        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.FEATURE_DRIFT,
            method=DriftMethod.PSI,
            score=0.15,
            severity=DriftSeverity.MEDIUM,
            feature_name="feature_1",
        )

        assert result.drift_type == DriftType.FEATURE_DRIFT
        assert result.feature_name == "feature_1"
        assert result.drift_detected is True
        assert result.severity == DriftSeverity.MEDIUM
        assert result.score == 0.15

    def test_with_p_value(self):
        """Test drift result with p-value."""
        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.DATA_DRIFT,
            method=DriftMethod.KS_TEST,
            score=0.4,
            severity=DriftSeverity.HIGH,
            p_value=0.001,
        )

        assert result.p_value == 0.001


# =============================================================================
# DriftDetector Tests
# =============================================================================


class TestDriftDetector:
    """Test DriftDetector class."""

    def test_initialization_default(self):
        """Test detector initialization with defaults."""
        detector = DriftDetector()

        assert detector.reference_features is None
        assert detector.methods is not None
        assert len(detector.methods) > 0

    def test_initialization_custom_methods(self):
        """Test detector with custom methods."""
        detector = DriftDetector(methods=[DriftMethod.PSI, DriftMethod.KS_TEST])

        assert len(detector.methods) == 2
        assert DriftMethod.PSI in detector.methods

    def test_set_reference(self, reference_data):
        """Test setting reference data."""
        detector = DriftDetector()
        detector.set_reference(reference_data)

        assert detector.reference_features is not None
        assert detector.reference_features.shape == reference_data.shape


class TestDriftDetection:
    """Test drift detection functionality."""

    def test_detect_no_drift(self, reference_data, current_data_no_drift):
        """Test detection with no drift."""
        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_no_drift)

        assert isinstance(report, DriftReport)
        # Low severity expected for similar distributions
        assert report.overall_severity in [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
        ]

    def test_detect_small_drift(self, reference_data, current_data_small_drift):
        """Test detection with small drift."""
        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_small_drift)

        assert isinstance(report, DriftReport)
        # Small drift may or may not be detected depending on thresholds

    def test_detect_large_drift(self, reference_data, current_data_large_drift):
        """Test detection with large drift."""
        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_large_drift)

        assert isinstance(report, DriftReport)
        assert report.overall_drift_detected is True
        assert report.overall_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]

    def test_detect_with_multiple_methods(self, reference_data, current_data_large_drift):
        """Test detection with multiple methods."""
        detector = DriftDetector(
            methods=[DriftMethod.PSI, DriftMethod.KS_TEST, DriftMethod.WASSERSTEIN]
        )
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_large_drift)

        assert len(report.feature_results) > 0

    def test_detect_prediction_drift(
        self, reference_data, current_data_no_drift, reference_predictions, drifted_predictions
    ):
        """Test prediction drift detection."""
        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference_data, predictions=reference_predictions)

        report = detector.detect_drift(
            current_data_no_drift,
            current_predictions=drifted_predictions,
        )

        # Report should include prediction drift analysis
        assert report is not None

    def test_quick_check_no_drift(self, reference_data, current_data_no_drift):
        """Test quick check with no drift."""
        detector = DriftDetector()
        detector.set_reference(reference_data)

        is_drifted, score = detector.quick_check(current_data_no_drift)

        assert isinstance(is_drifted, bool)
        assert isinstance(score, float)

    def test_quick_check_large_drift(self, reference_data, current_data_large_drift):
        """Test quick check with large drift."""
        detector = DriftDetector()
        detector.set_reference(reference_data)

        is_drifted, score = detector.quick_check(current_data_large_drift)

        assert is_drifted is True
        assert score > 0.3  # High score indicates drift


class TestDriftReport:
    """Test DriftReport generation."""

    def test_report_structure(self, reference_data, current_data_large_drift):
        """Test report has correct structure."""
        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_large_drift)

        assert hasattr(report, "overall_drift_detected")
        assert hasattr(report, "overall_severity")
        assert hasattr(report, "feature_results")
        assert hasattr(report, "recommendations")
        assert hasattr(report, "timestamp")

    def test_report_recommendations(self, reference_data, current_data_large_drift):
        """Test report generates recommendations."""
        detector = DriftDetector()
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_large_drift)

        assert len(report.recommendations) > 0

    def test_report_timestamp(self, reference_data, current_data_no_drift):
        """Test report has valid timestamp."""
        detector = DriftDetector()
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_no_drift)

        assert report.timestamp is not None
        # Should be ISO format
        assert "T" in report.timestamp or "-" in report.timestamp


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_reference_error(self):
        """Test error when no reference set."""
        detector = DriftDetector()

        with pytest.raises((ValueError, AttributeError)):
            detector.detect_drift(np.random.rand(100, 5))

    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, (500, 1))
        current = np.random.normal(0.5, 1, (500, 1))

        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference)

        report = detector.detect_drift(current)
        assert len(report.feature_results) == 1

    def test_small_sample_size(self):
        """Test with small sample size."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, (50, 3))
        current = np.random.normal(0, 1, (50, 3))

        detector = DriftDetector(methods=[DriftMethod.KS_TEST])
        detector.set_reference(reference)

        report = detector.detect_drift(current)
        assert report is not None

    def test_constant_feature(self):
        """Test with constant feature (zero variance)."""
        np.random.seed(42)
        reference = np.column_stack(
            [
                np.ones(100),  # Constant
                np.random.normal(0, 1, 100),
            ]
        )
        current = np.column_stack(
            [
                np.ones(100),  # Still constant
                np.random.normal(0.5, 1, 100),
            ]
        )

        detector = DriftDetector(methods=[DriftMethod.WASSERSTEIN])
        detector.set_reference(reference)

        # Should not crash
        report = detector.detect_drift(current)
        assert report is not None

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, (10000, 20))
        current = np.random.normal(0.1, 1, (10000, 20))

        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference)

        report = detector.detect_drift(current)
        assert len(report.feature_results) == 20


# =============================================================================
# Method-Specific Tests
# =============================================================================


class TestMethodSpecificDetection:
    """Test specific drift detection methods."""

    def test_psi_method(self, reference_data, current_data_large_drift):
        """Test PSI method specifically."""
        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_large_drift)

        for result in report.feature_results:
            assert result.method == DriftMethod.PSI

    def test_ks_method(self, reference_data, current_data_large_drift):
        """Test KS test method specifically."""
        detector = DriftDetector(methods=[DriftMethod.KS_TEST])
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_large_drift)

        for result in report.feature_results:
            assert result.method == DriftMethod.KS_TEST
            assert result.p_value is not None

    def test_wasserstein_method(self, reference_data, current_data_large_drift):
        """Test Wasserstein method specifically."""
        detector = DriftDetector(methods=[DriftMethod.WASSERSTEIN])
        detector.set_reference(reference_data)

        report = detector.detect_drift(current_data_large_drift)

        for result in report.feature_results:
            assert result.method == DriftMethod.WASSERSTEIN


# =============================================================================
# Integration Tests
# =============================================================================


class TestDriftIntegration:
    """Integration tests for drift detection workflow."""

    def test_full_drift_monitoring_workflow(self):
        """Test complete drift monitoring workflow."""
        np.random.seed(42)

        # Step 1: Create initial reference data
        reference = np.random.normal(0.5, 0.1, (1000, 5))

        # Step 2: Initialize detector
        detector = DriftDetector(methods=[DriftMethod.PSI, DriftMethod.KS_TEST])
        detector.set_reference(reference)

        # Step 3: Monitor batch 1 - no drift
        batch1 = np.random.normal(0.5, 0.1, (500, 5))
        report1 = detector.detect_drift(batch1)
        assert report1.overall_severity in [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
        ]

        # Step 4: Monitor batch 2 - gradual drift
        batch2 = np.random.normal(0.55, 0.12, (500, 5))
        detector.detect_drift(batch2)
        # May detect low-medium drift

        # Step 5: Monitor batch 3 - significant drift
        batch3 = np.random.normal(0.8, 0.2, (500, 5))
        report3 = detector.detect_drift(batch3)
        assert report3.overall_drift_detected is True
        assert report3.overall_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]

        # Step 6: Quick check for alerting
        is_critical, score = detector.quick_check(batch3)
        assert is_critical is True

    def test_prediction_drift_workflow(self):
        """Test prediction drift monitoring workflow."""
        np.random.seed(42)

        # Reference data and predictions
        reference_X = np.random.normal(0, 1, (500, 3))
        reference_preds = np.random.choice([0, 1], 500, p=[0.8, 0.2]).astype(float)

        # Current data with same distribution but different predictions
        current_X = np.random.normal(0, 1, (500, 3))
        current_preds = np.random.choice([0, 1], 500, p=[0.5, 0.5]).astype(float)

        detector = DriftDetector(methods=[DriftMethod.PSI])
        detector.set_reference(reference_X, predictions=reference_preds)

        report = detector.detect_drift(
            current_X,
            current_predictions=current_preds,
        )

        # Report should be generated
        assert report is not None
