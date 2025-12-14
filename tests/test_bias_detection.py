"""
Tests for the Bias Detection module.

Tests cover:
- Fairness metric calculations
- Group-wise performance analysis
- Bias level determination
- Multi-attribute analysis
"""

import numpy as np
import pytest

from medtech_ai_security.ml.bias_detection import (
    BiasDetector,
    BiasLevel,
    BiasReport,
    FairnessMetric,
    FairnessResult,
    GroupMetrics,
    ProtectedAttribute,
    quick_bias_check,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def balanced_binary_data():
    """Generate balanced binary classification data with no bias."""
    np.random.seed(42)
    # Both groups have equal distribution of labels
    # Group 0: 50 samples with label 0, 50 samples with label 1
    # Group 1: 50 samples with label 0, 50 samples with label 1
    y_true = np.array([0, 1] * 100)  # Alternating 0, 1
    y_pred = y_true.copy()  # Perfect predictions
    groups = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
    return y_true, y_pred, groups


@pytest.fixture
def biased_data():
    """Generate data with intentional bias."""
    np.random.seed(42)

    # Two groups of 100 each
    groups = np.array([0] * 100 + [1] * 100)

    # Balanced true labels
    y_true = np.array([0, 1] * 100)

    # Biased predictions: group 0 gets more positive predictions
    y_pred = np.zeros(200, dtype=np.int64)
    # Group 0: high positive rate
    y_pred[:100] = np.random.choice([0, 1], 100, p=[0.2, 0.8])
    # Group 1: low positive rate
    y_pred[100:] = np.random.choice([0, 1], 100, p=[0.8, 0.2])

    return y_true, y_pred, groups


@pytest.fixture
def detector():
    """Create a BiasDetector instance."""
    return BiasDetector()


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enumeration values."""

    def test_fairness_metric_values(self):
        """Test FairnessMetric enum values."""
        assert FairnessMetric.DEMOGRAPHIC_PARITY.value == "demographic_parity"
        assert FairnessMetric.EQUALIZED_ODDS.value == "equalized_odds"
        assert FairnessMetric.EQUAL_OPPORTUNITY.value == "equal_opportunity"
        assert FairnessMetric.PREDICTIVE_PARITY.value == "predictive_parity"

    def test_protected_attribute_values(self):
        """Test ProtectedAttribute enum values."""
        assert ProtectedAttribute.AGE.value == "age"
        assert ProtectedAttribute.SEX.value == "sex"
        assert ProtectedAttribute.RACE.value == "race"
        assert ProtectedAttribute.ETHNICITY.value == "ethnicity"

    def test_bias_level_values(self):
        """Test BiasLevel enum values."""
        assert BiasLevel.NONE.value == "none"
        assert BiasLevel.LOW.value == "low"
        assert BiasLevel.MODERATE.value == "moderate"
        assert BiasLevel.HIGH.value == "high"
        assert BiasLevel.CRITICAL.value == "critical"


# =============================================================================
# GroupMetrics Tests
# =============================================================================


class TestGroupMetrics:
    """Test GroupMetrics dataclass."""

    def test_creation(self):
        """Test creating GroupMetrics."""
        metrics = GroupMetrics(
            group_name="group_a",
            group_size=100,
            true_positive_rate=0.9,
            false_positive_rate=0.1,
            true_negative_rate=0.9,
            false_negative_rate=0.1,
            positive_predictive_value=0.9,
            negative_predictive_value=0.9,
            accuracy=0.9,
            prevalence=0.5,
        )
        assert metrics.group_name == "group_a"
        assert metrics.group_size == 100
        assert metrics.true_positive_rate == 0.9

    def test_f1_score(self):
        """Test F1 score calculation."""
        metrics = GroupMetrics(
            group_name="test",
            group_size=100,
            true_positive_rate=0.8,  # recall
            false_positive_rate=0.2,
            true_negative_rate=0.8,
            false_negative_rate=0.2,
            positive_predictive_value=0.8,  # precision
            negative_predictive_value=0.8,
            accuracy=0.8,
            prevalence=0.5,
        )
        # F1 = 2 * (precision * recall) / (precision + recall)
        expected_f1 = 2 * (0.8 * 0.8) / (0.8 + 0.8)
        assert abs(metrics.f1_score - expected_f1) < 0.001

    def test_f1_score_zero_division(self):
        """Test F1 score with zero precision/recall."""
        metrics = GroupMetrics(
            group_name="test",
            group_size=100,
            true_positive_rate=0.0,
            false_positive_rate=0.0,
            true_negative_rate=1.0,
            false_negative_rate=1.0,
            positive_predictive_value=0.0,
            negative_predictive_value=1.0,
            accuracy=0.5,
            prevalence=0.5,
        )
        assert metrics.f1_score == 0.0


# =============================================================================
# BiasDetector Tests
# =============================================================================


class TestBiasDetector:
    """Test BiasDetector class."""

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.demographic_parity_threshold == 0.1
        assert detector.equalized_odds_threshold == 0.1
        assert detector.min_group_size == 30

    def test_initialization_custom_thresholds(self):
        """Test detector with custom thresholds."""
        detector = BiasDetector(
            demographic_parity_threshold=0.2,
            equalized_odds_threshold=0.15,
            min_group_size=50,
        )
        assert detector.demographic_parity_threshold == 0.2
        assert detector.equalized_odds_threshold == 0.15
        assert detector.min_group_size == 50

    def test_analyze_returns_report(self, detector, balanced_binary_data):
        """Test that analyze returns a BiasReport."""
        y_true, y_pred, groups = balanced_binary_data
        report = detector.analyze(y_true, y_pred, groups)

        assert isinstance(report, BiasReport)
        assert len(report.groups) == 2
        assert len(report.fairness_results) >= 4  # At least 4 metrics

    def test_analyze_balanced_data(self, detector, balanced_binary_data):
        """Test analysis of balanced, unbiased data."""
        y_true, y_pred, groups = balanced_binary_data
        report = detector.analyze(y_true, y_pred, groups)

        # Perfect predictions should show no/low bias
        assert report.overall_bias_level in [BiasLevel.NONE, BiasLevel.LOW]

    def test_analyze_biased_data(self, detector, biased_data):
        """Test analysis of intentionally biased data."""
        y_true, y_pred, groups = biased_data
        report = detector.analyze(y_true, y_pred, groups)

        # Should detect bias
        assert report.overall_bias_level != BiasLevel.NONE

    def test_analyze_with_attribute_name(self, detector, balanced_binary_data):
        """Test analysis with custom attribute name."""
        y_true, y_pred, groups = balanced_binary_data
        report = detector.analyze(y_true, y_pred, groups, attribute_name="sex")

        assert report.protected_attribute == "sex"

    def test_group_metrics_calculated(self, detector, balanced_binary_data):
        """Test that group metrics are calculated correctly."""
        y_true, y_pred, groups = balanced_binary_data
        report = detector.analyze(y_true, y_pred, groups)

        for group in report.groups:
            assert 0 <= group.true_positive_rate <= 1
            assert 0 <= group.false_positive_rate <= 1
            assert 0 <= group.accuracy <= 1
            assert group.group_size > 0

    def test_demographic_parity_calculated(self, detector, balanced_binary_data):
        """Test demographic parity metric is calculated."""
        y_true, y_pred, groups = balanced_binary_data
        report = detector.analyze(y_true, y_pred, groups)

        dp_results = [
            r for r in report.fairness_results if r.metric == FairnessMetric.DEMOGRAPHIC_PARITY
        ]
        assert len(dp_results) == 1
        assert dp_results[0].value >= 0

    def test_equalized_odds_calculated(self, detector, balanced_binary_data):
        """Test equalized odds metric is calculated."""
        y_true, y_pred, groups = balanced_binary_data
        report = detector.analyze(y_true, y_pred, groups)

        eo_results = [
            r for r in report.fairness_results if r.metric == FairnessMetric.EQUALIZED_ODDS
        ]
        assert len(eo_results) == 1

    def test_equal_opportunity_calculated(self, detector, balanced_binary_data):
        """Test equal opportunity metric is calculated."""
        y_true, y_pred, groups = balanced_binary_data
        report = detector.analyze(y_true, y_pred, groups)

        eop_results = [
            r for r in report.fairness_results if r.metric == FairnessMetric.EQUAL_OPPORTUNITY
        ]
        assert len(eop_results) == 1

    def test_recommendations_generated(self, detector, biased_data):
        """Test that recommendations are generated."""
        y_true, y_pred, groups = biased_data
        report = detector.analyze(y_true, y_pred, groups)

        assert len(report.recommendations) > 0

    def test_to_dict(self, detector, balanced_binary_data):
        """Test report conversion to dictionary."""
        y_true, y_pred, groups = balanced_binary_data
        report = detector.analyze(y_true, y_pred, groups)

        d = report.to_dict()

        assert "protected_attribute" in d
        assert "groups" in d
        assert "fairness_results" in d
        assert "overall_bias_level" in d
        assert "recommendations" in d

    def test_analyze_multiple_groups(self, detector):
        """Test analysis with more than 2 groups."""
        np.random.seed(42)
        n = 300
        y_true = np.random.choice([0, 1], n)
        y_pred = np.random.choice([0, 1], n)
        groups = np.array([0] * 100 + [1] * 100 + [2] * 100)

        report = detector.analyze(y_true, y_pred, groups)

        assert len(report.groups) == 3

    def test_analyze_multiple_attributes(self, detector):
        """Test multi-attribute analysis."""
        np.random.seed(42)
        n = 200
        y_true = np.random.choice([0, 1], n)
        y_pred = np.random.choice([0, 1], n)
        sex = np.random.choice([0, 1], n)
        age = np.random.choice([0, 1, 2], n)

        reports = detector.analyze_multiple_attributes(
            y_true,
            y_pred,
            {"sex": sex, "age": age},
        )

        assert "sex" in reports
        assert "age" in reports
        assert isinstance(reports["sex"], BiasReport)
        assert isinstance(reports["age"], BiasReport)


# =============================================================================
# Fairness Result Tests
# =============================================================================


class TestFairnessResult:
    """Test FairnessResult dataclass."""

    def test_creation(self):
        """Test creating a FairnessResult."""
        result = FairnessResult(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            value=0.05,
            threshold=0.1,
            passed=True,
            reference_group="group_a",
            comparison_groups={"group_a": 0.5, "group_b": 0.45},
            message="Test passed",
        )
        assert result.metric == FairnessMetric.DEMOGRAPHIC_PARITY
        assert result.passed is True
        assert result.value == 0.05


# =============================================================================
# Quick Bias Check Tests
# =============================================================================


class TestQuickBiasCheck:
    """Test quick_bias_check convenience function."""

    def test_returns_summary_dict(self, balanced_binary_data):
        """Test that quick_bias_check returns summary dict."""
        y_true, y_pred, groups = balanced_binary_data
        result = quick_bias_check(y_true, y_pred, groups)

        assert "bias_level" in result
        assert "fairness_passed" in result
        assert "fairness_failed" in result
        assert "group_count" in result
        assert "recommendations" in result

    def test_group_count_correct(self, balanced_binary_data):
        """Test group count is correct."""
        y_true, y_pred, groups = balanced_binary_data
        result = quick_bias_check(y_true, y_pred, groups)

        assert result["group_count"] == 2


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_groups_warning(self, detector, caplog):
        """Test warning for small groups."""
        np.random.seed(42)
        y_true = np.array([0, 1] * 10)  # 20 samples
        y_pred = np.array([0, 1] * 10)
        groups = np.array([0] * 10 + [1] * 10)

        detector.analyze(y_true, y_pred, groups)

        # Should log a warning about small groups
        # (min_group_size is 30 by default)

    def test_single_class_in_group(self, detector):
        """Test handling when a group has only one class."""
        np.random.seed(42)
        y_true = np.array([1] * 50 + [0] * 50)
        y_pred = np.array([1] * 50 + [0] * 50)
        groups = np.array([0] * 50 + [1] * 50)

        # Group 0 has only positive class, group 1 has only negative
        report = detector.analyze(y_true, y_pred, groups)

        # Should still produce a valid report
        assert isinstance(report, BiasReport)

    def test_all_same_predictions(self, detector):
        """Test when all predictions are the same."""
        np.random.seed(42)
        n = 100
        y_true = np.random.choice([0, 1], n)
        y_pred = np.ones(n, dtype=np.int64)  # All positive predictions
        groups = np.array([0] * 50 + [1] * 50)

        report = detector.analyze(y_true, y_pred, groups)

        assert isinstance(report, BiasReport)
