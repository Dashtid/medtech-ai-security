"""
Tests for the Data Poisoning Defense module.

Tests cover:
- Training data validation (outlier detection)
- Influence analysis
- Batch analysis
- RONI defense
- Ensemble validation
- Full defense pipeline

"""

import numpy as np
import pytest

from medtech_ai_security.ml.data_poisoning import (
    BatchAnalysisResult,
    ContaminationType,
    DataPoisoningDefense,
    DefenseStrategy,
    EnsembleValidator,
    InfluenceAnalyzer,
    InfluenceResult,
    RONIDefense,
    TrainingDataValidator,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    BatchAnalyzer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def normal_data():
    """Generate normal (clean) data."""
    np.random.seed(42)
    return np.random.randn(100, 10)


@pytest.fixture
def data_with_outliers():
    """Generate data with clear outliers."""
    np.random.seed(42)
    normal = np.random.randn(90, 10)
    outliers = np.random.randn(10, 10) * 10 + 50  # Far from normal
    return np.vstack([normal, outliers])


@pytest.fixture
def binary_labels():
    """Generate binary classification labels."""
    return np.array([0] * 50 + [1] * 50)


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enumeration classes."""

    def test_contamination_type_values(self):
        """Test ContaminationType enum values."""
        assert ContaminationType.LABEL_FLIP.value == "label_flip"
        assert ContaminationType.FEATURE_PERTURBATION.value == "feature_perturbation"
        assert ContaminationType.BACKDOOR.value == "backdoor"
        assert ContaminationType.CLEAN_LABEL.value == "clean_label"

    def test_validation_severity_values(self):
        """Test ValidationSeverity enum values."""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.CRITICAL.value == "critical"

    def test_defense_strategy_values(self):
        """Test DefenseStrategy enum values."""
        assert DefenseStrategy.STATISTICAL.value == "statistical"
        assert DefenseStrategy.ISOLATION_FOREST.value == "isolation_forest"
        assert DefenseStrategy.LOCAL_OUTLIER.value == "local_outlier"
        assert DefenseStrategy.RONI.value == "roni"
        assert DefenseStrategy.ENSEMBLE.value == "ensemble"


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            check_name="test_check",
            passed=True,
            severity=ValidationSeverity.INFO,
            message="Test passed",
            details={"method": "test"},
        )
        assert result.check_name == "test_check"
        assert result.passed is True
        assert result.severity == ValidationSeverity.INFO

    def test_with_flagged_indices(self):
        """Test validation result with flagged indices."""
        result = ValidationResult(
            check_name="outlier_check",
            passed=False,
            severity=ValidationSeverity.WARNING,
            message="Found outliers",
            flagged_indices=[1, 5, 10],
        )
        assert result.passed is False
        assert result.flagged_indices == [1, 5, 10]


# =============================================================================
# ValidationReport Tests
# =============================================================================


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_creation(self):
        """Test creating a validation report."""
        report = ValidationReport(
            timestamp="2025-12-14T00:00:00",
            total_samples=100,
            total_features=10,
        )
        assert report.total_samples == 100
        assert report.total_features == 10
        assert report.overall_passed is True

    def test_with_results(self):
        """Test report with validation results."""
        result = ValidationResult(
            check_name="test",
            passed=True,
            severity=ValidationSeverity.INFO,
            message="OK",
        )
        report = ValidationReport(
            timestamp="2025-12-14T00:00:00",
            total_samples=100,
            total_features=10,
            results=[result],
        )
        assert len(report.results) == 1


# =============================================================================
# TrainingDataValidator Tests
# =============================================================================


class TestTrainingDataValidator:
    """Test TrainingDataValidator class."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = TrainingDataValidator()
        assert validator.contamination == 0.1

    def test_initialization_custom_contamination(self):
        """Test validator with custom contamination rate."""
        validator = TrainingDataValidator(contamination=0.05)
        assert validator.contamination == 0.05

    def test_validate_returns_report(self, normal_data):
        """Test that validate returns a ValidationReport."""
        validator = TrainingDataValidator(contamination=0.1)
        result = validator.validate(normal_data)

        assert isinstance(result, ValidationReport)
        assert result.total_samples == len(normal_data)

    def test_validate_with_outliers(self, data_with_outliers):
        """Test validation of data with outliers."""
        validator = TrainingDataValidator(contamination=0.15)
        result = validator.validate(data_with_outliers)

        assert isinstance(result, ValidationReport)
        # Should detect some contamination
        assert result.contamination_estimate > 0

    def test_validate_with_labels(self, normal_data, binary_labels):
        """Test validation with feature/label consistency."""
        validator = TrainingDataValidator()
        result = validator.validate(normal_data, binary_labels)

        assert isinstance(result, ValidationReport)
        # Should have label balance check
        assert len(result.results) >= 1


# =============================================================================
# InfluenceAnalyzer Tests
# =============================================================================


class TestInfluenceAnalyzer:
    """Test InfluenceAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = InfluenceAnalyzer()
        assert analyzer is not None

    def test_analyze_returns_list(self, normal_data, binary_labels):
        """Test that analyze returns a list of InfluenceResult."""
        analyzer = InfluenceAnalyzer()
        result = analyzer.analyze(normal_data, binary_labels)

        assert isinstance(result, list)
        if len(result) > 0:
            assert isinstance(result[0], InfluenceResult)


# =============================================================================
# BatchAnalyzer Tests
# =============================================================================


class TestBatchAnalyzer:
    """Test BatchAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = BatchAnalyzer()
        assert analyzer is not None

    def test_analyze_returns_list(self, normal_data):
        """Test that analyze returns a list of BatchAnalysisResult."""
        analyzer = BatchAnalyzer()
        result = analyzer.analyze(normal_data)

        assert isinstance(result, list)
        if len(result) > 0:
            assert isinstance(result[0], BatchAnalysisResult)

    def test_batch_has_statistics(self, normal_data):
        """Test that batch results contain statistics."""
        analyzer = BatchAnalyzer()
        result = analyzer.analyze(normal_data)

        if len(result) > 0:
            assert "mean" in result[0].statistics
            assert "std" in result[0].statistics


# =============================================================================
# RONIDefense Tests
# =============================================================================


class TestRONIDefense:
    """Test RONI (Reject On Negative Impact) Defense."""

    def test_initialization(self):
        """Test RONI initialization."""
        defense = RONIDefense()
        assert defense is not None

    def test_filter_returns_tuple(self, normal_data, binary_labels):
        """Test that filter returns a tuple of (X, y, indices)."""
        defense = RONIDefense()
        result = defense.filter(normal_data, binary_labels)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_filter_clean_data(self, normal_data, binary_labels):
        """Test RONI filter on clean data."""
        defense = RONIDefense()
        filtered_x, filtered_y, removed_indices = defense.filter(normal_data, binary_labels)

        # Should retain most clean data
        assert len(filtered_x) >= len(normal_data) * 0.5
        assert len(filtered_y) == len(filtered_x)


# =============================================================================
# EnsembleValidator Tests
# =============================================================================


class TestEnsembleValidator:
    """Test EnsembleValidator class."""

    def test_initialization(self):
        """Test ensemble initialization."""
        validator = EnsembleValidator()
        assert validator is not None

    def test_validate_returns_tuple(self, normal_data, binary_labels):
        """Test that validate returns a tuple."""
        validator = EnsembleValidator()
        result = validator.validate(normal_data, binary_labels)

        assert isinstance(result, tuple)
        assert len(result) == 2


# =============================================================================
# DataPoisoningDefense Tests
# =============================================================================


class TestDataPoisoningDefense:
    """Test the main DataPoisoningDefense class."""

    def test_initialization(self):
        """Test defense initialization."""
        defense = DataPoisoningDefense()
        assert defense is not None

    def test_analyze_returns_dict(self, normal_data, binary_labels):
        """Test that analyze returns a dictionary."""
        defense = DataPoisoningDefense()
        result = defense.analyze(normal_data, binary_labels)

        assert isinstance(result, dict)
        assert "validation" in result or "overall" in result

    def test_filter_returns_tuple(self, normal_data, binary_labels):
        """Test that filter returns a tuple."""
        defense = DataPoisoningDefense()
        result = defense.filter(normal_data, binary_labels)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_filter_clean_data(self, normal_data, binary_labels):
        """Test filtering clean data."""
        defense = DataPoisoningDefense()
        clean_x, clean_y, stats = defense.filter(normal_data, binary_labels)

        # Should retain most samples
        assert len(clean_x) >= len(normal_data) * 0.5
        assert len(clean_y) == len(clean_x)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_contamination(self):
        """Test invalid contamination rate."""
        with pytest.raises((ValueError, AssertionError)):
            TrainingDataValidator(contamination=1.5)

    def test_small_dataset(self):
        """Test with small dataset."""
        np.random.seed(42)
        small_data = np.random.randn(10, 5)
        labels = np.array([0] * 5 + [1] * 5)

        defense = DataPoisoningDefense()
        result = defense.analyze(small_data, labels)

        assert isinstance(result, dict)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestIntegration:
    """Integration tests for data poisoning defense."""

    def test_full_pipeline(self, normal_data, binary_labels):
        """Test complete defense pipeline."""
        defense = DataPoisoningDefense()

        # Analyze
        analysis = defense.analyze(normal_data, binary_labels)
        assert isinstance(analysis, dict)

        # Filter
        clean_x, clean_y, stats = defense.filter(normal_data, binary_labels)
        assert len(clean_x) > 0
        assert len(clean_y) == len(clean_x)
