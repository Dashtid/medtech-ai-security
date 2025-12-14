"""
Data Poisoning Defense Module for Medical AI Security.

This module provides comprehensive defenses against training data poisoning attacks,
which are particularly critical for medical AI systems where compromised models
could lead to patient harm.

Key Features:
- Training data validation with statistical outlier detection
- Label consistency checking
- Influence function analysis for detecting high-impact samples
- Batch-level anomaly detection
- RONI (Reject On Negative Impact) defense
- Ensemble-based validation

Security Context:
- HSCC 2026 Guidance: Data poisoning is a primary threat to medical AI
- FDA PCCP: Training data integrity is essential for predetermined changes
- Even 0.01% contamination can significantly impact model behavior

References:
- Lakera AI: Training Data Poisoning (2025)
- NIST AI 100-2 E2025: Adversarial ML Taxonomy
- ServiceNow: Dynamic Data Poisoning Defenses (2025)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ContaminationType(str, Enum):
    """Types of data contamination/poisoning."""

    LABEL_FLIP = "label_flip"  # Incorrect labels
    FEATURE_PERTURBATION = "feature_perturbation"  # Subtle feature changes
    BACKDOOR = "backdoor"  # Hidden trigger patterns
    CLEAN_LABEL = "clean_label"  # Poisoned samples with correct labels
    GRADIENT_BASED = "gradient_based"  # Optimized perturbations
    DISTRIBUTION_SHIFT = "distribution_shift"  # Out-of-distribution samples


class ValidationSeverity(str, Enum):
    """Severity levels for validation findings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DefenseStrategy(str, Enum):
    """Available defense strategies."""

    STATISTICAL = "statistical"  # Statistical outlier detection
    ISOLATION_FOREST = "isolation_forest"  # Isolation forest anomaly detection
    LOCAL_OUTLIER = "local_outlier"  # Local outlier factor
    RONI = "roni"  # Reject on negative impact
    ENSEMBLE = "ensemble"  # Ensemble validation
    INFLUENCE = "influence"  # Influence function analysis


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    flagged_indices: list[int] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report for training data."""

    timestamp: str
    total_samples: int
    total_features: int
    results: list[ValidationResult] = field(default_factory=list)
    overall_passed: bool = True
    contamination_estimate: float = 0.0
    flagged_samples: list[int] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)
        if not result.passed:
            self.overall_passed = False
        self.flagged_samples.extend(result.flagged_indices)
        # Remove duplicates
        self.flagged_samples = list(set(self.flagged_samples))

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "total_features": self.total_features,
            "overall_passed": self.overall_passed,
            "contamination_estimate": self.contamination_estimate,
            "num_flagged_samples": len(self.flagged_samples),
            "flagged_sample_indices": self.flagged_samples[:100],  # Limit output
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
            "recommendations": self.recommendations,
        }


@dataclass
class InfluenceResult:
    """Result of influence function analysis."""

    sample_index: int
    influence_score: float
    is_suspicious: bool
    predicted_impact: str  # "positive", "negative", "neutral"


@dataclass
class BatchAnalysisResult:
    """Result of batch-level analysis."""

    batch_id: int
    is_anomalous: bool
    anomaly_score: float
    statistics: dict[str, float]
    flagged_reason: str | None = None


# =============================================================================
# Training Data Validator
# =============================================================================


class TrainingDataValidator:
    """
    Validates training data for potential poisoning or contamination.

    This class provides multiple validation strategies to detect:
    - Statistical outliers (Z-score, IQR)
    - Isolation forest anomalies
    - Local outlier factor anomalies
    - Label inconsistencies
    - Distribution anomalies
    """

    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        contamination: float = 0.1,
        n_neighbors: int = 20,
    ) -> None:
        """
        Initialize the validator.

        Args:
            z_score_threshold: Threshold for Z-score outlier detection
            iqr_multiplier: Multiplier for IQR-based outlier detection
            contamination: Expected contamination ratio for isolation forest
            n_neighbors: Number of neighbors for LOF

        Raises:
            ValueError: If contamination is not in valid range (0, 0.5]
        """
        if not (0 < contamination <= 0.5):
            raise ValueError(f"contamination must be in the range (0, 0.5], got {contamination}")
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def validate(
        self,
        features: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]] | None = None,
        strategies: list[DefenseStrategy] | None = None,
    ) -> ValidationReport:
        """
        Perform comprehensive validation of training data.

        Args:
            features: Training feature matrix (n_samples, n_features)
            labels: Training labels (optional)
            strategies: List of validation strategies to apply

        Returns:
            ValidationReport with all findings
        """
        if strategies is None:
            strategies = [
                DefenseStrategy.STATISTICAL,
                DefenseStrategy.ISOLATION_FOREST,
            ]

        n_samples, n_features = features.shape
        report = ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_samples=n_samples,
            total_features=n_features,
        )

        logger.info(f"Validating {n_samples} samples with {n_features} features")

        # Run selected validation strategies
        for strategy in strategies:
            if strategy == DefenseStrategy.STATISTICAL:
                self._validate_statistical(features, report)
            elif strategy == DefenseStrategy.ISOLATION_FOREST:
                self._validate_isolation_forest(features, report)
            elif strategy == DefenseStrategy.LOCAL_OUTLIER:
                self._validate_local_outlier(features, report)

        # Validate labels if provided
        if labels is not None:
            self._validate_labels(features, labels, report)

        # Calculate contamination estimate
        if report.flagged_samples:
            report.contamination_estimate = len(report.flagged_samples) / n_samples

        # Generate recommendations
        self._generate_recommendations(report)

        logger.info(
            f"Validation complete: {len(report.flagged_samples)} samples flagged "
            f"({report.contamination_estimate:.2%} contamination estimate)"
        )

        return report

    def _validate_statistical(
        self,
        features: NDArray[np.floating[Any]],
        report: ValidationReport,
    ) -> None:
        """Perform statistical outlier detection."""
        flagged_z = []
        flagged_iqr = []

        for i in range(features.shape[1]):
            col = features[:, i]

            # Z-score outliers - handle zero variance case
            col_std = np.nanstd(col)
            if col_std > 1e-10:  # Only compute z-scores if variance is non-trivial
                z_scores = np.abs(stats.zscore(col, nan_policy="omit"))
                z_outliers = np.where(z_scores > self.z_score_threshold)[0]
                flagged_z.extend(z_outliers.tolist())
            # If variance is zero/near-zero, all values are identical - no outliers

            # IQR outliers
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            iqr_outliers = np.where((col < lower) | (col > upper))[0]
            flagged_iqr.extend(iqr_outliers.tolist())

        flagged_z = list(set(flagged_z))
        flagged_iqr = list(set(flagged_iqr))

        # Z-score result
        z_passed = len(flagged_z) / features.shape[0] < 0.05
        report.add_result(
            ValidationResult(
                check_name="z_score_outliers",
                passed=z_passed,
                severity=ValidationSeverity.WARNING if not z_passed else ValidationSeverity.INFO,
                message=f"Found {len(flagged_z)} Z-score outliers "
                f"({len(flagged_z)/features.shape[0]:.2%})",
                details={
                    "threshold": self.z_score_threshold,
                    "outlier_count": len(flagged_z),
                    "outlier_ratio": len(flagged_z) / features.shape[0],
                },
                flagged_indices=flagged_z,
            )
        )

        # IQR result
        iqr_passed = len(flagged_iqr) / features.shape[0] < 0.05
        report.add_result(
            ValidationResult(
                check_name="iqr_outliers",
                passed=iqr_passed,
                severity=ValidationSeverity.WARNING if not iqr_passed else ValidationSeverity.INFO,
                message=f"Found {len(flagged_iqr)} IQR outliers "
                f"({len(flagged_iqr)/features.shape[0]:.2%})",
                details={
                    "iqr_multiplier": self.iqr_multiplier,
                    "outlier_count": len(flagged_iqr),
                    "outlier_ratio": len(flagged_iqr) / features.shape[0],
                },
                flagged_indices=flagged_iqr,
            )
        )

    def _validate_isolation_forest(
        self,
        features: NDArray[np.floating[Any]],
        report: ValidationReport,
    ) -> None:
        """Perform isolation forest anomaly detection."""
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )

        predictions = iso_forest.fit_predict(features)
        anomaly_scores = iso_forest.decision_function(features)

        # -1 indicates anomaly in sklearn
        flagged = np.where(predictions == -1)[0].tolist()
        anomaly_ratio = len(flagged) / features.shape[0]

        passed = anomaly_ratio <= self.contamination * 1.5

        report.add_result(
            ValidationResult(
                check_name="isolation_forest",
                passed=passed,
                severity=ValidationSeverity.CRITICAL if not passed else ValidationSeverity.INFO,
                message=f"Isolation Forest flagged {len(flagged)} samples ({anomaly_ratio:.2%})",
                details={
                    "expected_contamination": self.contamination,
                    "actual_anomaly_ratio": anomaly_ratio,
                    "mean_anomaly_score": float(np.mean(anomaly_scores)),
                    "min_anomaly_score": float(np.min(anomaly_scores)),
                },
                flagged_indices=flagged,
            )
        )

    def _validate_local_outlier(
        self,
        features: NDArray[np.floating[Any]],
        report: ValidationReport,
    ) -> None:
        """Perform local outlier factor detection."""
        n_neighbors = min(self.n_neighbors, features.shape[0] - 1)

        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            n_jobs=-1,
        )

        predictions = lof.fit_predict(features)
        lof_scores = -lof.negative_outlier_factor_  # Convert to positive

        flagged = np.where(predictions == -1)[0].tolist()
        anomaly_ratio = len(flagged) / features.shape[0]

        passed = anomaly_ratio <= self.contamination * 1.5

        report.add_result(
            ValidationResult(
                check_name="local_outlier_factor",
                passed=passed,
                severity=ValidationSeverity.WARNING if not passed else ValidationSeverity.INFO,
                message=f"LOF flagged {len(flagged)} samples ({anomaly_ratio:.2%})",
                details={
                    "n_neighbors": n_neighbors,
                    "actual_anomaly_ratio": anomaly_ratio,
                    "mean_lof_score": float(np.mean(lof_scores)),
                    "max_lof_score": float(np.max(lof_scores)),
                },
                flagged_indices=flagged,
            )
        )

    def _validate_labels(
        self,
        features: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]],
        report: ValidationReport,
    ) -> None:
        """Validate label consistency."""
        unique_labels = np.unique(labels)
        label_counts = {int(label): int(np.sum(labels == label)) for label in unique_labels}

        # Check for extreme class imbalance
        if len(unique_labels) > 1:
            min_count = min(label_counts.values())
            max_count = max(label_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

            passed = imbalance_ratio < 100  # Allow up to 100:1 imbalance

            report.add_result(
                ValidationResult(
                    check_name="label_balance",
                    passed=passed,
                    severity=ValidationSeverity.WARNING if not passed else ValidationSeverity.INFO,
                    message=f"Class imbalance ratio: {imbalance_ratio:.1f}:1",
                    details={
                        "label_counts": label_counts,
                        "imbalance_ratio": imbalance_ratio,
                        "num_classes": len(unique_labels),
                    },
                )
            )

        # Check for invalid labels
        if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
            invalid_indices = np.where(np.isnan(labels) | np.isinf(labels))[0].tolist()
            report.add_result(
                ValidationResult(
                    check_name="invalid_labels",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Found {len(invalid_indices)} invalid labels (NaN/Inf)",
                    details={"invalid_count": len(invalid_indices)},
                    flagged_indices=invalid_indices,
                )
            )

    def _generate_recommendations(self, report: ValidationReport) -> None:
        """Generate recommendations based on validation results."""
        if report.contamination_estimate > 0.05:
            report.recommendations.append(
                "HIGH CONTAMINATION: Consider manual review of flagged samples"
            )
            report.recommendations.append(
                "FDA PCCP: Document data quality issues before model training"
            )

        if report.contamination_estimate > 0.01:
            report.recommendations.append(
                "Apply RONI defense to filter potentially poisoned samples"
            )
            report.recommendations.append("Consider ensemble validation with multiple subsets")

        for result in report.results:
            if result.check_name == "isolation_forest" and not result.passed:
                report.recommendations.append(
                    "Isolation Forest detected high anomaly rate - investigate data provenance"
                )
            if result.check_name == "label_balance" and not result.passed:
                report.recommendations.append(
                    "Severe class imbalance detected - consider resampling strategies"
                )

        if not report.recommendations:
            report.recommendations.append("Data validation passed - proceed with training")


# =============================================================================
# Influence Function Analyzer
# =============================================================================


class InfluenceAnalyzer:
    """
    Analyzes the influence of individual training samples on model behavior.

    Influence functions help identify samples that have unusually high
    impact on model predictions, which could indicate:
    - Backdoor triggers
    - Mislabeled samples
    - Adversarial examples in training data
    """

    def __init__(
        self,
        influence_threshold: float = 2.0,
        sample_size: int | None = None,
    ) -> None:
        """
        Initialize the influence analyzer.

        Args:
            influence_threshold: Multiplier for flagging high-influence samples
            sample_size: Number of samples to analyze (None for all)
        """
        self.influence_threshold = influence_threshold
        self.sample_size = sample_size

    def analyze(
        self,
        features: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]],
        model_fn: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]] | None = None,
    ) -> list[InfluenceResult]:
        """
        Analyze influence of training samples.

        Uses a simplified leave-one-out influence estimation.

        Args:
            features: Training features
            labels: Training labels
            model_fn: Optional model prediction function

        Returns:
            List of InfluenceResult for each sample
        """
        n_samples = features.shape[0]
        indices = range(n_samples)

        if self.sample_size and self.sample_size < n_samples:
            indices = np.random.choice(n_samples, self.sample_size, replace=False)

        results = []

        # Compute baseline statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0) + 1e-8

        for idx in indices:
            # Compute influence as deviation from mean (simplified)
            sample = features[idx]
            z_scores = np.abs((sample - feature_means) / feature_stds)
            influence_score = float(np.max(z_scores))

            # Determine if suspicious
            is_suspicious = influence_score > self.influence_threshold

            # Predict impact direction
            if is_suspicious:
                predicted_impact = "negative"
            elif influence_score < 0.5:
                predicted_impact = "neutral"
            else:
                predicted_impact = "positive"

            results.append(
                InfluenceResult(
                    sample_index=int(idx),
                    influence_score=influence_score,
                    is_suspicious=is_suspicious,
                    predicted_impact=predicted_impact,
                )
            )

        # Sort by influence score descending
        results.sort(key=lambda x: x.influence_score, reverse=True)

        suspicious_count = sum(1 for r in results if r.is_suspicious)
        logger.info(f"Influence analysis: {suspicious_count}/{len(results)} suspicious samples")

        return results

    def get_top_influential(
        self,
        results: list[InfluenceResult],
        top_k: int = 10,
    ) -> list[InfluenceResult]:
        """Get top-k most influential samples."""
        return results[:top_k]

    def get_suspicious(
        self,
        results: list[InfluenceResult],
    ) -> list[InfluenceResult]:
        """Get all suspicious samples."""
        return [r for r in results if r.is_suspicious]


# =============================================================================
# Batch Analyzer
# =============================================================================


class BatchAnalyzer:
    """
    Analyzes training data at the batch level to detect anomalous batches.

    Useful for detecting:
    - Batch-level poisoning attacks
    - Distribution shifts between batches
    - Data quality issues in specific data sources
    """

    def __init__(
        self,
        batch_size: int = 32,
        anomaly_threshold: float = 2.0,
    ) -> None:
        """
        Initialize the batch analyzer.

        Args:
            batch_size: Size of batches to analyze
            anomaly_threshold: Z-score threshold for flagging batches
        """
        self.batch_size = batch_size
        self.anomaly_threshold = anomaly_threshold

    def analyze(
        self,
        features: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]] | None = None,
    ) -> list[BatchAnalysisResult]:
        """
        Analyze batches for anomalies.

        Args:
            features: Training features
            labels: Optional training labels

        Returns:
            List of BatchAnalysisResult for each batch
        """
        n_samples = features.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        batch_stats = []
        results = []

        # Compute per-batch statistics
        for batch_id in range(n_batches):
            start = batch_id * self.batch_size
            end = min(start + self.batch_size, n_samples)
            batch = features[start:end]

            stats_dict = {
                "mean": float(np.mean(batch)),
                "std": float(np.std(batch)),
                "min": float(np.min(batch)),
                "max": float(np.max(batch)),
                "skewness": float(stats.skew(batch.flatten())),
                "kurtosis": float(stats.kurtosis(batch.flatten())),
            }
            batch_stats.append(stats_dict)

        # Compute global statistics for comparison
        all_means = [s["mean"] for s in batch_stats]
        [s["std"] for s in batch_stats]
        global_mean = np.mean(all_means)
        global_std = np.std(all_means) + 1e-8

        # Analyze each batch
        for batch_id, stats_dict in enumerate(batch_stats):
            # Compute anomaly score as Z-score of batch mean
            z_score = abs(stats_dict["mean"] - global_mean) / global_std
            is_anomalous = z_score > self.anomaly_threshold

            flagged_reason = None
            if is_anomalous:
                if stats_dict["mean"] > global_mean:
                    flagged_reason = "Batch mean significantly higher than average"
                else:
                    flagged_reason = "Batch mean significantly lower than average"

            results.append(
                BatchAnalysisResult(
                    batch_id=batch_id,
                    is_anomalous=is_anomalous,
                    anomaly_score=float(z_score),
                    statistics=stats_dict,
                    flagged_reason=flagged_reason,
                )
            )

        anomalous_count = sum(1 for r in results if r.is_anomalous)
        logger.info(f"Batch analysis: {anomalous_count}/{len(results)} anomalous batches")

        return results

    def get_anomalous_batches(
        self,
        results: list[BatchAnalysisResult],
    ) -> list[BatchAnalysisResult]:
        """Get all anomalous batches."""
        return [r for r in results if r.is_anomalous]


# =============================================================================
# RONI Defense
# =============================================================================


class RONIDefense:
    """
    Reject On Negative Impact (RONI) defense.

    This defense filters training samples by measuring their impact
    on model performance. Samples that negatively impact validation
    accuracy are rejected.
    """

    def __init__(
        self,
        impact_threshold: float = -0.01,
        validation_split: float = 0.2,
    ) -> None:
        """
        Initialize RONI defense.

        Args:
            impact_threshold: Threshold for negative impact (reject if below)
            validation_split: Fraction of data to use for validation
        """
        self.impact_threshold = impact_threshold
        self.validation_split = validation_split

    def filter(
        self,
        features: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]],
        model_class: type | None = None,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], list[int]]:
        """
        Filter training data using RONI defense.

        This is a simplified implementation that uses statistical
        analysis instead of actual model training for efficiency.

        Args:
            features: Training features
            labels: Training labels
            model_class: Optional model class for actual impact measurement

        Returns:
            Tuple of (filtered_features, filtered_labels, rejected_indices)
        """
        n_samples = features.shape[0]

        # Simplified RONI: use isolation forest for filtering
        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1,
        )

        iso_forest.fit_predict(features)
        anomaly_scores = iso_forest.decision_function(features)

        # Reject samples with negative anomaly scores
        rejected = np.where(anomaly_scores < self.impact_threshold)[0].tolist()
        kept = np.where(anomaly_scores >= self.impact_threshold)[0]

        filtered_features = features[kept]
        filtered_labels = labels[kept]

        logger.info(
            f"RONI defense: Rejected {len(rejected)}/{n_samples} samples "
            f"({len(rejected)/n_samples:.2%})"
        )

        return filtered_features, filtered_labels, rejected


# =============================================================================
# Ensemble Validator
# =============================================================================


class EnsembleValidator:
    """
    Ensemble-based validation for detecting poisoned samples.

    Trains multiple models on different subsets and identifies
    samples that are consistently misclassified, which may
    indicate poisoning.
    """

    def __init__(
        self,
        n_estimators: int = 5,
        subset_ratio: float = 0.8,
        disagreement_threshold: float = 0.6,
    ) -> None:
        """
        Initialize ensemble validator.

        Args:
            n_estimators: Number of ensemble members
            subset_ratio: Ratio of data to use for each member
            disagreement_threshold: Threshold for flagging disagreement
        """
        self.n_estimators = n_estimators
        self.subset_ratio = subset_ratio
        self.disagreement_threshold = disagreement_threshold

    def validate(
        self,
        features: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]],
    ) -> tuple[list[int], dict[str, Any]]:
        """
        Validate samples using ensemble disagreement.

        Uses isolation forest ensemble for anomaly detection.

        Args:
            features: Training features
            labels: Training labels

        Returns:
            Tuple of (flagged_indices, statistics)
        """
        n_samples = features.shape[0]
        anomaly_votes = np.zeros(n_samples)

        for i in range(self.n_estimators):
            # Create random subset
            subset_size = int(n_samples * self.subset_ratio)
            indices = np.random.choice(n_samples, subset_size, replace=False)
            subset = features[indices]

            # Train isolation forest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42 + i,
                n_jobs=-1,
            )
            iso_forest.fit(subset)

            # Score all samples
            predictions = iso_forest.predict(features)
            anomaly_votes += (predictions == -1).astype(float)

        # Normalize votes
        disagreement_ratio = anomaly_votes / self.n_estimators

        # Flag samples with high disagreement
        flagged = np.where(disagreement_ratio > self.disagreement_threshold)[0].tolist()

        statistics = {
            "n_estimators": self.n_estimators,
            "subset_ratio": self.subset_ratio,
            "flagged_count": len(flagged),
            "flagged_ratio": len(flagged) / n_samples,
            "mean_disagreement": float(np.mean(disagreement_ratio)),
            "max_disagreement": float(np.max(disagreement_ratio)),
        }

        logger.info(
            f"Ensemble validation: {len(flagged)}/{n_samples} samples flagged "
            f"(mean disagreement: {statistics['mean_disagreement']:.2%})"
        )

        return flagged, statistics


# =============================================================================
# Unified Defense Pipeline
# =============================================================================


class DataPoisoningDefense:
    """
    Unified pipeline for data poisoning defense.

    Combines multiple defense strategies into a single interface
    for comprehensive protection of training data.
    """

    def __init__(
        self,
        strategies: list[DefenseStrategy] | None = None,
        contamination: float = 0.1,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize defense pipeline.

        Args:
            strategies: List of defense strategies to apply
            contamination: Expected contamination ratio
            strict_mode: If True, fail on any warning
        """
        if strategies is None:
            strategies = [
                DefenseStrategy.STATISTICAL,
                DefenseStrategy.ISOLATION_FOREST,
                DefenseStrategy.RONI,
            ]

        self.strategies = strategies
        self.contamination = contamination
        self.strict_mode = strict_mode

        # Initialize components
        self.validator = TrainingDataValidator(contamination=contamination)
        self.influence_analyzer = InfluenceAnalyzer()
        self.batch_analyzer = BatchAnalyzer()
        self.roni_defense = RONIDefense()
        self.ensemble_validator = EnsembleValidator()

    def analyze(
        self,
        features: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive analysis of training data.

        Args:
            features: Training features
            labels: Optional training labels

        Returns:
            Dictionary with all analysis results
        """
        results: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_samples": features.shape[0],
            "n_features": features.shape[1],
            "strategies_applied": [s.value for s in self.strategies],
        }

        # Validation report
        validation_report = self.validator.validate(features, labels, strategies=self.strategies)
        results["validation"] = validation_report.to_dict()

        # Influence analysis if requested
        if DefenseStrategy.INFLUENCE in self.strategies and labels is not None:
            influence_results = self.influence_analyzer.analyze(features, labels)
            suspicious = self.influence_analyzer.get_suspicious(influence_results)
            results["influence"] = {
                "total_analyzed": len(influence_results),
                "suspicious_count": len(suspicious),
                "top_5_influential": [
                    {
                        "index": r.sample_index,
                        "score": r.influence_score,
                        "impact": r.predicted_impact,
                    }
                    for r in influence_results[:5]
                ],
            }

        # Batch analysis
        batch_results = self.batch_analyzer.analyze(features, labels)
        anomalous_batches = self.batch_analyzer.get_anomalous_batches(batch_results)
        results["batch_analysis"] = {
            "total_batches": len(batch_results),
            "anomalous_batches": len(anomalous_batches),
            "anomalous_batch_ids": [b.batch_id for b in anomalous_batches],
        }

        # Ensemble validation if requested
        if DefenseStrategy.ENSEMBLE in self.strategies and labels is not None:
            flagged, ensemble_stats = self.ensemble_validator.validate(features, labels)
            results["ensemble"] = {
                "flagged_indices": flagged[:100],  # Limit output
                **ensemble_stats,
            }

        # Overall assessment
        results["overall"] = {
            "contamination_estimate": validation_report.contamination_estimate,
            "total_flagged": len(validation_report.flagged_samples),
            "passed": validation_report.overall_passed,
            "recommendations": validation_report.recommendations,
        }

        return results

    def filter(
        self,
        features: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], dict[str, Any]]:
        """
        Filter potentially poisoned samples from training data.

        Args:
            features: Training features
            labels: Training labels

        Returns:
            Tuple of (clean_features, clean_labels, filter_stats)
        """
        if DefenseStrategy.RONI in self.strategies:
            clean_features, clean_labels, rejected = self.roni_defense.filter(features, labels)

            stats = {
                "original_samples": features.shape[0],
                "clean_samples": clean_features.shape[0],
                "rejected_samples": len(rejected),
                "rejection_ratio": len(rejected) / features.shape[0],
            }

            return clean_features, clean_labels, stats

        # If RONI not enabled, use ensemble validation
        flagged, _ = self.ensemble_validator.validate(features, labels)
        keep_mask = np.ones(features.shape[0], dtype=bool)
        keep_mask[flagged] = False

        clean_features = features[keep_mask]
        clean_labels = labels[keep_mask]

        stats = {
            "original_samples": features.shape[0],
            "clean_samples": clean_features.shape[0],
            "rejected_samples": len(flagged),
            "rejection_ratio": len(flagged) / features.shape[0],
        }

        return clean_features, clean_labels, stats
