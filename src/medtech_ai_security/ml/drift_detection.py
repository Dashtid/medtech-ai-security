"""
Model Drift Detection for Medical AI Systems.

This module provides statistical methods for detecting distribution drift
in ML model inputs and outputs, critical for maintaining FDA compliance
and ensuring continued safety of medical AI systems.

Key features:
- Statistical drift detection (KL divergence, PSI, Wasserstein)
- Feature-level drift monitoring
- Configurable alert thresholds
- HSCC 2026 guidance alignment for drift exploitation prevention

References:
- HSCC 2026 AI Cybersecurity Guidance (draft)
- FDA PCCP Guidance for AI/ML changes
- NIST AI 100-2 E2025 on model manipulation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp, wasserstein_distance

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class DriftType(str, Enum):
    """Types of drift that can be detected."""

    DATA_DRIFT = "data_drift"  # Input distribution shift
    CONCEPT_DRIFT = "concept_drift"  # P(Y|X) change
    PREDICTION_DRIFT = "prediction_drift"  # Output distribution shift
    FEATURE_DRIFT = "feature_drift"  # Individual feature shift


class DriftSeverity(str, Enum):
    """Severity levels for drift alerts."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftMethod(str, Enum):
    """Statistical methods for drift detection."""

    KL_DIVERGENCE = "kl_divergence"
    JS_DIVERGENCE = "js_divergence"  # Jensen-Shannon (symmetric)
    PSI = "psi"  # Population Stability Index
    WASSERSTEIN = "wasserstein"
    KS_TEST = "ks_test"  # Kolmogorov-Smirnov
    CHI_SQUARE = "chi_square"


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift severity classification."""

    low: float = 0.1
    medium: float = 0.2
    high: float = 0.3
    critical: float = 0.5

    def classify(self, score: float) -> DriftSeverity:
        """Classify drift severity based on score."""
        if score < self.low:
            return DriftSeverity.NONE
        elif score < self.medium:
            return DriftSeverity.LOW
        elif score < self.high:
            return DriftSeverity.MEDIUM
        elif score < self.critical:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


@dataclass
class DriftResult:
    """Result of a drift detection analysis."""

    drift_detected: bool
    drift_type: DriftType
    method: DriftMethod
    score: float
    severity: DriftSeverity
    p_value: float | None = None
    feature_name: str | None = None
    reference_stats: dict[str, float] = field(default_factory=dict)
    current_stats: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type.value,
            "method": self.method.value,
            "score": self.score,
            "severity": self.severity.value,
            "p_value": self.p_value,
            "feature_name": self.feature_name,
            "reference_stats": self.reference_stats,
            "current_stats": self.current_stats,
            "timestamp": self.timestamp,
            "recommendations": self.recommendations,
        }


@dataclass
class DriftReport:
    """Comprehensive drift analysis report."""

    overall_drift_detected: bool
    overall_severity: DriftSeverity
    feature_results: list[DriftResult]
    prediction_drift: DriftResult | None
    summary: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_drift_detected": self.overall_drift_detected,
            "overall_severity": self.overall_severity.value,
            "feature_results": [r.to_dict() for r in self.feature_results],
            "prediction_drift": (
                self.prediction_drift.to_dict() if self.prediction_drift else None
            ),
            "summary": self.summary,
            "timestamp": self.timestamp,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Statistical Drift Detection Methods
# =============================================================================


def compute_kl_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 50,
) -> float:
    """
    Compute KL divergence between reference and current distributions.

    KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))

    Note: KL divergence is asymmetric. We use reference as P and current as Q.

    Args:
        reference: Reference (baseline) distribution samples
        current: Current distribution samples
        num_bins: Number of histogram bins

    Returns:
        KL divergence score (0 = identical, higher = more different)
    """
    # Create histograms with same bins
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)

    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    ref_hist = ref_hist + epsilon
    cur_hist = cur_hist + epsilon

    # Normalize
    ref_hist = ref_hist / ref_hist.sum()
    cur_hist = cur_hist / cur_hist.sum()

    return float(entropy(ref_hist, cur_hist))


def compute_js_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 50,
) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric version of KL).

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)

    Args:
        reference: Reference distribution samples
        current: Current distribution samples
        num_bins: Number of histogram bins

    Returns:
        JS divergence score (0 = identical, 1 = maximally different)
    """
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)

    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)

    # Normalize
    ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
    cur_hist = cur_hist / (cur_hist.sum() + 1e-10)

    return float(jensenshannon(ref_hist, cur_hist))


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI is commonly used in credit risk modeling to detect distribution shift.
    PSI = sum((current% - reference%) * ln(current% / reference%))

    Interpretation:
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.2: Moderate shift, investigation needed
    - PSI >= 0.2: Significant shift, action required

    Args:
        reference: Reference distribution samples
        current: Current distribution samples
        num_bins: Number of bins

    Returns:
        PSI score
    """
    # Create bins from reference distribution
    _, bin_edges = np.histogram(reference, bins=num_bins)

    # Calculate bin percentages
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to percentages
    ref_pct = ref_counts / len(reference) + 1e-10
    cur_pct = cur_counts / len(current) + 1e-10

    # Calculate PSI
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

    return float(psi)


def compute_wasserstein(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """
    Compute Wasserstein distance (Earth Mover's Distance).

    Measures the minimum "work" required to transform one distribution
    into another. More robust to outliers than KL divergence.

    Args:
        reference: Reference distribution samples
        current: Current distribution samples

    Returns:
        Wasserstein distance
    """
    return float(wasserstein_distance(reference.flatten(), current.flatten()))


def compute_ks_test(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test for distribution difference.

    Tests the null hypothesis that two samples are drawn from the same
    continuous distribution.

    Args:
        reference: Reference distribution samples
        current: Current distribution samples

    Returns:
        Tuple of (KS statistic, p-value)
    """
    statistic, p_value = ks_2samp(reference.flatten(), current.flatten())
    return float(statistic), float(p_value)


# =============================================================================
# Drift Detector Class
# =============================================================================


class DriftDetector:
    """
    Comprehensive drift detection for ML model monitoring.

    Supports multiple statistical methods and provides actionable
    recommendations for medical AI systems.

    Example:
        >>> detector = DriftDetector()
        >>> detector.set_reference(training_features, training_predictions)
        >>> report = detector.detect_drift(new_features, new_predictions)
        >>> if report.overall_drift_detected:
        ...     print(f"Drift detected: {report.overall_severity}")
    """

    def __init__(
        self,
        methods: list[DriftMethod] | None = None,
        thresholds: DriftThresholds | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize drift detector.

        Args:
            methods: Statistical methods to use (default: PSI, JS, Wasserstein)
            thresholds: Drift severity thresholds
            feature_names: Names for input features
        """
        self.methods = methods or [
            DriftMethod.PSI,
            DriftMethod.JS_DIVERGENCE,
            DriftMethod.WASSERSTEIN,
        ]
        self.thresholds = thresholds or DriftThresholds()
        self.feature_names = feature_names

        # Reference distributions
        self.reference_features: np.ndarray | None = None
        self.reference_predictions: np.ndarray | None = None
        self.reference_stats: dict[str, dict[str, float]] = {}

    def set_reference(
        self,
        features: np.ndarray,
        predictions: np.ndarray | None = None,
    ) -> None:
        """
        Set reference (baseline) distributions from training data.

        Args:
            features: Training feature matrix (n_samples, n_features)
            predictions: Model predictions on training data (optional)
        """
        self.reference_features = np.asarray(features)
        if predictions is not None:
            self.reference_predictions = np.asarray(predictions)

        # Compute reference statistics
        self._compute_reference_stats()
        logger.info(
            f"Reference set with {len(features)} samples, "
            f"{features.shape[1] if len(features.shape) > 1 else 1} features"
        )

    def _compute_reference_stats(self) -> None:
        """Compute summary statistics for reference distributions."""
        if self.reference_features is None:
            return

        features = self.reference_features
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)

        n_features = features.shape[1]
        names = self.feature_names or [f"feature_{i}" for i in range(n_features)]

        for i, name in enumerate(names):
            col = features[:, i]
            self.reference_stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "median": float(np.median(col)),
                "q25": float(np.percentile(col, 25)),
                "q75": float(np.percentile(col, 75)),
            }

    def detect_drift(
        self,
        current_features: np.ndarray,
        current_predictions: np.ndarray | None = None,
        method: DriftMethod | None = None,
    ) -> DriftReport:
        """
        Detect drift between reference and current distributions.

        Args:
            current_features: Current feature matrix
            current_predictions: Current model predictions (optional)
            method: Specific method to use (default: all configured methods)

        Returns:
            Comprehensive drift report
        """
        if self.reference_features is None:
            raise ValueError("Reference not set. Call set_reference() first.")

        current_features = np.asarray(current_features)
        if len(current_features.shape) == 1:
            current_features = current_features.reshape(-1, 1)

        methods_to_use = [method] if method else self.methods
        feature_results: list[DriftResult] = []
        prediction_drift: DriftResult | None = None

        # Detect feature drift
        n_features = min(
            self.reference_features.shape[1] if len(self.reference_features.shape) > 1 else 1,
            current_features.shape[1] if len(current_features.shape) > 1 else 1,
        )
        names = self.feature_names or [f"feature_{i}" for i in range(n_features)]

        for i, name in enumerate(names[:n_features]):
            ref_col = (
                self.reference_features[:, i]
                if len(self.reference_features.shape) > 1
                else self.reference_features
            )
            cur_col = (
                current_features[:, i] if len(current_features.shape) > 1 else current_features
            )

            for m in methods_to_use:
                result = self._detect_single_drift(
                    ref_col, cur_col, m, DriftType.FEATURE_DRIFT, name
                )
                feature_results.append(result)

        # Detect prediction drift
        if current_predictions is not None and self.reference_predictions is not None:
            current_predictions = np.asarray(current_predictions)
            for m in methods_to_use:
                prediction_drift = self._detect_single_drift(
                    self.reference_predictions.flatten(),
                    current_predictions.flatten(),
                    m,
                    DriftType.PREDICTION_DRIFT,
                    "predictions",
                )
                break  # Use first method for predictions

        # Generate report
        return self._generate_report(feature_results, prediction_drift)

    def _detect_single_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        method: DriftMethod,
        drift_type: DriftType,
        feature_name: str,
    ) -> DriftResult:
        """Detect drift for a single feature using specified method."""
        score = 0.0
        p_value = None

        if method == DriftMethod.KL_DIVERGENCE:
            score = compute_kl_divergence(reference, current)
        elif method == DriftMethod.JS_DIVERGENCE:
            score = compute_js_divergence(reference, current)
        elif method == DriftMethod.PSI:
            score = compute_psi(reference, current)
        elif method == DriftMethod.WASSERSTEIN:
            # Normalize Wasserstein by reference std for comparability
            raw_wasserstein = compute_wasserstein(reference, current)
            ref_std = np.std(reference) + 1e-10
            score = raw_wasserstein / ref_std
        elif method == DriftMethod.KS_TEST:
            ks_stat, p_value = compute_ks_test(reference, current)
            score = ks_stat

        severity = self.thresholds.classify(score)
        drift_detected = severity != DriftSeverity.NONE

        # Current statistics
        current_stats = {
            "mean": float(np.mean(current)),
            "std": float(np.std(current)),
            "min": float(np.min(current)),
            "max": float(np.max(current)),
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            drift_detected, severity, drift_type, feature_name, score
        )

        return DriftResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            method=method,
            score=round(score, 6),
            severity=severity,
            p_value=round(p_value, 6) if p_value is not None else None,
            feature_name=feature_name,
            reference_stats=self.reference_stats.get(feature_name, {}),
            current_stats=current_stats,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        drift_detected: bool,
        severity: DriftSeverity,
        drift_type: DriftType,
        feature_name: str,
        score: float,
    ) -> list[str]:
        """Generate actionable recommendations based on drift detection."""
        recommendations = []

        if not drift_detected:
            return recommendations

        if severity == DriftSeverity.LOW:
            recommendations.append(
                f"Monitor {feature_name} - low drift detected (score: {score:.4f})"
            )
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append(f"Investigate {feature_name} - moderate drift detected")
            recommendations.append("Consider collecting more recent training data")
        elif severity == DriftSeverity.HIGH:
            recommendations.append(f"ACTION REQUIRED: High drift in {feature_name}")
            recommendations.append("Evaluate model performance on recent data")
            recommendations.append("Consider model retraining with updated distribution")
        elif severity == DriftSeverity.CRITICAL:
            recommendations.append(
                f"CRITICAL: Severe drift in {feature_name} - model reliability compromised"
            )
            recommendations.append("Immediate model validation required")
            recommendations.append("FDA PCCP: Document drift detection and mitigation steps")
            recommendations.append("Consider halting automated predictions until validated")

        return recommendations

    def _generate_report(
        self,
        feature_results: list[DriftResult],
        prediction_drift: DriftResult | None,
    ) -> DriftReport:
        """Generate comprehensive drift report."""
        # Determine overall drift
        all_results = feature_results.copy()
        if prediction_drift:
            all_results.append(prediction_drift)

        drift_severities = [r.severity for r in all_results if r.drift_detected]

        if not drift_severities:
            overall_severity = DriftSeverity.NONE
            overall_drift = False
        else:
            # Use maximum severity
            severity_order = [
                DriftSeverity.NONE,
                DriftSeverity.LOW,
                DriftSeverity.MEDIUM,
                DriftSeverity.HIGH,
                DriftSeverity.CRITICAL,
            ]
            max_severity = max(drift_severities, key=lambda s: severity_order.index(s))
            overall_severity = max_severity
            overall_drift = True

        # Summary statistics
        summary = {
            "total_features_analyzed": len(feature_results),
            "features_with_drift": sum(1 for r in feature_results if r.drift_detected),
            "prediction_drift_detected": (
                prediction_drift.drift_detected if prediction_drift else False
            ),
            "methods_used": list({r.method.value for r in all_results}),
            "severity_counts": {
                s.value: sum(1 for r in all_results if r.severity == s) for s in DriftSeverity
            },
        }

        # Aggregate recommendations
        all_recommendations = []
        for r in all_results:
            all_recommendations.extend(r.recommendations)

        # Add overall recommendations
        if overall_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            all_recommendations.append(
                "HSCC 2026 Guidance: Drift exploitation is a key AI security threat"
            )
            all_recommendations.append("Consider implementing automated drift monitoring alerts")

        return DriftReport(
            overall_drift_detected=overall_drift,
            overall_severity=overall_severity,
            feature_results=feature_results,
            prediction_drift=prediction_drift,
            summary=summary,
            recommendations=list(set(all_recommendations)),  # Deduplicate
        )

    def quick_check(
        self,
        current_features: np.ndarray,
        method: DriftMethod = DriftMethod.PSI,
    ) -> tuple[bool, float]:
        """
        Quick drift check returning just boolean and score.

        Useful for real-time monitoring pipelines.

        Args:
            current_features: Current feature matrix
            method: Statistical method to use

        Returns:
            Tuple of (drift_detected, score)
        """
        if self.reference_features is None:
            raise ValueError("Reference not set. Call set_reference() first.")

        ref_flat = self.reference_features.flatten()
        cur_flat = np.asarray(current_features).flatten()

        if method == DriftMethod.PSI:
            score = compute_psi(ref_flat, cur_flat)
        elif method == DriftMethod.JS_DIVERGENCE:
            score = compute_js_divergence(ref_flat, cur_flat)
        elif method == DriftMethod.WASSERSTEIN:
            score = compute_wasserstein(ref_flat, cur_flat) / (np.std(ref_flat) + 1e-10)
        else:
            score = compute_psi(ref_flat, cur_flat)

        severity = self.thresholds.classify(score)
        return severity != DriftSeverity.NONE, score
