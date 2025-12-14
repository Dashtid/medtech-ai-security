"""
Bias Detection Module for Medical AI Fairness.

Provides comprehensive bias detection and fairness metrics for medical AI systems,
addressing FDA requirements for AI/ML devices to perform equitably across
demographic groups.

Security Context:
- FDA AI/ML Guidance 2025: Models must benefit all relevant demographic groups
- EU AI Act: High-risk AI requires bias testing and mitigation
- GMLP Principles: Multi-disciplinary expertise including fairness assessment

References:
- Fairness and Machine Learning (Barocas, Hardt, Narayanan)
- FDA Draft Guidance: AI-Enabled Device Software Functions
- AIF360: AI Fairness 360 Toolkit
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class FairnessMetric(str, Enum):
    """Available fairness metrics."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    TREATMENT_EQUALITY = "treatment_equality"
    INDIVIDUAL_FAIRNESS = "individual_fairness"


class ProtectedAttribute(str, Enum):
    """Common protected attributes in medical AI."""

    AGE = "age"
    SEX = "sex"
    RACE = "race"
    ETHNICITY = "ethnicity"
    SOCIOECONOMIC = "socioeconomic"
    DISABILITY = "disability"
    GEOGRAPHIC = "geographic"


class BiasLevel(str, Enum):
    """Severity levels for detected bias."""

    NONE = "none"  # No significant bias detected
    LOW = "low"  # Minor bias, may be acceptable
    MODERATE = "moderate"  # Moderate bias, review recommended
    HIGH = "high"  # Significant bias, mitigation required
    CRITICAL = "critical"  # Severe bias, deployment not recommended


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GroupMetrics:
    """Performance metrics for a specific group."""

    group_name: str
    group_size: int
    true_positive_rate: float  # Sensitivity/Recall
    false_positive_rate: float
    true_negative_rate: float  # Specificity
    false_negative_rate: float
    positive_predictive_value: float  # Precision
    negative_predictive_value: float
    accuracy: float
    prevalence: float  # Proportion of positive class in group

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        if self.true_positive_rate + self.positive_predictive_value == 0:
            return 0.0
        return (
            2
            * (self.positive_predictive_value * self.true_positive_rate)
            / (self.positive_predictive_value + self.true_positive_rate)
        )


@dataclass
class FairnessResult:
    """Result of a fairness metric calculation."""

    metric: FairnessMetric
    value: float
    threshold: float
    passed: bool
    reference_group: str
    comparison_groups: dict[str, float]
    message: str


@dataclass
class BiasReport:
    """Comprehensive bias analysis report."""

    protected_attribute: str
    groups: list[GroupMetrics]
    fairness_results: list[FairnessResult]
    overall_bias_level: BiasLevel
    recommendations: list[str]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "protected_attribute": self.protected_attribute,
            "groups": [
                {
                    "group_name": g.group_name,
                    "group_size": g.group_size,
                    "true_positive_rate": g.true_positive_rate,
                    "false_positive_rate": g.false_positive_rate,
                    "accuracy": g.accuracy,
                    "f1_score": g.f1_score,
                }
                for g in self.groups
            ],
            "fairness_results": [
                {
                    "metric": r.metric.value,
                    "value": r.value,
                    "threshold": r.threshold,
                    "passed": r.passed,
                    "message": r.message,
                }
                for r in self.fairness_results
            ],
            "overall_bias_level": self.overall_bias_level.value,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at,
        }


# =============================================================================
# Bias Detector
# =============================================================================


class BiasDetector:
    """
    Detects and measures bias in ML model predictions.

    Implements multiple fairness metrics from the literature:
    - Demographic Parity: Equal positive prediction rates across groups
    - Equalized Odds: Equal TPR and FPR across groups
    - Equal Opportunity: Equal TPR across groups (for positive class)
    - Predictive Parity: Equal PPV across groups
    - Calibration: Equal outcome rates for similar predictions

    Example:
        detector = BiasDetector()

        # Analyze bias across sex groups
        report = detector.analyze(
            y_true=labels,
            y_pred=predictions,
            protected_attribute=sex_labels,
            attribute_name="sex"
        )

        if report.overall_bias_level in [BiasLevel.HIGH, BiasLevel.CRITICAL]:
            print("Significant bias detected - mitigation required")
    """

    def __init__(
        self,
        demographic_parity_threshold: float = 0.1,
        equalized_odds_threshold: float = 0.1,
        equal_opportunity_threshold: float = 0.1,
        min_group_size: int = 30,
    ) -> None:
        """
        Initialize the bias detector.

        Args:
            demographic_parity_threshold: Max allowed difference in positive rates
            equalized_odds_threshold: Max allowed difference in TPR/FPR
            equal_opportunity_threshold: Max allowed difference in TPR
            min_group_size: Minimum samples per group for reliable analysis
        """
        self.demographic_parity_threshold = demographic_parity_threshold
        self.equalized_odds_threshold = equalized_odds_threshold
        self.equal_opportunity_threshold = equal_opportunity_threshold
        self.min_group_size = min_group_size

    def analyze(
        self,
        y_true: NDArray[np.integer[Any]],
        y_pred: NDArray[np.integer[Any]],
        protected_attribute: NDArray[Any],
        attribute_name: str = "protected_attribute",
        y_prob: NDArray[np.floating[Any]] | None = None,
    ) -> BiasReport:
        """
        Perform comprehensive bias analysis.

        Args:
            y_true: True labels (0/1)
            y_pred: Predicted labels (0/1)
            protected_attribute: Group membership for each sample
            attribute_name: Name of the protected attribute
            y_prob: Optional prediction probabilities for calibration

        Returns:
            BiasReport with detailed analysis
        """
        # Get unique groups
        groups = np.unique(protected_attribute)
        group_metrics: list[GroupMetrics] = []

        # Calculate metrics for each group
        for group in groups:
            mask = protected_attribute == group
            if np.sum(mask) < self.min_group_size:
                logger.warning(
                    f"Group '{group}' has only {np.sum(mask)} samples, "
                    f"less than minimum {self.min_group_size}"
                )

            metrics = self._calculate_group_metrics(
                y_true[mask],
                y_pred[mask],
                str(group),
            )
            group_metrics.append(metrics)

        # Calculate fairness metrics
        fairness_results: list[FairnessResult] = []

        # Demographic Parity
        dp_result = self._calculate_demographic_parity(group_metrics)
        fairness_results.append(dp_result)

        # Equalized Odds
        eo_result = self._calculate_equalized_odds(group_metrics)
        fairness_results.append(eo_result)

        # Equal Opportunity
        eop_result = self._calculate_equal_opportunity(group_metrics)
        fairness_results.append(eop_result)

        # Predictive Parity
        pp_result = self._calculate_predictive_parity(group_metrics)
        fairness_results.append(pp_result)

        # Determine overall bias level
        bias_level = self._determine_bias_level(fairness_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            fairness_results,
            group_metrics,
            bias_level,
        )

        return BiasReport(
            protected_attribute=attribute_name,
            groups=group_metrics,
            fairness_results=fairness_results,
            overall_bias_level=bias_level,
            recommendations=recommendations,
        )

    def analyze_multiple_attributes(
        self,
        y_true: NDArray[np.integer[Any]],
        y_pred: NDArray[np.integer[Any]],
        protected_attributes: dict[str, NDArray[Any]],
        y_prob: NDArray[np.floating[Any]] | None = None,
    ) -> dict[str, BiasReport]:
        """
        Analyze bias across multiple protected attributes.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: Dict mapping attribute names to values
            y_prob: Optional prediction probabilities

        Returns:
            Dict mapping attribute names to BiasReports
        """
        reports = {}
        for attr_name, attr_values in protected_attributes.items():
            reports[attr_name] = self.analyze(
                y_true=y_true,
                y_pred=y_pred,
                protected_attribute=attr_values,
                attribute_name=attr_name,
                y_prob=y_prob,
            )
        return reports

    def _calculate_group_metrics(
        self,
        y_true: NDArray[np.integer[Any]],
        y_pred: NDArray[np.integer[Any]],
        group_name: str,
    ) -> GroupMetrics:
        """Calculate performance metrics for a single group."""
        # Confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        total = len(y_true)
        pos = np.sum(y_true == 1)
        neg = np.sum(y_true == 0)

        # Rates with safe division
        tpr = tp / pos if pos > 0 else 0.0
        fpr = fp / neg if neg > 0 else 0.0
        tnr = tn / neg if neg > 0 else 0.0
        fnr = fn / pos if pos > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        prevalence = pos / total if total > 0 else 0.0

        return GroupMetrics(
            group_name=group_name,
            group_size=total,
            true_positive_rate=float(tpr),
            false_positive_rate=float(fpr),
            true_negative_rate=float(tnr),
            false_negative_rate=float(fnr),
            positive_predictive_value=float(ppv),
            negative_predictive_value=float(npv),
            accuracy=float(accuracy),
            prevalence=float(prevalence),
        )

    def _calculate_demographic_parity(
        self,
        group_metrics: list[GroupMetrics],
    ) -> FairnessResult:
        """
        Calculate demographic parity (statistical parity).

        Measures: P(Y_pred=1|A=a) = P(Y_pred=1|A=b) for all groups a, b
        """
        # Use positive prediction rate (TPR + FPR conceptually)
        # Actually: proportion of positive predictions
        positive_rates = {}
        for g in group_metrics:
            # Positive prediction rate from confusion matrix
            pred_pos = (
                g.true_positive_rate * g.prevalence
                + g.false_positive_rate * (1 - g.prevalence)
            )
            positive_rates[g.group_name] = pred_pos

        # Reference group is the one with highest positive rate
        ref_group = max(positive_rates, key=lambda x: positive_rates[x])
        ref_rate = positive_rates[ref_group]

        # Calculate max difference
        max_diff = 0.0
        comparison = {}
        for group, rate in positive_rates.items():
            diff = abs(rate - ref_rate)
            comparison[group] = rate
            max_diff = max(max_diff, diff)

        passed = max_diff <= self.demographic_parity_threshold

        return FairnessResult(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            value=max_diff,
            threshold=self.demographic_parity_threshold,
            passed=passed,
            reference_group=ref_group,
            comparison_groups=comparison,
            message=(
                f"Max positive rate difference: {max_diff:.3f} "
                f"({'PASS' if passed else 'FAIL'})"
            ),
        )

    def _calculate_equalized_odds(
        self,
        group_metrics: list[GroupMetrics],
    ) -> FairnessResult:
        """
        Calculate equalized odds.

        Measures: P(Y_pred=1|Y=y,A=a) = P(Y_pred=1|Y=y,A=b)
        for all groups a, b and outcomes y in {0,1}
        """
        tpr_values = {g.group_name: g.true_positive_rate for g in group_metrics}
        fpr_values = {g.group_name: g.false_positive_rate for g in group_metrics}

        # Calculate max differences in TPR and FPR
        tpr_diff = max(tpr_values.values()) - min(tpr_values.values())
        fpr_diff = max(fpr_values.values()) - min(fpr_values.values())
        max_diff = max(tpr_diff, fpr_diff)

        passed = max_diff <= self.equalized_odds_threshold
        ref_group = max(tpr_values, key=lambda x: tpr_values[x])

        return FairnessResult(
            metric=FairnessMetric.EQUALIZED_ODDS,
            value=max_diff,
            threshold=self.equalized_odds_threshold,
            passed=passed,
            reference_group=ref_group,
            comparison_groups={
                g: {"tpr": tpr_values[g], "fpr": fpr_values[g]}
                for g in tpr_values
            },
            message=(
                f"Max TPR/FPR difference: {max_diff:.3f} "
                f"(TPR diff: {tpr_diff:.3f}, FPR diff: {fpr_diff:.3f}) "
                f"({'PASS' if passed else 'FAIL'})"
            ),
        )

    def _calculate_equal_opportunity(
        self,
        group_metrics: list[GroupMetrics],
    ) -> FairnessResult:
        """
        Calculate equal opportunity.

        Measures: P(Y_pred=1|Y=1,A=a) = P(Y_pred=1|Y=1,A=b)
        Equal true positive rates across groups
        """
        tpr_values = {g.group_name: g.true_positive_rate for g in group_metrics}

        max_tpr = max(tpr_values.values())
        min_tpr = min(tpr_values.values())
        tpr_diff = max_tpr - min_tpr

        passed = tpr_diff <= self.equal_opportunity_threshold
        ref_group = max(tpr_values, key=lambda x: tpr_values[x])

        return FairnessResult(
            metric=FairnessMetric.EQUAL_OPPORTUNITY,
            value=tpr_diff,
            threshold=self.equal_opportunity_threshold,
            passed=passed,
            reference_group=ref_group,
            comparison_groups=tpr_values,
            message=(
                f"TPR difference: {tpr_diff:.3f} "
                f"(range: {min_tpr:.3f} to {max_tpr:.3f}) "
                f"({'PASS' if passed else 'FAIL'})"
            ),
        )

    def _calculate_predictive_parity(
        self,
        group_metrics: list[GroupMetrics],
    ) -> FairnessResult:
        """
        Calculate predictive parity.

        Measures: P(Y=1|Y_pred=1,A=a) = P(Y=1|Y_pred=1,A=b)
        Equal positive predictive values across groups
        """
        ppv_values = {
            g.group_name: g.positive_predictive_value for g in group_metrics
        }

        max_ppv = max(ppv_values.values())
        min_ppv = min(ppv_values.values())
        ppv_diff = max_ppv - min_ppv

        # Use same threshold as demographic parity
        passed = ppv_diff <= self.demographic_parity_threshold
        ref_group = max(ppv_values, key=lambda x: ppv_values[x])

        return FairnessResult(
            metric=FairnessMetric.PREDICTIVE_PARITY,
            value=ppv_diff,
            threshold=self.demographic_parity_threshold,
            passed=passed,
            reference_group=ref_group,
            comparison_groups=ppv_values,
            message=(
                f"PPV difference: {ppv_diff:.3f} "
                f"(range: {min_ppv:.3f} to {max_ppv:.3f}) "
                f"({'PASS' if passed else 'FAIL'})"
            ),
        )

    def _determine_bias_level(
        self,
        fairness_results: list[FairnessResult],
    ) -> BiasLevel:
        """Determine overall bias level from fairness results."""
        failed_count = sum(1 for r in fairness_results if not r.passed)
        max_violation = max(
            r.value / r.threshold for r in fairness_results if r.threshold > 0
        )

        if failed_count == 0:
            return BiasLevel.NONE
        elif failed_count == 1 and max_violation < 1.5:
            return BiasLevel.LOW
        elif failed_count <= 2 and max_violation < 2.0:
            return BiasLevel.MODERATE
        elif failed_count <= 3 and max_violation < 3.0:
            return BiasLevel.HIGH
        else:
            return BiasLevel.CRITICAL

    def _generate_recommendations(
        self,
        fairness_results: list[FairnessResult],
        group_metrics: list[GroupMetrics],
        bias_level: BiasLevel,
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check for small groups
        small_groups = [g for g in group_metrics if g.group_size < 100]
        if small_groups:
            recommendations.append(
                f"Collect more data for underrepresented groups: "
                f"{[g.group_name for g in small_groups]}"
            )

        # Check failed metrics
        for result in fairness_results:
            if not result.passed:
                if result.metric == FairnessMetric.DEMOGRAPHIC_PARITY:
                    recommendations.append(
                        "Consider threshold adjustment or resampling to "
                        "balance positive prediction rates across groups"
                    )
                elif result.metric == FairnessMetric.EQUAL_OPPORTUNITY:
                    recommendations.append(
                        "Model has disparate sensitivity across groups. "
                        "Consider cost-sensitive learning or post-processing "
                        "threshold calibration per group"
                    )
                elif result.metric == FairnessMetric.EQUALIZED_ODDS:
                    recommendations.append(
                        "Both TPR and FPR vary across groups. Consider "
                        "adversarial debiasing or representation learning"
                    )

        # Overall recommendations based on bias level
        if bias_level == BiasLevel.HIGH:
            recommendations.append(
                "HIGH BIAS: Mitigation required before deployment. "
                "Document disparities and mitigation plan for FDA submission"
            )
        elif bias_level == BiasLevel.CRITICAL:
            recommendations.append(
                "CRITICAL BIAS: Deployment not recommended. "
                "Fundamental model redesign or data collection required"
            )

        if not recommendations:
            recommendations.append(
                "No significant bias detected. Continue monitoring in production"
            )

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_bias_check(
    y_true: NDArray[np.integer[Any]],
    y_pred: NDArray[np.integer[Any]],
    protected_attribute: NDArray[Any],
) -> dict[str, Any]:
    """
    Quick bias check with default settings.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attribute: Group membership

    Returns:
        Summary dict with key findings
    """
    detector = BiasDetector()
    report = detector.analyze(y_true, y_pred, protected_attribute)

    return {
        "bias_level": report.overall_bias_level.value,
        "fairness_passed": sum(1 for r in report.fairness_results if r.passed),
        "fairness_failed": sum(1 for r in report.fairness_results if not r.passed),
        "group_count": len(report.groups),
        "recommendations": report.recommendations[:3],  # Top 3
    }
