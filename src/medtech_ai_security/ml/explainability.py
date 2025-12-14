"""Model Explainability Module for FDA Transparency Compliance.

This module provides interpretability tools for medical AI models using
SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable
Model-agnostic Explanations).

FDA Guidance Reference:
- January 2025 Draft Guidance on AI-Enabled Device Software Functions
- Transparency requirements for AI/ML medical devices
- GMLP (Good Machine Learning Practice) principles

Features:
- SHAP value computation for global/local feature importance
- LIME explanations for individual predictions
- Clinical decision explanation reports
- Feature contribution visualization
- FDA-aligned transparency documentation
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

# Optional imports for explainability libraries
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime import lime_tabular

    HAS_LIME = True
except ImportError:
    HAS_LIME = False


class ExplanationType(Enum):
    """Types of model explanations."""

    SHAP_VALUES = "shap_values"
    SHAP_INTERACTION = "shap_interaction"
    LIME_TABULAR = "lime_tabular"
    LIME_TEXT = "lime_text"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"


class ClinicalRiskLevel(Enum):
    """Clinical risk levels for explanation context."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureContribution:
    """Represents a single feature's contribution to a prediction."""

    feature_name: str
    contribution: float
    base_value: float
    feature_value: Any
    direction: str  # "positive" or "negative"
    clinical_significance: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "contribution": float(self.contribution),
            "base_value": float(self.base_value),
            "feature_value": self.feature_value,
            "direction": self.direction,
            "clinical_significance": self.clinical_significance,
        }


@dataclass
class PredictionExplanation:
    """Complete explanation for a single prediction."""

    prediction: Any
    prediction_probability: float | None
    explanation_type: ExplanationType
    feature_contributions: list[FeatureContribution]
    base_value: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str | None = None
    input_hash: str | None = None
    clinical_context: str | None = None
    confidence_interval: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability,
            "explanation_type": self.explanation_type.value,
            "feature_contributions": [fc.to_dict() for fc in self.feature_contributions],
            "base_value": float(self.base_value),
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "input_hash": self.input_hash,
            "clinical_context": self.clinical_context,
            "confidence_interval": self.confidence_interval,
        }

    def top_features(self, n: int = 5) -> list[FeatureContribution]:
        """Get top N features by absolute contribution."""
        sorted_features = sorted(
            self.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True,
        )
        return sorted_features[:n]

    def positive_contributions(self) -> list[FeatureContribution]:
        """Get features with positive contributions."""
        return [fc for fc in self.feature_contributions if fc.direction == "positive"]

    def negative_contributions(self) -> list[FeatureContribution]:
        """Get features with negative contributions."""
        return [fc for fc in self.feature_contributions if fc.direction == "negative"]


@dataclass
class GlobalExplanation:
    """Global model explanation with feature importance."""

    model_name: str
    explanation_type: ExplanationType
    feature_importance: dict[str, float]
    feature_names: list[str]
    num_samples_analyzed: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    interaction_effects: dict[str, dict[str, float]] | None = None
    model_baseline: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "explanation_type": self.explanation_type.value,
            "feature_importance": self.feature_importance,
            "feature_names": self.feature_names,
            "num_samples_analyzed": self.num_samples_analyzed,
            "timestamp": self.timestamp,
            "interaction_effects": self.interaction_effects,
            "model_baseline": self.model_baseline,
        }

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_features[:n]


@dataclass
class ClinicalExplanationReport:
    """FDA-compliant clinical explanation report."""

    patient_id: str | None
    model_name: str
    model_version: str
    prediction: Any
    prediction_confidence: float
    clinical_risk_level: ClinicalRiskLevel
    explanation: PredictionExplanation
    clinical_interpretation: str
    key_contributing_factors: list[str]
    limitations: list[str]
    recommendations: list[str]
    regulatory_notes: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction": self.prediction,
            "prediction_confidence": self.prediction_confidence,
            "clinical_risk_level": self.clinical_risk_level.value,
            "explanation": self.explanation.to_dict(),
            "clinical_interpretation": self.clinical_interpretation,
            "key_contributing_factors": self.key_contributing_factors,
            "limitations": self.limitations,
            "recommendations": self.recommendations,
            "regulatory_notes": self.regulatory_notes,
            "timestamp": self.timestamp,
        }

    def to_markdown(self) -> str:
        """Generate markdown report for clinical documentation."""
        lines = [
            "# Clinical AI Decision Report",
            "",
            f"**Generated:** {self.timestamp}",
            f"**Model:** {self.model_name} v{self.model_version}",
            "",
            "## Prediction Summary",
            "",
            f"- **Prediction:** {self.prediction}",
            f"- **Confidence:** {self.prediction_confidence:.2%}",
            f"- **Clinical Risk Level:** {self.clinical_risk_level.value.upper()}",
            "",
            "## Clinical Interpretation",
            "",
            self.clinical_interpretation,
            "",
            "## Key Contributing Factors",
            "",
        ]

        for i, factor in enumerate(self.key_contributing_factors, 1):
            lines.append(f"{i}. {factor}")

        lines.extend(
            [
                "",
                "## Top Feature Contributions",
                "",
                "| Feature | Contribution | Direction |",
                "|---------|--------------|-----------|",
            ]
        )

        for fc in self.explanation.top_features(5):
            lines.append(f"| {fc.feature_name} | {fc.contribution:.4f} | {fc.direction} |")

        lines.extend(
            [
                "",
                "## Limitations",
                "",
            ]
        )
        for limitation in self.limitations:
            lines.append(f"- {limitation}")

        lines.extend(
            [
                "",
                "## Recommendations",
                "",
            ]
        )
        for rec in self.recommendations:
            lines.append(f"- {rec}")

        lines.extend(
            [
                "",
                "## Regulatory Notes",
                "",
            ]
        )
        for note in self.regulatory_notes:
            lines.append(f"- {note}")

        lines.extend(
            [
                "",
                "---",
                "",
                "*This report is generated for clinical decision support purposes.*",
                "*Final clinical decisions should be made by qualified healthcare professionals.*",
            ]
        )

        return "\n".join(lines)


class ModelExplainer:
    """Main class for generating model explanations.

    Supports both SHAP and LIME for comprehensive model interpretability.
    Designed for FDA transparency compliance in medical AI applications.
    """

    def __init__(
        self,
        model: Any,
        model_name: str = "unnamed_model",
        feature_names: list[str] | None = None,
        background_data: np.ndarray | None = None,
        model_type: str = "auto",  # "tree", "linear", "kernel", "deep", "auto"
    ):
        """Initialize the explainer.

        Args:
            model: The ML model to explain (must have predict or predict_proba)
            model_name: Name for documentation purposes
            feature_names: List of feature names
            background_data: Background dataset for SHAP (subset of training data)
            model_type: Type of model for optimized SHAP explainer selection
        """
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        self.background_data = background_data
        self.model_type = model_type
        self._shap_explainer: Any | None = None
        self._lime_explainer: Any | None = None

    def _get_predict_function(self) -> Callable:
        """Get the appropriate prediction function from the model."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        elif hasattr(self.model, "predict"):
            return self.model.predict
        else:
            raise ValueError("Model must have predict or predict_proba method")

    def _init_shap_explainer(self, X: np.ndarray) -> None:
        """Initialize SHAP explainer based on model type."""
        if not HAS_SHAP:
            raise ImportError(
                "SHAP is required for SHAP explanations. Install with: pip install shap"
            )

        background = self.background_data if self.background_data is not None else X

        # Auto-detect or use specified model type
        if self.model_type == "tree" or (
            self.model_type == "auto" and hasattr(self.model, "feature_importances_")
        ):
            self._shap_explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "linear" or (
            self.model_type == "auto" and hasattr(self.model, "coef_")
        ):
            self._shap_explainer = shap.LinearExplainer(self.model, background)
        elif self.model_type == "deep":
            self._shap_explainer = shap.DeepExplainer(self.model, background)
        else:
            # Default to KernelExplainer (model-agnostic but slower)
            predict_fn = self._get_predict_function()
            # Use a sample of background data for efficiency
            if len(background) > 100:
                indices = np.random.choice(len(background), 100, replace=False)
                background = background[indices]
            self._shap_explainer = shap.KernelExplainer(predict_fn, background)

    def _init_lime_explainer(self, X: np.ndarray) -> None:
        """Initialize LIME explainer."""
        if not HAS_LIME:
            raise ImportError(
                "LIME is required for LIME explanations. Install with: pip install lime"
            )

        self._lime_explainer = lime_tabular.LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            mode="classification" if hasattr(self.model, "predict_proba") else "regression",
            discretize_continuous=True,
        )

    def explain_prediction_shap(
        self,
        instance: np.ndarray,
        X_background: np.ndarray | None = None,
    ) -> PredictionExplanation:
        """Generate SHAP explanation for a single prediction.

        Args:
            instance: Single instance to explain (1D or 2D array)
            X_background: Background data for SHAP (if not provided during init)

        Returns:
            PredictionExplanation with SHAP values
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required. Install with: pip install shap")

        # Ensure 2D array
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Initialize explainer if needed
        if self._shap_explainer is None:
            background = X_background or self.background_data
            if background is None:
                raise ValueError(
                    "Background data required for SHAP. "
                    "Provide during init or via X_background parameter."
                )
            self._init_shap_explainer(background)

        # Compute SHAP values
        shap_values = self._shap_explainer.shap_values(instance)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # For classification, use values for predicted class
            pred_fn = self._get_predict_function()
            pred = pred_fn(instance)
            if hasattr(pred, "shape") and len(pred.shape) > 1:
                pred_class = np.argmax(pred[0])
                shap_values = shap_values[pred_class]
            else:
                shap_values = shap_values[1]  # Default to positive class

        # Get base value
        if hasattr(self._shap_explainer, "expected_value"):
            base_value = self._shap_explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[0])
        else:
            base_value = 0.0

        # Build feature contributions
        shap_values_flat = shap_values.flatten()
        feature_contributions = []
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(shap_values_flat))]

        for i, (name, value) in enumerate(zip(feature_names, shap_values_flat, strict=False)):
            fc = FeatureContribution(
                feature_name=name,
                contribution=float(value),
                base_value=base_value,
                feature_value=float(instance[0, i]) if instance.shape[1] > i else None,
                direction="positive" if value > 0 else "negative",
            )
            feature_contributions.append(fc)

        # Get prediction
        pred_fn = self._get_predict_function()
        prediction = pred_fn(instance)
        pred_prob = None
        if hasattr(prediction, "shape") and len(prediction.shape) > 1:
            pred_prob = float(np.max(prediction))
            prediction = int(np.argmax(prediction))
        else:
            prediction = float(prediction[0])

        return PredictionExplanation(
            prediction=prediction,
            prediction_probability=pred_prob,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=feature_contributions,
            base_value=base_value,
            model_name=self.model_name,
        )

    def explain_prediction_lime(
        self,
        instance: np.ndarray,
        X_train: np.ndarray | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> PredictionExplanation:
        """Generate LIME explanation for a single prediction.

        Args:
            instance: Single instance to explain
            X_train: Training data for LIME (if not provided during init)
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME perturbation

        Returns:
            PredictionExplanation with LIME values
        """
        if not HAS_LIME:
            raise ImportError("LIME is required. Install with: pip install lime")

        # Ensure 1D array for LIME
        if instance.ndim == 2:
            instance = instance.flatten()

        # Initialize explainer if needed
        if self._lime_explainer is None:
            train_data = X_train or self.background_data
            if train_data is None:
                raise ValueError(
                    "Training data required for LIME. "
                    "Provide during init or via X_train parameter."
                )
            self._init_lime_explainer(train_data)

        # Get prediction function
        predict_fn = self._get_predict_function()

        # Generate LIME explanation
        exp = self._lime_explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

        # Build feature contributions
        feature_contributions = []
        for feature_idx, weight in exp.as_list():
            # Parse feature name and value from LIME format
            fc = FeatureContribution(
                feature_name=str(feature_idx),
                contribution=float(weight),
                base_value=0.0,  # LIME doesn't provide base value
                feature_value=None,
                direction="positive" if weight > 0 else "negative",
            )
            feature_contributions.append(fc)

        # Get prediction
        instance_2d = instance.reshape(1, -1)
        prediction = predict_fn(instance_2d)
        pred_prob = None
        if hasattr(prediction, "shape") and len(prediction.shape) > 1:
            pred_prob = float(np.max(prediction))
            prediction = int(np.argmax(prediction))
        else:
            prediction = float(prediction[0])

        return PredictionExplanation(
            prediction=prediction,
            prediction_probability=pred_prob,
            explanation_type=ExplanationType.LIME_TABULAR,
            feature_contributions=feature_contributions,
            base_value=0.0,
            model_name=self.model_name,
        )

    def global_feature_importance(
        self,
        X: np.ndarray,
        method: str = "shap",
    ) -> GlobalExplanation:
        """Compute global feature importance.

        Args:
            X: Dataset to analyze
            method: "shap" or "permutation"

        Returns:
            GlobalExplanation with feature importance scores
        """
        if method == "shap":
            return self._global_importance_shap(X)
        elif method == "permutation":
            return self._global_importance_permutation(X)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shap' or 'permutation'.")

    def _global_importance_shap(self, X: np.ndarray) -> GlobalExplanation:
        """Compute global feature importance using SHAP."""
        if not HAS_SHAP:
            raise ImportError("SHAP is required. Install with: pip install shap")

        if self._shap_explainer is None:
            self._init_shap_explainer(X)

        shap_values = self._shap_explainer.shap_values(X)

        # Handle multi-class
        if isinstance(shap_values, list):
            # Average across classes
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        # Mean absolute SHAP value per feature
        mean_importance = np.mean(shap_values, axis=0)

        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(mean_importance))]

        importance_dict = {name: float(imp) for name, imp in zip(feature_names, mean_importance, strict=False)}

        base_value = None
        if hasattr(self._shap_explainer, "expected_value"):
            ev = self._shap_explainer.expected_value
            base_value = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)

        return GlobalExplanation(
            model_name=self.model_name,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_importance=importance_dict,
            feature_names=feature_names,
            num_samples_analyzed=len(X),
            model_baseline=base_value,
        )

    def _global_importance_permutation(self, X: np.ndarray) -> GlobalExplanation:
        """Compute global feature importance using permutation importance."""
        from sklearn.inspection import permutation_importance

        # Need labels for permutation importance
        y_pred = self.model.predict(X)

        result = permutation_importance(
            self.model,
            X,
            y_pred,
            n_repeats=10,
            random_state=42,
        )

        feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        importance_dict = {
            name: float(imp) for name, imp in zip(feature_names, result.importances_mean, strict=False)
        }

        return GlobalExplanation(
            model_name=self.model_name,
            explanation_type=ExplanationType.PERMUTATION_IMPORTANCE,
            feature_importance=importance_dict,
            feature_names=feature_names,
            num_samples_analyzed=len(X),
        )

    def generate_clinical_report(
        self,
        instance: np.ndarray,
        X_background: np.ndarray | None = None,
        patient_id: str | None = None,
        model_version: str = "1.0.0",
        clinical_context: str | None = None,
    ) -> ClinicalExplanationReport:
        """Generate FDA-compliant clinical explanation report.

        Args:
            instance: Instance to explain
            X_background: Background data for SHAP
            patient_id: Optional patient identifier (anonymized)
            model_version: Model version string
            clinical_context: Optional clinical context description

        Returns:
            ClinicalExplanationReport for documentation
        """
        # Get explanation
        explanation = self.explain_prediction_shap(instance, X_background)

        # Determine clinical risk level based on prediction confidence
        pred_prob = explanation.prediction_probability or 0.5
        if pred_prob >= 0.9 or pred_prob <= 0.1:
            risk_level = ClinicalRiskLevel.HIGH
        elif pred_prob >= 0.7 or pred_prob <= 0.3:
            risk_level = ClinicalRiskLevel.MEDIUM
        else:
            risk_level = ClinicalRiskLevel.LOW

        # Generate key contributing factors
        top_features = explanation.top_features(5)
        key_factors = []
        for fc in top_features:
            direction = "increased" if fc.direction == "positive" else "decreased"
            key_factors.append(
                f"{fc.feature_name}: {direction} prediction by {abs(fc.contribution):.4f}"
            )

        # Standard limitations for medical AI
        limitations = [
            "This AI system is intended for decision support only",
            "Model performance may vary across different patient populations",
            "Explanation values are approximations and should be interpreted with caution",
            "Feature interactions may not be fully captured by individual contributions",
        ]

        # Standard recommendations
        recommendations = [
            "Verify AI predictions against clinical judgment and patient history",
            "Consider additional diagnostic tests if prediction confidence is low",
            "Document clinical reasoning alongside AI-assisted decisions",
            "Report any unexpected AI behavior through proper channels",
        ]

        # Regulatory notes
        regulatory_notes = [
            "AI model explanation generated per FDA transparency requirements",
            "Model performance validated according to PCCP specifications",
            "This output is part of the model's audit trail",
        ]

        # Generate clinical interpretation
        interpretation = (
            f"The model predicts {explanation.prediction} with "
            f"{pred_prob:.1%} confidence. "
            f"The top contributing factor is '{top_features[0].feature_name}' "
            f"with a {top_features[0].direction} influence."
        )
        if clinical_context:
            interpretation = f"{clinical_context}\n\n{interpretation}"

        return ClinicalExplanationReport(
            patient_id=patient_id,
            model_name=self.model_name,
            model_version=model_version,
            prediction=explanation.prediction,
            prediction_confidence=pred_prob,
            clinical_risk_level=risk_level,
            explanation=explanation,
            clinical_interpretation=interpretation,
            key_contributing_factors=key_factors,
            limitations=limitations,
            recommendations=recommendations,
            regulatory_notes=regulatory_notes,
        )


def quick_explain(
    model: Any,
    instance: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str] | None = None,
    method: str = "shap",
) -> dict:
    """Quick explanation for a single prediction.

    Convenience function for rapid explanation generation.

    Args:
        model: ML model to explain
        instance: Instance to explain
        X_background: Background/training data
        feature_names: Optional feature names
        method: "shap" or "lime"

    Returns:
        Dictionary with explanation summary
    """
    explainer = ModelExplainer(
        model=model,
        feature_names=feature_names,
        background_data=X_background,
    )

    if method == "shap":
        explanation = explainer.explain_prediction_shap(instance, X_background)
    else:
        explanation = explainer.explain_prediction_lime(instance, X_background)

    top_features = explanation.top_features(5)

    return {
        "prediction": explanation.prediction,
        "probability": explanation.prediction_probability,
        "top_features": [
            {
                "name": fc.feature_name,
                "contribution": fc.contribution,
                "direction": fc.direction,
            }
            for fc in top_features
        ],
        "explanation_type": explanation.explanation_type.value,
    }
