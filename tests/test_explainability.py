"""
Tests for Model Explainability Module.

Tests SHAP and LIME-based model explanations for FDA transparency compliance.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from medtech_ai_security.ml.explainability import (
    ClinicalExplanationReport,
    ClinicalRiskLevel,
    ExplanationType,
    FeatureContribution,
    GlobalExplanation,
    ModelExplainer,
    PredictionExplanation,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dataset():
    """Generate a sample classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(10)]
    return X, y, feature_names


@pytest.fixture
def trained_classifier(sample_dataset):
    """Create a trained classifier."""
    X, y, _ = sample_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_logistic(sample_dataset):
    """Create a trained logistic regression model."""
    X, y, _ = sample_dataset
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)
    return model


@pytest.fixture
def explainer(trained_classifier, sample_dataset):
    """Create a ModelExplainer instance."""
    X, _, feature_names = sample_dataset
    return ModelExplainer(
        model=trained_classifier,
        model_name="test_classifier",
        feature_names=feature_names,
        background_data=X[:50],
    )


@pytest.fixture
def explainer_logistic(trained_logistic, sample_dataset):
    """Create a ModelExplainer for logistic regression."""
    X, _, feature_names = sample_dataset
    return ModelExplainer(
        model=trained_logistic,
        model_name="test_logistic",
        feature_names=feature_names,
        background_data=X[:50],
    )


# =============================================================================
# Data Class Tests
# =============================================================================


class TestFeatureContribution:
    """Tests for FeatureContribution dataclass."""

    def test_creation(self):
        """Test creating a FeatureContribution."""
        fc = FeatureContribution(
            feature_name="age",
            contribution=0.15,
            base_value=0.5,
            feature_value=45.0,
            direction="positive",
        )
        assert fc.feature_name == "age"
        assert fc.feature_value == 45.0
        assert fc.contribution == 0.15
        assert fc.base_value == 0.5
        assert fc.direction == "positive"

    def test_optional_clinical_significance(self):
        """Test FeatureContribution with clinical significance."""
        fc = FeatureContribution(
            feature_name="blood_pressure",
            contribution=0.25,
            base_value=0.5,
            feature_value=140.0,
            direction="positive",
            clinical_significance="High blood pressure increases risk",
        )
        assert fc.clinical_significance == "High blood pressure increases risk"

    def test_to_dict(self):
        """Test converting FeatureContribution to dictionary."""
        fc = FeatureContribution(
            feature_name="age",
            contribution=0.15,
            base_value=0.5,
            feature_value=45.0,
            direction="positive",
        )
        data = fc.to_dict()
        assert data["feature_name"] == "age"
        assert data["contribution"] == 0.15
        assert data["direction"] == "positive"


class TestPredictionExplanation:
    """Tests for PredictionExplanation dataclass."""

    def test_creation(self):
        """Test creating a PredictionExplanation."""
        contributions = [
            FeatureContribution("f1", 0.1, 0.5, 1.0, "positive"),
            FeatureContribution("f2", -0.05, 0.5, 2.0, "negative"),
        ]
        exp = PredictionExplanation(
            prediction=1,
            prediction_probability=0.85,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
        )
        assert exp.explanation_type == ExplanationType.SHAP_VALUES
        assert exp.prediction == 1
        assert exp.prediction_probability == 0.85
        assert len(exp.feature_contributions) == 2
        assert exp.base_value == 0.5

    def test_top_features_sorted(self):
        """Test that top_features returns sorted contributions."""
        contributions = [
            FeatureContribution("f1", 0.1, 0.5, 1.0, "positive"),
            FeatureContribution("f2", 0.3, 0.5, 2.0, "positive"),
            FeatureContribution("f3", -0.2, 0.5, 3.0, "negative"),
            FeatureContribution("f4", 0.05, 0.5, 4.0, "positive"),
        ]
        exp = PredictionExplanation(
            prediction=1,
            prediction_probability=0.85,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
        )
        top = exp.top_features(n=2)
        assert len(top) == 2
        assert top[0].feature_name == "f2"  # Highest absolute contribution
        assert top[1].feature_name == "f3"  # Second highest

    def test_to_dict(self):
        """Test converting PredictionExplanation to dictionary."""
        contributions = [
            FeatureContribution("f1", 0.1, 0.5, 1.0, "positive"),
        ]
        exp = PredictionExplanation(
            prediction=1,
            prediction_probability=0.85,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
        )
        data = exp.to_dict()
        assert data["prediction"] == 1
        assert data["explanation_type"] == "shap_values"
        assert len(data["feature_contributions"]) == 1


class TestGlobalExplanation:
    """Tests for GlobalExplanation dataclass."""

    def test_creation(self):
        """Test creating a GlobalExplanation."""
        importance = {"f1": 0.3, "f2": 0.5, "f3": 0.2}
        exp = GlobalExplanation(
            model_name="test_model",
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_importance=importance,
            feature_names=["f1", "f2", "f3"],
            num_samples_analyzed=100,
        )
        assert exp.explanation_type == ExplanationType.SHAP_VALUES
        assert exp.feature_importance == importance
        assert exp.num_samples_analyzed == 100

    def test_top_features(self):
        """Test top_features returns sorted list."""
        importance = {"f1": 0.3, "f2": 0.5, "f3": 0.2}
        exp = GlobalExplanation(
            model_name="test_model",
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_importance=importance,
            feature_names=["f1", "f2", "f3"],
            num_samples_analyzed=100,
        )
        top = exp.top_features(n=2)
        assert top[0] == ("f2", 0.5)
        assert top[1] == ("f1", 0.3)

    def test_to_dict(self):
        """Test converting GlobalExplanation to dictionary."""
        importance = {"f1": 0.3}
        exp = GlobalExplanation(
            model_name="test_model",
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_importance=importance,
            feature_names=["f1"],
            num_samples_analyzed=100,
        )
        data = exp.to_dict()
        assert data["model_name"] == "test_model"
        assert data["num_samples_analyzed"] == 100


class TestClinicalExplanationReport:
    """Tests for ClinicalExplanationReport dataclass."""

    def test_creation(self):
        """Test creating a ClinicalExplanationReport."""
        contributions = [
            FeatureContribution("blood_pressure", 0.2, 0.5, 140.0, "positive"),
        ]
        explanation = PredictionExplanation(
            prediction=1,
            prediction_probability=0.85,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
        )
        report = ClinicalExplanationReport(
            patient_id="P001",
            model_name="risk_classifier",
            model_version="2.0.0",
            prediction=1,
            prediction_confidence=0.85,
            clinical_risk_level=ClinicalRiskLevel.HIGH,
            explanation=explanation,
            clinical_interpretation="High cardiovascular risk detected",
            key_contributing_factors=["Elevated blood pressure"],
            limitations=["Model trained on limited demographics"],
            recommendations=["Consult cardiologist"],
            regulatory_notes=["FDA cleared for decision support only"],
        )
        assert report.patient_id == "P001"
        assert report.model_name == "risk_classifier"
        assert report.clinical_risk_level == ClinicalRiskLevel.HIGH

    def test_to_markdown(self):
        """Test markdown report generation."""
        contributions = [
            FeatureContribution("blood_pressure", 0.2, 0.5, 140.0, "positive"),
            FeatureContribution("age", 0.15, 0.5, 65.0, "positive"),
        ]
        explanation = PredictionExplanation(
            prediction=1,
            prediction_probability=0.85,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
        )
        report = ClinicalExplanationReport(
            patient_id="P001",
            model_name="risk_classifier",
            model_version="2.0.0",
            prediction=1,
            prediction_confidence=0.85,
            clinical_risk_level=ClinicalRiskLevel.HIGH,
            explanation=explanation,
            clinical_interpretation="High risk detected",
            key_contributing_factors=["Blood pressure", "Age"],
            limitations=["Limited data"],
            recommendations=["Follow up"],
            regulatory_notes=["For decision support"],
        )
        md = report.to_markdown()

        assert "# Clinical AI Decision Report" in md
        assert "risk_classifier" in md
        assert "2.0.0" in md
        assert "HIGH" in md


# =============================================================================
# ModelExplainer Tests
# =============================================================================


class TestModelExplainer:
    """Tests for ModelExplainer class."""

    def test_initialization(self, trained_classifier, sample_dataset):
        """Test ModelExplainer initialization."""
        X, _, feature_names = sample_dataset
        explainer = ModelExplainer(
            model=trained_classifier,
            model_name="test_model",
            feature_names=feature_names,
        )
        assert explainer.model == trained_classifier
        assert explainer.feature_names == feature_names
        assert explainer.model_name == "test_model"

    def test_initialization_without_feature_names(self, trained_classifier):
        """Test initialization without feature names."""
        explainer = ModelExplainer(
            model=trained_classifier,
            model_name="test_model",
        )
        # Should not raise, feature names can be None
        assert explainer.feature_names is None

    def test_initialization_with_background_data(self, trained_classifier, sample_dataset):
        """Test initialization with background data."""
        X, _, feature_names = sample_dataset
        explainer = ModelExplainer(
            model=trained_classifier,
            model_name="test_model",
            feature_names=feature_names,
            background_data=X[:50],
        )
        assert explainer.background_data is not None
        assert len(explainer.background_data) == 50


class TestModelExplainerEdgeCases:
    """Edge case tests for ModelExplainer."""

    def test_single_feature_model(self):
        """Test explainer with single-feature model."""
        X = np.random.rand(100, 1)
        y = (X[:, 0] > 0.5).astype(int)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        explainer = ModelExplainer(
            model=model,
            model_name="single_feature_model",
            feature_names=["single_feature"],
        )

        assert explainer.model_name == "single_feature_model"
        assert explainer.feature_names == ["single_feature"]

    def test_multiclass_model(self):
        """Test explainer with multiclass model."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_informative=3,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = ModelExplainer(
            model=model,
            model_name="multiclass_model",
            feature_names=[f"f{i}" for i in range(5)],
        )

        assert explainer.model_name == "multiclass_model"


# =============================================================================
# Dataclass Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for dataclass serialization."""

    def test_feature_contribution_roundtrip(self):
        """Test FeatureContribution serialization."""
        fc = FeatureContribution(
            feature_name="test",
            contribution=0.5,
            base_value=0.3,
            feature_value=10.0,
            direction="positive",
        )
        data = fc.to_dict()
        assert data["feature_name"] == "test"
        assert data["contribution"] == 0.5

    def test_prediction_explanation_roundtrip(self):
        """Test PredictionExplanation serialization."""
        fc = FeatureContribution("f1", 0.1, 0.5, 1.0, "positive")
        exp = PredictionExplanation(
            prediction=1,
            prediction_probability=0.9,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=[fc],
            base_value=0.5,
        )
        data = exp.to_dict()
        assert data["prediction"] == 1
        assert data["explanation_type"] == "shap_values"

    def test_global_explanation_roundtrip(self):
        """Test GlobalExplanation serialization."""
        exp = GlobalExplanation(
            model_name="test",
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            feature_importance={"a": 0.5, "b": 0.3},
            feature_names=["a", "b"],
            num_samples_analyzed=100,
        )
        data = exp.to_dict()
        assert data["model_name"] == "test"
        assert data["num_samples_analyzed"] == 100


# =============================================================================
# Risk Level Classification Tests
# =============================================================================


class TestRiskLevelClassification:
    """Tests for clinical risk level classification."""

    def test_risk_levels_exist(self):
        """Test that all risk levels are defined."""
        assert ClinicalRiskLevel.LOW.value == "low"
        assert ClinicalRiskLevel.MEDIUM.value == "medium"
        assert ClinicalRiskLevel.HIGH.value == "high"
        assert ClinicalRiskLevel.CRITICAL.value == "critical"


# =============================================================================
# FDA Compliance Tests
# =============================================================================


class TestFDACompliance:
    """Tests for FDA transparency compliance features."""

    def test_report_contains_regulatory_notes(self):
        """Test that clinical report includes regulatory notes section."""
        contributions = [
            FeatureContribution("f1", 0.2, 0.5, 1.0, "positive"),
        ]
        explanation = PredictionExplanation(
            prediction=1,
            prediction_probability=0.85,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
        )
        report = ClinicalExplanationReport(
            patient_id="TEST",
            model_name="test_model",
            model_version="1.0.0",
            prediction=1,
            prediction_confidence=0.85,
            clinical_risk_level=ClinicalRiskLevel.MEDIUM,
            explanation=explanation,
            clinical_interpretation="Test interpretation",
            key_contributing_factors=["Factor 1"],
            limitations=["Limitation 1"],
            recommendations=["Recommendation 1"],
            regulatory_notes=["FDA cleared for clinical decision support"],
        )

        md = report.to_markdown()
        assert "Regulatory Notes" in md
        assert "FDA" in md

    def test_report_includes_timestamp(self):
        """Test that clinical report includes timestamp."""
        contributions = [
            FeatureContribution("f1", 0.2, 0.5, 1.0, "positive"),
        ]
        explanation = PredictionExplanation(
            prediction=1,
            prediction_probability=0.85,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
        )
        report = ClinicalExplanationReport(
            patient_id="TEST",
            model_name="test_model",
            model_version="1.0.0",
            prediction=1,
            prediction_confidence=0.85,
            clinical_risk_level=ClinicalRiskLevel.LOW,
            explanation=explanation,
            clinical_interpretation="Test",
            key_contributing_factors=[],
            limitations=[],
            recommendations=[],
            regulatory_notes=[],
        )

        assert report.timestamp is not None
        md = report.to_markdown()
        assert "Generated:" in md

    def test_report_includes_model_metadata(self):
        """Test that report includes complete model metadata."""
        contributions = [
            FeatureContribution("f1", 0.2, 0.5, 1.0, "positive"),
        ]
        explanation = PredictionExplanation(
            prediction=1,
            prediction_probability=0.85,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
        )
        report = ClinicalExplanationReport(
            patient_id="METADATA_TEST",
            model_name="test_classifier",
            model_version="1.0.0",
            prediction=1,
            prediction_confidence=0.85,
            clinical_risk_level=ClinicalRiskLevel.MEDIUM,
            explanation=explanation,
            clinical_interpretation="Test",
            key_contributing_factors=[],
            limitations=[],
            recommendations=[],
            regulatory_notes=[],
        )

        assert report.model_name == "test_classifier"
        assert report.model_version == "1.0.0"
        md = report.to_markdown()
        assert "test_classifier" in md
        assert "1.0.0" in md


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the explainability module."""

    def test_explanation_types_enum(self):
        """Test all explanation types are accessible."""
        assert ExplanationType.SHAP_VALUES.value == "shap_values"
        assert ExplanationType.LIME_TABULAR.value == "lime_tabular"
        assert ExplanationType.FEATURE_IMPORTANCE.value == "feature_importance"

    def test_complete_report_workflow(self):
        """Test complete workflow of creating a clinical report."""
        # Create feature contributions
        contributions = [
            FeatureContribution("heart_rate", 0.25, 0.5, 95.0, "positive",
                              "Elevated heart rate contributes to risk"),
            FeatureContribution("age", 0.20, 0.5, 65.0, "positive",
                              "Advanced age increases risk"),
            FeatureContribution("blood_pressure", 0.15, 0.5, 145.0, "positive"),
            FeatureContribution("cholesterol", -0.10, 0.5, 180.0, "negative"),
        ]

        # Create explanation
        explanation = PredictionExplanation(
            prediction="High Risk",
            prediction_probability=0.82,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=0.5,
            model_name="cardiovascular_risk_model",
            clinical_context="Cardiovascular risk assessment",
        )

        # Create clinical report
        report = ClinicalExplanationReport(
            patient_id="P12345",
            model_name="cardiovascular_risk_model",
            model_version="2.1.0",
            prediction="High Risk",
            prediction_confidence=0.82,
            clinical_risk_level=ClinicalRiskLevel.HIGH,
            explanation=explanation,
            clinical_interpretation="Based on the analysis, the patient shows elevated cardiovascular risk factors.",
            key_contributing_factors=[
                "Elevated heart rate (95 bpm)",
                "Advanced age (65 years)",
                "High blood pressure (145 mmHg)",
            ],
            limitations=[
                "Model trained primarily on Western populations",
                "Does not account for family history",
            ],
            recommendations=[
                "Recommend follow-up with cardiologist",
                "Consider lifestyle modifications",
                "Schedule stress test within 30 days",
            ],
            regulatory_notes=[
                "FDA 510(k) cleared for clinical decision support",
                "Not intended to replace clinical judgment",
            ],
        )

        # Validate report
        assert report.patient_id == "P12345"
        assert len(explanation.top_features(3)) == 3
        assert explanation.positive_contributions()[0].feature_name == "heart_rate"

        # Generate markdown
        md = report.to_markdown()
        assert len(md) > 500
        assert "cardiovascular_risk_model" in md
        assert "High Risk" in md or "HIGH" in md
        assert "heart_rate" in md
