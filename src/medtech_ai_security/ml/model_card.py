"""
Model Card Generator for FDA Transparency Compliance.

Generates standardized model cards following Google's Model Card format,
extended for medical AI requirements per FDA guidance on AI/ML devices.

Security Context:
- FDA AI/ML Guidance 2025: Transparency requirements for AI-enabled devices
- GMLP Principles: Documentation throughout the product lifecycle
- EU AI Act: Documentation requirements for high-risk AI systems

References:
- Google Model Cards for Model Reporting (Mitchell et al., 2019)
- FDA Draft Guidance: AI-Enabled Device Software Functions
- NIST AI RMF: AI Risk Management Framework
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class RiskLevel(str, Enum):
    """FDA risk classification for medical devices."""

    CLASS_I = "class_i"  # Low risk
    CLASS_II = "class_ii"  # Moderate risk
    CLASS_III = "class_iii"  # High risk


class IntendedUse(str, Enum):
    """Intended use categories for medical AI."""

    DIAGNOSTIC = "diagnostic"
    SCREENING = "screening"
    MONITORING = "monitoring"
    THERAPEUTIC = "therapeutic"
    PREDICTIVE = "predictive"
    TRIAGE = "triage"
    CLINICAL_DECISION_SUPPORT = "clinical_decision_support"
    OTHER = "other"


class DatasetType(str, Enum):
    """Types of datasets used in training."""

    PROPRIETARY = "proprietary"
    PUBLIC = "public"
    SYNTHETIC = "synthetic"
    MIXED = "mixed"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelDetails:
    """Basic model information."""

    name: str
    version: str
    description: str
    architecture: str
    framework: str
    created_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    developers: list[str] = field(default_factory=list)
    contact: str | None = None
    license: str = "Proprietary"
    citation: str | None = None


@dataclass
class IntendedUseDetails:
    """Intended use and users of the model."""

    primary_use: IntendedUse
    primary_users: list[str]  # e.g., ["radiologists", "pathologists"]
    use_cases: list[str]  # Specific use case descriptions
    out_of_scope_uses: list[str]  # What the model should NOT be used for
    clinical_workflow: str | None = None  # Where it fits in clinical workflow


@dataclass
class TrainingDataDetails:
    """Training data documentation."""

    dataset_type: DatasetType
    dataset_name: str | None = None
    dataset_size: int | None = None
    collection_method: str | None = None
    preprocessing: list[str] = field(default_factory=list)
    demographics: dict[str, Any] = field(default_factory=dict)
    geographic_distribution: list[str] = field(default_factory=list)
    temporal_range: str | None = None  # e.g., "2018-2023"
    class_distribution: dict[str, float] = field(default_factory=dict)
    known_biases: list[str] = field(default_factory=list)
    data_quality_notes: list[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""

    overall_metrics: dict[str, float]  # e.g., {"accuracy": 0.95, "auc": 0.98}
    subgroup_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )
    threshold_analysis: dict[str, Any] = field(default_factory=dict)
    comparison_to_baseline: str | None = None
    failure_cases: list[str] = field(default_factory=list)


@dataclass
class EthicalConsiderations:
    """Ethical and fairness considerations."""

    sensitive_attributes: list[str]  # e.g., ["age", "sex", "race", "ethnicity"]
    fairness_metrics: dict[str, float] = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)
    mitigation_strategies: list[str] = field(default_factory=list)
    human_oversight: str | None = None
    transparency_notes: list[str] = field(default_factory=list)


@dataclass
class RegulatoryInformation:
    """Regulatory and compliance information."""

    fda_risk_class: RiskLevel
    fda_submission_type: str | None = None  # "510(k)", "De Novo", "PMA"
    fda_clearance_number: str | None = None
    predicate_devices: list[str] = field(default_factory=list)
    standards_compliance: list[str] = field(default_factory=list)
    quality_system: str = "21 CFR Part 820"
    pccp_enabled: bool = False  # Predetermined Change Control Plan
    lifecycle_monitoring: bool = True


@dataclass
class CybersecurityInformation:
    """Cybersecurity and safety information."""

    threat_model: str | None = None
    adversarial_testing: bool = False
    adversarial_robustness: dict[str, float] = field(default_factory=dict)
    data_poisoning_defense: bool = False
    model_integrity_verified: bool = False
    integrity_hash: str | None = None
    sbom_available: bool = False
    vulnerability_disclosure: str | None = None
    update_mechanism: str | None = None


@dataclass
class QuantitativeAnalysis:
    """Detailed quantitative analysis."""

    test_set_size: int
    evaluation_methodology: str
    cross_validation: str | None = None
    statistical_tests: list[str] = field(default_factory=list)
    effect_sizes: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelCard:
    """Complete model card following Google/FDA format."""

    model_details: ModelDetails
    intended_use: IntendedUseDetails
    training_data: TrainingDataDetails
    performance: PerformanceMetrics
    ethical_considerations: EthicalConsiderations
    regulatory: RegulatoryInformation
    cybersecurity: CybersecurityInformation
    quantitative_analysis: QuantitativeAnalysis | None = None
    additional_information: dict[str, Any] = field(default_factory=dict)
    version_history: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model card to dictionary."""
        return {
            "model_details": self._dataclass_to_dict(self.model_details),
            "intended_use": self._dataclass_to_dict(self.intended_use),
            "training_data": self._dataclass_to_dict(self.training_data),
            "performance": self._dataclass_to_dict(self.performance),
            "ethical_considerations": self._dataclass_to_dict(
                self.ethical_considerations
            ),
            "regulatory": self._dataclass_to_dict(self.regulatory),
            "cybersecurity": self._dataclass_to_dict(self.cybersecurity),
            "quantitative_analysis": (
                self._dataclass_to_dict(self.quantitative_analysis)
                if self.quantitative_analysis
                else None
            ),
            "additional_information": self.additional_information,
            "version_history": self.version_history,
            "generated_at": self.generated_at,
        }

    def _dataclass_to_dict(self, obj: Any) -> dict[str, Any]:
        """Convert a dataclass to dict, handling enums."""
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list):
                result[key] = [
                    v.value if isinstance(v, Enum) else v for v in value
                ]
            else:
                result[key] = value
        return result


# =============================================================================
# Model Card Generator
# =============================================================================


class ModelCardGenerator:
    """
    Generates model cards for FDA compliance and transparency.

    Supports multiple output formats:
    - JSON: Machine-readable format
    - Markdown: Human-readable documentation
    - HTML: Web-ready presentation

    Example:
        generator = ModelCardGenerator()

        card = generator.create_card(
            model_details=ModelDetails(
                name="Chest X-Ray Classifier",
                version="1.0.0",
                description="Detects pneumonia in chest X-rays",
                architecture="ResNet-50",
                framework="PyTorch 2.0",
            ),
            intended_use=IntendedUseDetails(...),
            ...
        )

        generator.save_as_markdown(card, "model_card.md")
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        pass

    def create_card(
        self,
        model_details: ModelDetails,
        intended_use: IntendedUseDetails,
        training_data: TrainingDataDetails,
        performance: PerformanceMetrics,
        ethical_considerations: EthicalConsiderations,
        regulatory: RegulatoryInformation,
        cybersecurity: CybersecurityInformation,
        quantitative_analysis: QuantitativeAnalysis | None = None,
        additional_information: dict[str, Any] | None = None,
    ) -> ModelCard:
        """
        Create a complete model card.

        Args:
            model_details: Basic model information
            intended_use: Intended use details
            training_data: Training data documentation
            performance: Performance metrics
            ethical_considerations: Ethics and fairness info
            regulatory: Regulatory compliance info
            cybersecurity: Security information
            quantitative_analysis: Optional detailed analysis
            additional_information: Optional extra info

        Returns:
            Complete ModelCard
        """
        return ModelCard(
            model_details=model_details,
            intended_use=intended_use,
            training_data=training_data,
            performance=performance,
            ethical_considerations=ethical_considerations,
            regulatory=regulatory,
            cybersecurity=cybersecurity,
            quantitative_analysis=quantitative_analysis,
            additional_information=additional_information or {},
        )

    def save_as_json(
        self,
        card: ModelCard,
        output_path: str | Path,
    ) -> None:
        """Save model card as JSON."""
        path = Path(output_path)
        with open(path, "w") as f:
            json.dump(card.to_dict(), f, indent=2)
        logger.info(f"Saved model card to {path}")

    def save_as_markdown(
        self,
        card: ModelCard,
        output_path: str | Path,
    ) -> None:
        """Save model card as Markdown."""
        path = Path(output_path)
        md = self._generate_markdown(card)
        with open(path, "w") as f:
            f.write(md)
        logger.info(f"Saved model card to {path}")

    def _generate_markdown(self, card: ModelCard) -> str:
        """Generate Markdown representation of model card."""
        lines = []

        # Header
        lines.append(f"# Model Card: {card.model_details.name}")
        lines.append("")
        lines.append(f"**Version**: {card.model_details.version}")
        lines.append(f"**Generated**: {card.generated_at}")
        lines.append("")

        # Model Details
        lines.append("## Model Details")
        lines.append("")
        lines.append(f"- **Name**: {card.model_details.name}")
        lines.append(f"- **Version**: {card.model_details.version}")
        lines.append(f"- **Description**: {card.model_details.description}")
        lines.append(f"- **Architecture**: {card.model_details.architecture}")
        lines.append(f"- **Framework**: {card.model_details.framework}")
        lines.append(f"- **Created**: {card.model_details.created_date}")
        if card.model_details.developers:
            lines.append(
                f"- **Developers**: {', '.join(card.model_details.developers)}"
            )
        lines.append(f"- **License**: {card.model_details.license}")
        lines.append("")

        # Intended Use
        lines.append("## Intended Use")
        lines.append("")
        lines.append(
            f"- **Primary Use**: {card.intended_use.primary_use.value.replace('_', ' ').title()}"
        )
        lines.append(
            f"- **Primary Users**: {', '.join(card.intended_use.primary_users)}"
        )
        lines.append("")
        lines.append("### Use Cases")
        for use_case in card.intended_use.use_cases:
            lines.append(f"- {use_case}")
        lines.append("")
        lines.append("### Out of Scope Uses")
        for oos in card.intended_use.out_of_scope_uses:
            lines.append(f"- {oos}")
        lines.append("")

        # Training Data
        lines.append("## Training Data")
        lines.append("")
        lines.append(
            f"- **Dataset Type**: {card.training_data.dataset_type.value.title()}"
        )
        if card.training_data.dataset_name:
            lines.append(f"- **Dataset Name**: {card.training_data.dataset_name}")
        if card.training_data.dataset_size:
            lines.append(
                f"- **Dataset Size**: {card.training_data.dataset_size:,} samples"
            )
        if card.training_data.temporal_range:
            lines.append(
                f"- **Temporal Range**: {card.training_data.temporal_range}"
            )
        if card.training_data.geographic_distribution:
            lines.append(
                f"- **Geographic Distribution**: {', '.join(card.training_data.geographic_distribution)}"
            )
        lines.append("")

        if card.training_data.demographics:
            lines.append("### Demographics")
            for key, value in card.training_data.demographics.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if card.training_data.known_biases:
            lines.append("### Known Biases")
            for bias in card.training_data.known_biases:
                lines.append(f"- {bias}")
            lines.append("")

        # Performance Metrics
        lines.append("## Performance")
        lines.append("")
        lines.append("### Overall Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for metric, value in card.performance.overall_metrics.items():
            if isinstance(value, float):
                lines.append(f"| {metric} | {value:.4f} |")
            else:
                lines.append(f"| {metric} | {value} |")
        lines.append("")

        if card.performance.subgroup_metrics:
            lines.append("### Subgroup Performance")
            lines.append("")
            for group, metrics in card.performance.subgroup_metrics.items():
                lines.append(f"**{group}**:")
                for metric, value in metrics.items():
                    lines.append(f"  - {metric}: {value:.4f}")
            lines.append("")

        if card.performance.failure_cases:
            lines.append("### Known Failure Cases")
            for case in card.performance.failure_cases:
                lines.append(f"- {case}")
            lines.append("")

        # Ethical Considerations
        lines.append("## Ethical Considerations")
        lines.append("")
        lines.append(
            f"- **Sensitive Attributes**: {', '.join(card.ethical_considerations.sensitive_attributes)}"
        )
        if card.ethical_considerations.human_oversight:
            lines.append(
                f"- **Human Oversight**: {card.ethical_considerations.human_oversight}"
            )
        lines.append("")

        if card.ethical_considerations.known_limitations:
            lines.append("### Known Limitations")
            for limitation in card.ethical_considerations.known_limitations:
                lines.append(f"- {limitation}")
            lines.append("")

        if card.ethical_considerations.mitigation_strategies:
            lines.append("### Mitigation Strategies")
            for strategy in card.ethical_considerations.mitigation_strategies:
                lines.append(f"- {strategy}")
            lines.append("")

        # Regulatory Information
        lines.append("## Regulatory Information")
        lines.append("")
        lines.append(
            f"- **FDA Risk Class**: {card.regulatory.fda_risk_class.value.replace('_', ' ').title()}"
        )
        if card.regulatory.fda_submission_type:
            lines.append(
                f"- **Submission Type**: {card.regulatory.fda_submission_type}"
            )
        if card.regulatory.fda_clearance_number:
            lines.append(
                f"- **Clearance Number**: {card.regulatory.fda_clearance_number}"
            )
        lines.append(f"- **Quality System**: {card.regulatory.quality_system}")
        lines.append(
            f"- **PCCP Enabled**: {'Yes' if card.regulatory.pccp_enabled else 'No'}"
        )
        lines.append(
            f"- **Lifecycle Monitoring**: {'Yes' if card.regulatory.lifecycle_monitoring else 'No'}"
        )
        lines.append("")

        if card.regulatory.standards_compliance:
            lines.append("### Standards Compliance")
            for standard in card.regulatory.standards_compliance:
                lines.append(f"- {standard}")
            lines.append("")

        # Cybersecurity
        lines.append("## Cybersecurity")
        lines.append("")
        lines.append(
            f"- **Adversarial Testing**: {'Yes' if card.cybersecurity.adversarial_testing else 'No'}"
        )
        lines.append(
            f"- **Data Poisoning Defense**: {'Yes' if card.cybersecurity.data_poisoning_defense else 'No'}"
        )
        lines.append(
            f"- **Model Integrity Verified**: {'Yes' if card.cybersecurity.model_integrity_verified else 'No'}"
        )
        lines.append(
            f"- **SBOM Available**: {'Yes' if card.cybersecurity.sbom_available else 'No'}"
        )
        if card.cybersecurity.integrity_hash:
            lines.append(
                f"- **Integrity Hash**: `{card.cybersecurity.integrity_hash[:32]}...`"
            )
        lines.append("")

        if card.cybersecurity.adversarial_robustness:
            lines.append("### Adversarial Robustness")
            lines.append("")
            lines.append("| Attack | Robustness |")
            lines.append("|--------|------------|")
            for attack, robustness in card.cybersecurity.adversarial_robustness.items():
                lines.append(f"| {attack} | {robustness:.2%} |")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(
            "*This model card was generated following FDA AI/ML guidance and Google Model Card format.*"
        )

        return "\n".join(lines)

    def load_from_json(self, json_path: str | Path) -> ModelCard:
        """Load a model card from JSON file."""
        path = Path(json_path)
        with open(path) as f:
            data = json.load(f)
        return self._dict_to_model_card(data)

    def _dict_to_model_card(self, data: dict[str, Any]) -> ModelCard:
        """Convert dictionary to ModelCard."""
        # This is a simplified implementation
        # In production, you'd want more robust deserialization
        return ModelCard(
            model_details=ModelDetails(**data["model_details"]),
            intended_use=IntendedUseDetails(
                primary_use=IntendedUse(data["intended_use"]["primary_use"]),
                primary_users=data["intended_use"]["primary_users"],
                use_cases=data["intended_use"]["use_cases"],
                out_of_scope_uses=data["intended_use"]["out_of_scope_uses"],
            ),
            training_data=TrainingDataDetails(
                dataset_type=DatasetType(data["training_data"]["dataset_type"]),
                **{
                    k: v
                    for k, v in data["training_data"].items()
                    if k != "dataset_type"
                },
            ),
            performance=PerformanceMetrics(**data["performance"]),
            ethical_considerations=EthicalConsiderations(
                **data["ethical_considerations"]
            ),
            regulatory=RegulatoryInformation(
                fda_risk_class=RiskLevel(data["regulatory"]["fda_risk_class"]),
                **{
                    k: v
                    for k, v in data["regulatory"].items()
                    if k != "fda_risk_class"
                },
            ),
            cybersecurity=CybersecurityInformation(**data["cybersecurity"]),
            additional_information=data.get("additional_information", {}),
            version_history=data.get("version_history", []),
        )


# =============================================================================
# Template Generator
# =============================================================================


def generate_template() -> ModelCard:
    """
    Generate a template model card with placeholder values.

    Returns:
        ModelCard with template values
    """
    generator = ModelCardGenerator()

    return generator.create_card(
        model_details=ModelDetails(
            name="[Model Name]",
            version="1.0.0",
            description="[Brief description of the model]",
            architecture="[e.g., ResNet-50, EfficientNet-B4]",
            framework="[e.g., PyTorch 2.0, TensorFlow 2.15]",
            developers=["[Developer Name]"],
            contact="[contact@example.com]",
        ),
        intended_use=IntendedUseDetails(
            primary_use=IntendedUse.DIAGNOSTIC,
            primary_users=["[e.g., radiologists, pathologists]"],
            use_cases=["[Specific use case 1]", "[Specific use case 2]"],
            out_of_scope_uses=[
                "[What the model should NOT be used for]",
                "Pediatric populations (if not validated)",
            ],
            clinical_workflow="[Where it fits in clinical workflow]",
        ),
        training_data=TrainingDataDetails(
            dataset_type=DatasetType.MIXED,
            dataset_name="[Dataset Name]",
            dataset_size=10000,
            collection_method="[How data was collected]",
            preprocessing=["Normalization", "Data augmentation"],
            demographics={
                "age_range": "[e.g., 18-85]",
                "sex_distribution": "[e.g., 48% female, 52% male]",
            },
            geographic_distribution=["[e.g., USA, EU]"],
            temporal_range="[e.g., 2018-2023]",
            known_biases=["[Known bias 1]"],
        ),
        performance=PerformanceMetrics(
            overall_metrics={
                "accuracy": 0.95,
                "sensitivity": 0.92,
                "specificity": 0.97,
                "auc": 0.98,
                "f1_score": 0.94,
            },
            subgroup_metrics={
                "female": {"accuracy": 0.94, "sensitivity": 0.91},
                "male": {"accuracy": 0.96, "sensitivity": 0.93},
            },
            failure_cases=[
                "[Known failure case 1]",
                "[Known failure case 2]",
            ],
        ),
        ethical_considerations=EthicalConsiderations(
            sensitive_attributes=["age", "sex", "race", "ethnicity"],
            fairness_metrics={
                "demographic_parity": 0.95,
                "equalized_odds": 0.92,
            },
            known_limitations=[
                "[Limitation 1]",
                "[Limitation 2]",
            ],
            mitigation_strategies=[
                "[Mitigation strategy 1]",
                "[Mitigation strategy 2]",
            ],
            human_oversight="[Required level of human oversight]",
        ),
        regulatory=RegulatoryInformation(
            fda_risk_class=RiskLevel.CLASS_II,
            fda_submission_type="510(k)",
            standards_compliance=[
                "IEC 62304:2006/AMD1:2015",
                "IEC 81001-5-1:2021",
                "ISO 14971:2019",
            ],
            pccp_enabled=True,
            lifecycle_monitoring=True,
        ),
        cybersecurity=CybersecurityInformation(
            threat_model="[Link to threat model documentation]",
            adversarial_testing=True,
            adversarial_robustness={
                "FGSM": 0.85,
                "PGD": 0.78,
                "C&W": 0.72,
            },
            data_poisoning_defense=True,
            model_integrity_verified=True,
            sbom_available=True,
            vulnerability_disclosure="See SECURITY.md",
            update_mechanism="[OTA updates / Manual updates]",
        ),
    )
