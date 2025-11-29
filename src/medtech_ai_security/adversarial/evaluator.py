"""
Robustness Evaluation Framework for Medical AI Models

Provides comprehensive evaluation of model robustness against various
adversarial attacks with metrics relevant to medical AI applications.

Evaluation Metrics:
- Clean accuracy: Performance on unperturbed images
- Robust accuracy: Performance under attack
- Attack success rate: Percentage of successful adversarial examples
- Perturbation metrics: L2, L-inf norms
- Clinical impact assessment: Severity of misclassifications

Reports:
- JSON report with all metrics
- Vulnerability assessment
- Defense recommendations

References:
- https://www.sciencedirect.com/science/article/pii/S2211568425001044 (Adversarial AI in radiology 2025)
- https://dl.acm.org/doi/10.1145/3702638 (ACM survey on medical image adversarial)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from medtech_ai_security.adversarial.attacks import (
    AdversarialAttacker,
    AttackResult,
    AttackType,
)
from medtech_ai_security.adversarial.defenses import (
    AdversarialDefender,
    DefenseResult,
    DefenseType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RobustnessReport:
    """Comprehensive robustness evaluation report."""

    model_name: str
    evaluation_date: str
    clean_accuracy: float
    attack_results: dict = field(default_factory=dict)
    defense_results: dict = field(default_factory=dict)
    vulnerability_assessment: str = ""
    recommendations: list = field(default_factory=list)
    clinical_risk_level: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "evaluation_date": self.evaluation_date,
            "clean_accuracy": float(self.clean_accuracy),
            "attack_results": self.attack_results,
            "defense_results": self.defense_results,
            "vulnerability_assessment": self.vulnerability_assessment,
            "recommendations": self.recommendations,
            "clinical_risk_level": self.clinical_risk_level,
            "metadata": self.metadata,
        }

    def save(self, path: Path | str) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved robustness report to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "RobustnessReport":
        """Load report from JSON file."""
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluator for medical AI models.

    Evaluates model robustness against multiple attack types
    and generates detailed vulnerability reports.

    Attributes:
        model: Target model to evaluate
        model_name: Name for reporting
        num_classes: Number of output classes
    """

    # Clinical severity mapping for medical imaging
    CLINICAL_SEVERITY = {
        "malignant_to_benign": "CRITICAL",  # False negative - missed cancer
        "benign_to_malignant": "HIGH",  # False positive - unnecessary treatment
        "wrong_class": "MEDIUM",  # General misclassification
        "correct": "NONE",  # No clinical impact
    }

    def __init__(
        self,
        model: Callable,
        model_name: str = "medical_ai_model",
        num_classes: int = 2,
        class_names: Optional[list] = None,
    ):
        """
        Initialize robustness evaluator.

        Args:
            model: Target model to evaluate
            model_name: Name for reporting
            num_classes: Number of output classes
            class_names: Optional class names (e.g., ["benign", "malignant"])
        """
        self.model = model
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        self.attacker = AdversarialAttacker(
            model, num_classes=num_classes
        )
        self.defender = AdversarialDefender(model)

    def evaluate_clean_accuracy(
        self,
        images: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute accuracy on clean (unperturbed) images."""
        predictions = self.model(images)
        if hasattr(predictions, "numpy"):
            predictions = predictions.numpy()

        if len(predictions.shape) == 1 or predictions.shape[-1] == 1:
            pred_classes = (np.squeeze(predictions) > 0.5).astype(int)
        else:
            pred_classes = np.argmax(predictions, axis=1)

        return float(np.mean(pred_classes == labels))

    def evaluate_attack(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        attack_type: AttackType,
        **attack_params,
    ) -> dict:
        """
        Evaluate model robustness against specific attack.

        Args:
            images: Test images
            labels: True labels
            attack_type: Type of attack to evaluate
            **attack_params: Attack-specific parameters

        Returns:
            Dictionary with attack metrics
        """
        logger.info(f"Evaluating {attack_type.value} attack...")

        result = self.attacker.attack(
            images, labels, attack_type, **attack_params
        )

        # Compute robust accuracy
        adversarial_preds = self.model(result.adversarial_images)
        if hasattr(adversarial_preds, "numpy"):
            adversarial_preds = adversarial_preds.numpy()

        if len(adversarial_preds.shape) == 1 or adversarial_preds.shape[-1] == 1:
            adv_classes = (np.squeeze(adversarial_preds) > 0.5).astype(int)
        else:
            adv_classes = np.argmax(adversarial_preds, axis=1)

        robust_accuracy = float(np.mean(adv_classes == labels))

        return {
            "attack_type": attack_type.value,
            "attack_params": attack_params,
            "success_rate": result.success_rate,
            "robust_accuracy": robust_accuracy,
            "mean_perturbation_l2": result.mean_perturbation_l2,
            "mean_perturbation_linf": result.mean_perturbation_linf,
            "num_samples": result.num_samples,
        }

    def evaluate_defense(
        self,
        clean_images: np.ndarray,
        clean_labels: np.ndarray,
        adversarial_images: np.ndarray,
        defense_type: DefenseType,
        **defense_params,
    ) -> dict:
        """
        Evaluate effectiveness of a defense.

        Args:
            clean_images: Clean test images
            clean_labels: True labels
            adversarial_images: Adversarial examples
            defense_type: Type of defense to evaluate
            **defense_params: Defense-specific parameters

        Returns:
            Dictionary with defense metrics
        """
        logger.info(f"Evaluating {defense_type.value} defense...")

        # Get defense function
        defense_map = {
            DefenseType.JPEG_COMPRESSION: self.defender.jpeg_compression,
            DefenseType.BIT_DEPTH_REDUCTION: self.defender.bit_depth_reduction,
            DefenseType.GAUSSIAN_BLUR: self.defender.gaussian_blur,
            DefenseType.SPATIAL_SMOOTHING: self.defender.spatial_smoothing,
            DefenseType.FEATURE_SQUEEZING: self.defender.feature_squeezing,
            DefenseType.ENSEMBLE: self.defender.ensemble_defense,
        }

        defense_fn = defense_map.get(defense_type)
        if defense_fn is None:
            logger.warning(f"Defense {defense_type} not implemented")
            return {}

        result = self.defender.evaluate_defense(
            clean_images,
            clean_labels,
            adversarial_images,
            defense_type,
            defense_fn,
            defense_params,
        )

        return result.to_dict()

    def assess_clinical_impact(
        self,
        original_labels: np.ndarray,
        adversarial_predictions: np.ndarray,
    ) -> dict:
        """
        Assess clinical impact of adversarial misclassifications.

        For medical imaging, misclassification severity depends on
        the type of error (false negative vs false positive).

        Args:
            original_labels: True labels
            adversarial_predictions: Predictions on adversarial images

        Returns:
            Dictionary with clinical impact assessment
        """
        if hasattr(adversarial_predictions, "numpy"):
            adversarial_predictions = adversarial_predictions.numpy()

        if len(adversarial_predictions.shape) == 1 or adversarial_predictions.shape[-1] == 1:
            pred_classes = (np.squeeze(adversarial_predictions) > 0.5).astype(int)
        else:
            pred_classes = np.argmax(adversarial_predictions, axis=1)

        # Count misclassification types
        impact_counts = {
            "critical": 0,  # Malignant -> Benign (missed cancer)
            "high": 0,  # Benign -> Malignant (false alarm)
            "medium": 0,  # Other misclassifications
            "none": 0,  # Correct
        }

        for true_label, pred_label in zip(original_labels, pred_classes):
            if true_label == pred_label:
                impact_counts["none"] += 1
            elif self.num_classes == 2:
                # Binary classification (assuming class 1 = positive/malignant)
                if true_label == 1 and pred_label == 0:
                    impact_counts["critical"] += 1  # False negative
                elif true_label == 0 and pred_label == 1:
                    impact_counts["high"] += 1  # False positive
                else:
                    impact_counts["medium"] += 1
            else:
                impact_counts["medium"] += 1

        total = len(original_labels)
        impact_rates = {k: v / total for k, v in impact_counts.items()}

        # Overall risk level
        if impact_rates["critical"] > 0.05:
            risk_level = "CRITICAL"
        elif impact_rates["high"] > 0.10:
            risk_level = "HIGH"
        elif impact_rates["medium"] > 0.20:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "impact_counts": impact_counts,
            "impact_rates": impact_rates,
            "risk_level": risk_level,
        }

    def generate_recommendations(
        self,
        attack_results: dict,
        defense_results: dict,
        clinical_impact: dict,
    ) -> list:
        """
        Generate defense recommendations based on evaluation.

        Args:
            attack_results: Results from attack evaluations
            defense_results: Results from defense evaluations
            clinical_impact: Clinical impact assessment

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check vulnerability to attacks
        for attack_name, result in attack_results.items():
            if result.get("success_rate", 0) > 0.5:
                recommendations.append(
                    f"HIGH VULNERABILITY: {attack_name} achieves {result['success_rate']:.0%} "
                    f"success rate. Implement adversarial training with {attack_name} examples."
                )
            elif result.get("success_rate", 0) > 0.2:
                recommendations.append(
                    f"MODERATE VULNERABILITY: {attack_name} achieves {result['success_rate']:.0%} "
                    f"success rate. Consider input preprocessing defenses."
                )

        # Check perturbation magnitudes
        for attack_name, result in attack_results.items():
            if result.get("mean_perturbation_linf", 1) < 0.01:
                recommendations.append(
                    f"SMALL PERTURBATION WARNING: {attack_name} succeeds with "
                    f"L-inf={result['mean_perturbation_linf']:.4f}. "
                    f"Model is sensitive to imperceptible changes."
                )

        # Check clinical impact
        if clinical_impact.get("risk_level") == "CRITICAL":
            recommendations.append(
                "CRITICAL CLINICAL RISK: Adversarial attacks cause significant "
                "false negatives (missed diagnoses). Immediate remediation required."
            )
        elif clinical_impact.get("risk_level") == "HIGH":
            recommendations.append(
                "HIGH CLINICAL RISK: Adversarial attacks cause significant "
                "false positives. Review clinical workflow safeguards."
            )

        # Defense effectiveness
        best_defense = None
        best_gain = 0
        for defense_name, result in defense_results.items():
            gain = result.get("accuracy_gain_adversarial", 0)
            if gain > best_gain:
                best_gain = gain
                best_defense = defense_name

        if best_defense and best_gain > 0.1:
            recommendations.append(
                f"RECOMMENDED DEFENSE: {best_defense} improves adversarial accuracy "
                f"by {best_gain:.0%}. Consider deployment."
            )

        if not recommendations:
            recommendations.append(
                "Model shows reasonable robustness. Continue monitoring with "
                "regular adversarial testing."
            )

        return recommendations

    def full_evaluation(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        attack_configs: Optional[list] = None,
        defense_configs: Optional[list] = None,
    ) -> RobustnessReport:
        """
        Run comprehensive robustness evaluation.

        Args:
            images: Test images
            labels: True labels
            attack_configs: List of (attack_type, params) tuples
            defense_configs: List of (defense_type, params) tuples

        Returns:
            RobustnessReport with all metrics
        """
        logger.info(f"Starting full robustness evaluation for {self.model_name}")

        # Default attack configurations
        if attack_configs is None:
            attack_configs = [
                (AttackType.FGSM, {"epsilon": 0.01}),
                (AttackType.FGSM, {"epsilon": 0.03}),
                (AttackType.PGD, {"epsilon": 0.01, "alpha": 0.001, "num_iterations": 40}),
                (AttackType.PGD, {"epsilon": 0.03, "alpha": 0.003, "num_iterations": 40}),
            ]

        # Default defense configurations
        if defense_configs is None:
            defense_configs = [
                (DefenseType.JPEG_COMPRESSION, {"quality": 75}),
                (DefenseType.BIT_DEPTH_REDUCTION, {"bits": 5}),
                (DefenseType.GAUSSIAN_BLUR, {"sigma": 1.0}),
                (DefenseType.FEATURE_SQUEEZING, {"bit_depth": 5, "blur_sigma": 0.5}),
            ]

        # Evaluate clean accuracy
        clean_accuracy = self.evaluate_clean_accuracy(images, labels)
        logger.info(f"Clean accuracy: {clean_accuracy:.2%}")

        # Evaluate attacks
        attack_results = {}
        strongest_attack_images = None
        worst_success_rate = 0

        for attack_type, params in attack_configs:
            key = f"{attack_type.value}_{params.get('epsilon', '')}"
            result = self.evaluate_attack(images, labels, attack_type, **params)
            attack_results[key] = result

            # Track strongest attack for defense testing
            if result["success_rate"] > worst_success_rate:
                worst_success_rate = result["success_rate"]
                attack_result = self.attacker.attack(
                    images, labels, attack_type, **params
                )
                strongest_attack_images = attack_result.adversarial_images

        # Evaluate defenses using strongest attack
        defense_results = {}
        if strongest_attack_images is not None:
            for defense_type, params in defense_configs:
                key = f"{defense_type.value}"
                result = self.evaluate_defense(
                    images, labels, strongest_attack_images, defense_type, **params
                )
                defense_results[key] = result

        # Clinical impact assessment
        if strongest_attack_images is not None:
            adv_preds = self.model(strongest_attack_images)
            clinical_impact = self.assess_clinical_impact(labels, adv_preds)
        else:
            clinical_impact = {"risk_level": "LOW"}

        # Generate recommendations
        recommendations = self.generate_recommendations(
            attack_results, defense_results, clinical_impact
        )

        # Vulnerability assessment
        vulnerable_attacks = [
            k for k, v in attack_results.items() if v.get("success_rate", 0) > 0.3
        ]
        if len(vulnerable_attacks) > 2:
            vulnerability_assessment = (
                "HIGH VULNERABILITY: Model is susceptible to multiple attack types. "
                "Adversarial training strongly recommended."
            )
        elif len(vulnerable_attacks) > 0:
            vulnerability_assessment = (
                "MODERATE VULNERABILITY: Model shows weakness to some attacks. "
                "Consider defensive measures."
            )
        else:
            vulnerability_assessment = (
                "LOW VULNERABILITY: Model demonstrates reasonable robustness. "
                "Continue regular testing."
            )

        report = RobustnessReport(
            model_name=self.model_name,
            evaluation_date=datetime.now().isoformat(),
            clean_accuracy=clean_accuracy,
            attack_results=attack_results,
            defense_results=defense_results,
            vulnerability_assessment=vulnerability_assessment,
            recommendations=recommendations,
            clinical_risk_level=clinical_impact.get("risk_level", "UNKNOWN"),
            metadata={
                "num_samples": len(images),
                "num_classes": self.num_classes,
                "class_names": self.class_names,
                "clinical_impact": clinical_impact,
            },
        )

        logger.info(f"Evaluation complete. Vulnerability: {vulnerability_assessment}")
        return report


def main():
    """CLI entry point for adversarial robustness evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate adversarial robustness of medical AI models"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Attack command
    attack_parser = subparsers.add_parser("attack", help="Run adversarial attacks")
    attack_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model file (.keras or .h5)",
    )
    attack_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test data directory",
    )
    attack_parser.add_argument(
        "--attack",
        type=str,
        choices=["fgsm", "pgd", "cw_l2"],
        default="pgd",
        help="Attack type",
    )
    attack_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Perturbation magnitude",
    )
    attack_parser.add_argument(
        "--output",
        type=str,
        help="Output path for adversarial examples",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Full robustness evaluation")
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model file",
    )
    eval_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test data directory",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default="reports/robustness_report.json",
        help="Output path for report",
    )
    eval_parser.add_argument(
        "--model-name",
        type=str,
        default="medical_ai_model",
        help="Model name for report",
    )

    # Defend command
    defend_parser = subparsers.add_parser("defend", help="Apply defenses")
    defend_parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to adversarial images",
    )
    defend_parser.add_argument(
        "--defense",
        type=str,
        choices=["jpeg", "blur", "squeeze"],
        default="squeeze",
        help="Defense type",
    )
    defend_parser.add_argument(
        "--output",
        type=str,
        help="Output path for defended images",
    )

    args = parser.parse_args()

    if args.command == "attack":
        print("\n" + "=" * 60)
        print("ADVERSARIAL ATTACK")
        print("=" * 60)

        try:
            import tensorflow as tf

            # Load model
            model = tf.keras.models.load_model(args.model)
            print(f"[+] Loaded model from {args.model}")

            # Load data
            data_path = Path(args.data)
            images = np.load(data_path / "images.npy")
            labels = np.load(data_path / "labels.npy")
            print(f"[+] Loaded {len(images)} test images")

            # Run attack
            attacker = AdversarialAttacker(model)
            attack_type = AttackType(args.attack)

            if attack_type == AttackType.FGSM:
                result = attacker.fgsm(images, labels, epsilon=args.epsilon)
            elif attack_type == AttackType.PGD:
                result = attacker.pgd(images, labels, epsilon=args.epsilon)
            else:
                result = attacker.cw_l2(images, labels)

            print(f"\n[+] Attack Results:")
            print(f"    Success Rate: {result.success_rate:.2%}")
            print(f"    Mean L2 Perturbation: {result.mean_perturbation_l2:.4f}")
            print(f"    Mean L-inf Perturbation: {result.mean_perturbation_linf:.4f}")

            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                np.save(output_path / "adversarial_images.npy", result.adversarial_images)
                np.save(output_path / "perturbations.npy", result.perturbations)
                print(f"\n[+] Saved adversarial examples to {args.output}")

        except ImportError:
            print("[-] TensorFlow required for model loading")

    elif args.command == "evaluate":
        print("\n" + "=" * 60)
        print("ROBUSTNESS EVALUATION")
        print("=" * 60)

        try:
            import tensorflow as tf

            # Load model
            model = tf.keras.models.load_model(args.model)
            print(f"[+] Loaded model from {args.model}")

            # Load data
            data_path = Path(args.data)
            images = np.load(data_path / "images.npy")
            labels = np.load(data_path / "labels.npy")
            print(f"[+] Loaded {len(images)} test images")

            # Run evaluation
            evaluator = RobustnessEvaluator(
                model, model_name=args.model_name
            )
            report = evaluator.full_evaluation(images, labels)

            # Print summary
            print(f"\n{'=' * 60}")
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"Clean Accuracy: {report.clean_accuracy:.2%}")
            print(f"Vulnerability: {report.vulnerability_assessment}")
            print(f"Clinical Risk: {report.clinical_risk_level}")
            print(f"\nAttack Results:")
            for name, result in report.attack_results.items():
                print(f"  {name}: {result['success_rate']:.2%} success, "
                      f"{result['robust_accuracy']:.2%} robust accuracy")
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")

            # Save report
            report.save(args.output)
            print(f"\n[+] Report saved to {args.output}")

        except ImportError:
            print("[-] TensorFlow required for model loading")

    elif args.command == "defend":
        print("\n" + "=" * 60)
        print("APPLY DEFENSE")
        print("=" * 60)

        # Load adversarial images
        images = np.load(args.images)
        print(f"[+] Loaded {len(images)} images")

        defender = AdversarialDefender()

        if args.defense == "jpeg":
            defended = defender.jpeg_compression(images, quality=75)
        elif args.defense == "blur":
            defended = defender.gaussian_blur(images, sigma=1.0)
        else:
            defended = defender.feature_squeezing(images)

        print(f"[+] Applied {args.defense} defense")

        if args.output:
            np.save(args.output, defended)
            print(f"[+] Saved defended images to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
