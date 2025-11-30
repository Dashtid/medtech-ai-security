"""Unit tests for Phase 4: Adversarial ML Testing."""

import numpy as np
import pytest

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
from medtech_ai_security.adversarial.evaluator import (
    RobustnessEvaluator,
    RobustnessReport,
)


class TestAttackType:
    """Test AttackType enumeration."""

    def test_attack_types_exist(self):
        """Test attack types are defined."""
        assert AttackType.FGSM.value == "fgsm"
        assert AttackType.PGD.value == "pgd"

    def test_all_attack_types(self):
        """Test all attack types have string values."""
        for attack in AttackType:
            assert isinstance(attack.value, str)
            assert len(attack.value) > 0


class TestDefenseType:
    """Test DefenseType enumeration."""

    def test_defense_types_exist(self):
        """Test defense types are defined."""
        assert DefenseType.GAUSSIAN_BLUR.value == "gaussian_blur"
        assert DefenseType.JPEG_COMPRESSION.value == "jpeg_compression"
        assert DefenseType.FEATURE_SQUEEZING.value == "feature_squeezing"


class TestAdversarialAttacker:
    """Test AdversarialAttacker functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""

        def model(x):
            # Simple thresholding model
            mean_val = np.mean(x, axis=(1, 2, 3))
            probs = np.column_stack([1 - mean_val, mean_val])
            return probs

        return model

    @pytest.fixture
    def attacker(self, simple_model):
        """Create an attacker instance."""
        return AdversarialAttacker(
            model=simple_model,
            clip_min=0.0,
            clip_max=1.0,
            num_classes=2,
        )

    @pytest.fixture
    def sample_images(self):
        """Generate sample images."""
        np.random.seed(42)
        return np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.5

    @pytest.fixture
    def sample_labels(self):
        """Generate sample labels."""
        return np.array([0] * 10)

    def test_attacker_initialization(self, attacker):
        """Test attacker initializes correctly."""
        assert attacker.clip_min == 0.0
        assert attacker.clip_max == 1.0
        assert attacker.num_classes == 2

    def test_fgsm_attack(self, attacker, sample_images, sample_labels):
        """Test FGSM attack."""
        result = attacker.fgsm(
            images=sample_images,
            labels=sample_labels,
            epsilon=0.03,
        )

        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.FGSM
        assert result.adversarial_images.shape == sample_images.shape
        # Perturbation should be bounded
        assert result.mean_perturbation_linf <= 0.03 + 1e-5

    def test_pgd_attack(self, attacker, sample_images, sample_labels):
        """Test PGD attack."""
        result = attacker.pgd(
            images=sample_images,
            labels=sample_labels,
            epsilon=0.03,
            alpha=0.007,
            num_iterations=5,
        )

        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.PGD
        assert result.adversarial_images.shape == sample_images.shape

    def test_adversarial_clipping(self, attacker, sample_images, sample_labels):
        """Test that adversarial examples are clipped correctly."""
        result = attacker.fgsm(
            images=sample_images,
            labels=sample_labels,
            epsilon=0.5,  # Large epsilon
        )

        # All values should be in valid range
        assert np.all(result.adversarial_images >= 0.0)
        assert np.all(result.adversarial_images <= 1.0)


class TestAdversarialDefender:
    """Test AdversarialDefender functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            probs = np.column_stack([1 - mean_val, mean_val])
            return probs

        return model

    @pytest.fixture
    def defender(self, simple_model):
        """Create a defender instance."""
        return AdversarialDefender(model=simple_model)

    @pytest.fixture
    def sample_images(self):
        """Generate sample adversarial images."""
        np.random.seed(42)
        return np.random.rand(10, 28, 28, 1).astype(np.float32)

    def test_gaussian_blur_defense(self, defender, sample_images):
        """Test Gaussian blur defense."""
        defended = defender.gaussian_blur(
            images=sample_images,
            sigma=1.0,
        )

        assert defended.shape == sample_images.shape
        # Blur should smooth the image (reduce variance)
        assert np.var(defended) <= np.var(sample_images) * 1.1

    def test_jpeg_compression_defense(self, defender, sample_images):
        """Test JPEG compression defense."""
        defended = defender.jpeg_compression(
            images=sample_images,
            quality=75,
        )

        assert defended.shape == sample_images.shape
        # Values should still be in valid range
        assert np.all(defended >= 0)
        assert np.all(defended <= 1)

    def test_feature_squeezing_defense(self, defender, sample_images):
        """Test feature squeezing defense."""
        defended = defender.feature_squeezing(
            images=sample_images,
            bit_depth=4,
        )

        assert defended.shape == sample_images.shape


class TestRobustnessReport:
    """Test RobustnessReport dataclass."""

    def test_report_creation(self):
        """Test creating a robustness report."""
        report = RobustnessReport(
            model_name="test_model",
            evaluation_date="2024-01-15",
            clean_accuracy=0.95,
        )

        assert report.model_name == "test_model"
        assert report.clean_accuracy == 0.95

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = RobustnessReport(
            model_name="test_model",
            evaluation_date="2024-01-15",
            clean_accuracy=0.95,
            attack_results={"fgsm": {"success_rate": 0.7}},
        )

        result_dict = report.to_dict()

        assert result_dict["model_name"] == "test_model"
        assert result_dict["clean_accuracy"] == 0.95
        assert "attack_results" in result_dict

    def test_report_save_and_load(self, tmp_path):
        """Test saving and loading report to/from file."""
        report = RobustnessReport(
            model_name="medical_classifier",
            evaluation_date="2024-01-15",
            clean_accuracy=0.92,
            attack_results={
                "fgsm": {"success_rate": 0.65, "robust_accuracy": 0.35},
                "pgd": {"success_rate": 0.78, "robust_accuracy": 0.22},
            },
            defense_results={
                "gaussian_blur": {"defended_accuracy": 0.75},
            },
            vulnerability_assessment="Model vulnerable to gradient-based attacks",
            recommendations=["Apply adversarial training", "Use input preprocessing"],
            clinical_risk_level="HIGH",
            metadata={"total_samples": 1000},
        )

        # Save
        report_path = tmp_path / "robustness_report.json"
        report.save(report_path)

        assert report_path.exists()

        # Load
        loaded = RobustnessReport.load(report_path)

        assert loaded.model_name == "medical_classifier"
        assert loaded.clean_accuracy == 0.92
        assert loaded.attack_results["fgsm"]["success_rate"] == 0.65
        assert loaded.clinical_risk_level == "HIGH"
        assert len(loaded.recommendations) == 2

    def test_report_with_all_fields(self):
        """Test report with all optional fields populated."""
        report = RobustnessReport(
            model_name="xray_classifier",
            evaluation_date="2024-01-20",
            clean_accuracy=0.88,
            attack_results={
                "fgsm": {"success_rate": 0.55, "mean_perturbation": 0.03},
                "pgd": {"success_rate": 0.70, "mean_perturbation": 0.025},
                "cw": {"success_rate": 0.80, "mean_perturbation": 0.01},
            },
            defense_results={
                "jpeg": {"defended_accuracy": 0.70},
                "blur": {"defended_accuracy": 0.72},
            },
            vulnerability_assessment="High vulnerability to optimization-based attacks",
            recommendations=[
                "Implement adversarial training",
                "Add input preprocessing",
                "Consider ensemble methods",
            ],
            clinical_risk_level="CRITICAL",
            metadata={
                "dataset": "chest_xray",
                "num_samples": 5000,
                "evaluation_time": 120.5,
            },
        )

        result_dict = report.to_dict()

        assert len(result_dict["attack_results"]) == 3
        assert len(result_dict["defense_results"]) == 2
        assert len(result_dict["recommendations"]) == 3
        assert result_dict["metadata"]["num_samples"] == 5000


class TestRobustnessEvaluator:
    """Test RobustnessEvaluator functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for evaluation."""

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            probs = np.column_stack([1 - mean_val, mean_val])
            return probs

        return model

    @pytest.fixture
    def evaluator(self, simple_model):
        """Create an evaluator instance."""
        return RobustnessEvaluator(
            model=simple_model,
            model_name="test_classifier",
            num_classes=2,
        )

    @pytest.fixture
    def sample_images(self):
        """Generate sample images."""
        np.random.seed(42)
        return np.random.rand(20, 28, 28, 1).astype(np.float32) * 0.4

    @pytest.fixture
    def sample_labels(self):
        """Generate sample labels."""
        return np.array([0] * 20)

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly."""
        assert evaluator.model_name == "test_classifier"
        assert evaluator.num_classes == 2

    def test_evaluate_clean_accuracy(self, evaluator, sample_images, sample_labels):
        """Test evaluating clean accuracy."""
        accuracy = evaluator.evaluate_clean_accuracy(
            images=sample_images,
            labels=sample_labels,
        )

        assert 0 <= accuracy <= 1

    def test_evaluate_attack(self, evaluator, sample_images, sample_labels):
        """Test evaluating a single attack."""
        result = evaluator.evaluate_attack(
            images=sample_images,
            labels=sample_labels,
            attack_type=AttackType.FGSM,
            epsilon=0.03,
        )

        assert isinstance(result, dict)
        assert "attack_type" in result
        assert "success_rate" in result

    def test_assess_clinical_impact(self, evaluator, sample_labels):
        """Test clinical impact assessment."""
        # Simulate adversarial predictions (some misclassifications)
        adversarial_preds = np.zeros((20, 2))
        adversarial_preds[:15, 0] = 1.0  # Correct predictions
        adversarial_preds[15:, 1] = 1.0  # Misclassifications

        impact = evaluator.assess_clinical_impact(
            original_labels=sample_labels,
            adversarial_predictions=adversarial_preds,
        )

        assert "risk_level" in impact
        assert "impact_counts" in impact
        assert impact["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TestIntegration:
    """Integration tests for adversarial module."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple binary classifier."""

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            probs = np.column_stack([1 - mean_val, mean_val])
            return probs

        return model

    def test_attack_and_defend_pipeline(self, simple_model):
        """Test complete attack and defense pipeline."""
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 10)

        # Attack
        attacker = AdversarialAttacker(model=simple_model)
        attack_result = attacker.fgsm(
            images=images,
            labels=labels,
            epsilon=0.05,
        )

        # Defend
        defender = AdversarialDefender(model=simple_model)
        defended_images = defender.gaussian_blur(
            images=attack_result.adversarial_images,
            sigma=1.0,
        )

        # Verify pipeline completed
        assert attack_result.adversarial_images.shape == images.shape
        assert defended_images.shape == images.shape

    def test_multiple_attacks_comparison(self, simple_model):
        """Test comparing multiple attack methods."""
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 10)

        attacker = AdversarialAttacker(model=simple_model)

        # Run FGSM
        fgsm_result = attacker.fgsm(
            images=images,
            labels=labels,
            epsilon=0.03,
        )

        # Run PGD
        pgd_result = attacker.pgd(
            images=images,
            labels=labels,
            epsilon=0.03,
            alpha=0.007,
            num_iterations=5,
        )

        # Both should produce valid results
        assert fgsm_result.attack_type == AttackType.FGSM
        assert pgd_result.attack_type == AttackType.PGD


class TestDefenseResult:
    """Test DefenseResult dataclass."""

    def test_defense_result_creation(self):
        """Test creating a DefenseResult."""
        result = DefenseResult(
            original_images=np.zeros((10, 28, 28, 1)),
            defended_images=np.zeros((10, 28, 28, 1)),
            defense_type=DefenseType.GAUSSIAN_BLUR,
            defense_params={"sigma": 1.0},
            clean_accuracy_before=0.95,
            clean_accuracy_after=0.93,
            adversarial_accuracy_before=0.30,
            adversarial_accuracy_after=0.65,
            accuracy_drop_clean=0.02,
            accuracy_gain_adversarial=0.35,
        )

        assert result.defense_type == DefenseType.GAUSSIAN_BLUR
        assert result.accuracy_gain_adversarial == 0.35

    def test_defense_result_to_dict(self):
        """Test converting DefenseResult to dictionary."""
        result = DefenseResult(
            original_images=np.zeros((5, 28, 28, 1)),
            defended_images=np.zeros((5, 28, 28, 1)),
            defense_type=DefenseType.JPEG_COMPRESSION,
            defense_params={"quality": 75},
            clean_accuracy_before=0.92,
            clean_accuracy_after=0.90,
            adversarial_accuracy_before=0.25,
            adversarial_accuracy_after=0.70,
            accuracy_drop_clean=0.02,
            accuracy_gain_adversarial=0.45,
        )

        result_dict = result.to_dict()

        assert result_dict["defense_type"] == "jpeg_compression"
        assert result_dict["clean_accuracy_before"] == 0.92
        assert result_dict["accuracy_gain_adversarial"] == 0.45


class TestAdversarialDefenderAdvanced:
    """Advanced tests for AdversarialDefender."""

    @pytest.fixture
    def sample_images(self):
        """Generate sample images."""
        np.random.seed(42)
        return np.random.rand(10, 28, 28, 1).astype(np.float32)

    def test_bit_depth_reduction(self):
        """Test bit depth reduction defense."""
        defender = AdversarialDefender()
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)

        defended = defender.bit_depth_reduction(images, bits=4)

        assert defended.shape == images.shape
        # Reduced bit depth means fewer unique values
        assert np.all(defended >= 0)
        assert np.all(defended <= 1)

    def test_bit_depth_reduction_255_range(self):
        """Test bit depth reduction with 0-255 range images."""
        defender = AdversarialDefender()
        np.random.seed(42)
        images = (np.random.rand(5, 28, 28, 1) * 255).astype(np.float32)

        defended = defender.bit_depth_reduction(images, bits=5)

        assert defended.shape == images.shape
        assert np.all(defended >= 0)
        assert np.all(defended <= 255)

    def test_spatial_smoothing(self, sample_images):
        """Test spatial smoothing (median filter) defense."""
        defender = AdversarialDefender()

        defended = defender.spatial_smoothing(sample_images, kernel_size=3)

        assert defended.shape == sample_images.shape

    def test_ensemble_defense_default(self, sample_images):
        """Test ensemble defense with default settings."""
        defender = AdversarialDefender()

        defended = defender.ensemble_defense(sample_images)

        assert defended.shape == sample_images.shape
        assert defended.dtype == np.float32

    def test_ensemble_defense_custom(self, sample_images):
        """Test ensemble defense with custom defenses."""
        defender = AdversarialDefender()

        custom_defenses = [
            (defender.gaussian_blur, {"sigma": 0.5}),
            (defender.bit_depth_reduction, {"bits": 6}),
        ]

        defended = defender.ensemble_defense(sample_images, defenses=custom_defenses)

        assert defended.shape == sample_images.shape

    def test_detect_adversarial(self):
        """Test adversarial detection."""

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])

        defender = AdversarialDefender(model=model)
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32)

        is_adv, scores = defender.detect_adversarial(images, threshold=0.1)

        assert len(is_adv) == 10
        assert len(scores) == 10
        assert is_adv.dtype == bool

    def test_detect_adversarial_no_model(self):
        """Test adversarial detection raises error without model."""
        defender = AdversarialDefender(model=None)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)

        with pytest.raises(ValueError, match="Model required"):
            defender.detect_adversarial(images)

    def test_evaluate_defense(self):
        """Test defense evaluation."""

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])

        defender = AdversarialDefender(model=model)
        np.random.seed(42)
        clean_images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        clean_labels = np.array([0] * 10)
        adv_images = clean_images + 0.1  # Simple perturbation

        result = defender.evaluate_defense(
            clean_images=clean_images,
            clean_labels=clean_labels,
            adversarial_images=adv_images,
            defense_type=DefenseType.GAUSSIAN_BLUR,
            defense_fn=defender.gaussian_blur,
            defense_params={"sigma": 1.0},
        )

        assert isinstance(result, DefenseResult)
        assert 0 <= result.clean_accuracy_before <= 1
        assert 0 <= result.clean_accuracy_after <= 1

    def test_evaluate_defense_no_model(self):
        """Test defense evaluation raises error without model."""
        defender = AdversarialDefender(model=None)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)
        labels = np.array([0] * 5)

        with pytest.raises(ValueError, match="Model required"):
            defender.evaluate_defense(
                images, labels, images,
                DefenseType.GAUSSIAN_BLUR,
                defender.gaussian_blur,
                {"sigma": 1.0},
            )


class TestRobustnessEvaluatorAdvanced:
    """Advanced tests for RobustnessEvaluator."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])

        return model

    @pytest.fixture
    def evaluator(self, simple_model):
        """Create an evaluator."""
        return RobustnessEvaluator(
            model=simple_model,
            model_name="test_model",
            num_classes=2,
            class_names=["benign", "malignant"],
        )

    def test_evaluator_with_custom_class_names(self, simple_model):
        """Test evaluator with custom class names."""
        evaluator = RobustnessEvaluator(
            model=simple_model,
            model_name="medical_classifier",
            num_classes=2,
            class_names=["normal", "abnormal"],
        )

        assert evaluator.class_names == ["normal", "abnormal"]

    def test_evaluate_clean_accuracy_binary(self, evaluator):
        """Test clean accuracy with binary predictions."""
        np.random.seed(42)
        images = np.random.rand(20, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 20)

        accuracy = evaluator.evaluate_clean_accuracy(images, labels)

        assert 0 <= accuracy <= 1

    def test_generate_recommendations_high_vulnerability(self, evaluator):
        """Test recommendation generation for high vulnerability."""
        attack_results = {
            "fgsm": {"success_rate": 0.75, "mean_perturbation_linf": 0.005},
            "pgd": {"success_rate": 0.60, "mean_perturbation_linf": 0.01},
        }
        defense_results = {
            "gaussian_blur": {"accuracy_gain_adversarial": 0.25},
        }
        clinical_impact = {"risk_level": "CRITICAL"}

        recommendations = evaluator.generate_recommendations(
            attack_results, defense_results, clinical_impact
        )

        assert len(recommendations) > 0
        # Should have recommendations for high vulnerability
        assert any("HIGH VULNERABILITY" in r or "CRITICAL" in r for r in recommendations)

    def test_generate_recommendations_moderate_vulnerability(self, evaluator):
        """Test recommendation generation for moderate vulnerability."""
        attack_results = {
            "fgsm": {"success_rate": 0.35},
        }
        defense_results = {}
        clinical_impact = {"risk_level": "MEDIUM"}

        recommendations = evaluator.generate_recommendations(
            attack_results, defense_results, clinical_impact
        )

        assert len(recommendations) > 0
        assert any("MODERATE" in r for r in recommendations)

    def test_generate_recommendations_low_vulnerability(self, evaluator):
        """Test recommendation generation when model is robust."""
        attack_results = {
            "fgsm": {"success_rate": 0.10},
        }
        defense_results = {}
        clinical_impact = {"risk_level": "LOW"}

        recommendations = evaluator.generate_recommendations(
            attack_results, defense_results, clinical_impact
        )

        assert len(recommendations) > 0
        assert any("reasonable robustness" in r.lower() for r in recommendations)

    def test_generate_recommendations_best_defense(self, evaluator):
        """Test recommendation generation highlights best defense."""
        attack_results = {"fgsm": {"success_rate": 0.5}}
        defense_results = {
            "gaussian_blur": {"accuracy_gain_adversarial": 0.15},
            "jpeg": {"accuracy_gain_adversarial": 0.05},
        }
        clinical_impact = {"risk_level": "MEDIUM"}

        recommendations = evaluator.generate_recommendations(
            attack_results, defense_results, clinical_impact
        )

        assert any("gaussian_blur" in r for r in recommendations)

    def test_assess_clinical_impact_multiclass(self, simple_model):
        """Test clinical impact assessment for multiclass."""
        evaluator = RobustnessEvaluator(
            model=simple_model,
            model_name="test_model",
            num_classes=3,  # Multiclass
        )

        labels = np.array([0, 1, 2, 0, 1, 2])
        # Create predictions with some misclassifications
        preds = np.array([
            [0.9, 0.05, 0.05],  # Correct (class 0)
            [0.1, 0.8, 0.1],   # Correct (class 1)
            [0.1, 0.1, 0.8],   # Correct (class 2)
            [0.1, 0.8, 0.1],   # Wrong (class 1 instead of 0)
            [0.8, 0.1, 0.1],   # Wrong (class 0 instead of 1)
            [0.1, 0.8, 0.1],   # Wrong (class 1 instead of 2)
        ])

        impact = evaluator.assess_clinical_impact(labels, preds)

        assert "risk_level" in impact
        assert "impact_counts" in impact
        assert impact["impact_counts"]["medium"] == 3  # 3 misclassifications

    def test_evaluate_defense_method(self, evaluator):
        """Test evaluating a defense method."""
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 10)
        adv_images = images + 0.1

        result = evaluator.evaluate_defense(
            clean_images=images,
            clean_labels=labels,
            adversarial_images=adv_images,
            defense_type=DefenseType.GAUSSIAN_BLUR,
            sigma=1.0,
        )

        assert isinstance(result, dict)
        assert "defense_type" in result
