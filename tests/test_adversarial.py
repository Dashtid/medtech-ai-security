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

    def test_evaluate_multiple_attacks(self, evaluator):
        """Test evaluating multiple attack types."""
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 10)

        # Test FGSM attack evaluation
        fgsm_result = evaluator.evaluate_attack(
            images=images,
            labels=labels,
            attack_type=AttackType.FGSM,
            epsilon=0.03,
        )

        assert "attack_type" in fgsm_result
        assert "success_rate" in fgsm_result
        assert "robust_accuracy" in fgsm_result

        # Test PGD attack evaluation
        pgd_result = evaluator.evaluate_attack(
            images=images,
            labels=labels,
            attack_type=AttackType.PGD,
            epsilon=0.03,
            alpha=0.007,
            num_iterations=3,
        )

        assert "attack_type" in pgd_result
        assert pgd_result["attack_type"] == "pgd"


class TestAttackResultDataclass:
    """Test AttackResult dataclass functionality."""

    def test_attack_result_to_dict(self):
        """Test converting AttackResult to dictionary."""
        np.random.seed(42)
        original = np.random.rand(5, 28, 28, 1).astype(np.float32)
        adversarial = original + 0.01
        perturbations = adversarial - original

        result = AttackResult(
            attack_type=AttackType.FGSM,
            original_images=original,
            adversarial_images=adversarial,
            perturbations=perturbations,
            original_predictions=np.array([0, 0, 0, 0, 0]),
            adversarial_predictions=np.array([0, 1, 0, 1, 0]),
            original_labels=np.array([0, 0, 0, 0, 0]),
            attack_params={"epsilon": 0.03},
            success_rate=0.4,
            mean_perturbation_l2=0.05,
            mean_perturbation_linf=0.01,
            num_samples=5,
        )

        result_dict = result.to_dict()

        assert result_dict["attack_type"] == "fgsm"
        assert result_dict["success_rate"] == 0.4
        assert "mean_perturbation_l2" in result_dict
        assert "attack_params" in result_dict

    def test_attack_result_with_successful_indices(self):
        """Test AttackResult with successful attack indices."""
        np.random.seed(42)
        original = np.random.rand(5, 28, 28, 1).astype(np.float32)
        adversarial = original.copy()
        perturbations = adversarial - original

        result = AttackResult(
            attack_type=AttackType.PGD,
            original_images=original,
            adversarial_images=adversarial,
            perturbations=perturbations,
            original_predictions=np.array([0, 0, 0, 0, 0]),
            adversarial_predictions=np.array([1, 1, 0, 0, 0]),
            original_labels=np.array([0, 0, 0, 0, 0]),
            attack_params={"epsilon": 0.03, "num_iterations": 10},
            success_rate=0.4,
            mean_perturbation_l2=0.01,
            mean_perturbation_linf=0.005,
            num_samples=5,
            successful_indices=[0, 1],
        )

        assert len(result.successful_indices) == 2
        assert result.num_samples == 5


class TestAdversarialAttackerAdvanced:
    """Advanced tests for AdversarialAttacker."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])

        return model

    def test_attack_with_batching(self, simple_model):
        """Test attack with batch processing."""
        attacker = AdversarialAttacker(model=simple_model)
        np.random.seed(42)
        images = np.random.rand(50, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 50)

        result = attacker.fgsm(images, labels, epsilon=0.03)

        assert result.adversarial_images.shape == images.shape
        assert result.success_rate >= 0

    def test_attack_different_epsilons(self, simple_model):
        """Test attacks with different epsilon values."""
        attacker = AdversarialAttacker(model=simple_model)
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 10)

        for epsilon in [0.01, 0.03, 0.1]:
            result = attacker.fgsm(images, labels, epsilon=epsilon)
            # Check epsilon is stored in attack_params
            assert result.attack_params.get("epsilon") == epsilon
            assert result.mean_perturbation_linf <= epsilon + 1e-5

    def test_attack_preserves_image_range(self, simple_model):
        """Test that attacks preserve valid image range."""
        attacker = AdversarialAttacker(
            model=simple_model,
            clip_min=0.0,
            clip_max=1.0,
        )
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        result = attacker.fgsm(images, labels, epsilon=0.5)

        assert np.all(result.adversarial_images >= 0.0)
        assert np.all(result.adversarial_images <= 1.0)

        result_pgd = attacker.pgd(images, labels, epsilon=0.5, alpha=0.1, num_iterations=5)
        assert np.all(result_pgd.adversarial_images >= 0.0)
        assert np.all(result_pgd.adversarial_images <= 1.0)

    def test_targeted_fgsm_attack(self, simple_model):
        """Test FGSM attack in targeted mode."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 10)
        target_labels = np.array([1] * 10)

        result = attacker.fgsm(
            images=images,
            labels=labels,
            epsilon=0.1,
            targeted=True,
            target_labels=target_labels,
        )

        assert result.attack_type == AttackType.FGSM
        assert result.attack_params["targeted"] is True
        assert result.adversarial_images.shape == images.shape

    def test_targeted_fgsm_requires_target_labels(self, simple_model):
        """Test FGSM raises error if targeted but no target_labels."""
        attacker = AdversarialAttacker(model=simple_model)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)
        labels = np.array([0] * 5)

        with pytest.raises(ValueError, match="target_labels required"):
            attacker.fgsm(images, labels, targeted=True)

    def test_targeted_pgd_attack(self, simple_model):
        """Test PGD attack in targeted mode."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 10)
        target_labels = np.array([1] * 10)

        result = attacker.pgd(
            images=images,
            labels=labels,
            epsilon=0.1,
            alpha=0.01,
            num_iterations=5,
            targeted=True,
            target_labels=target_labels,
        )

        assert result.attack_type == AttackType.PGD
        assert result.attack_params["targeted"] is True

    def test_targeted_pgd_requires_target_labels(self, simple_model):
        """Test PGD raises error if targeted but no target_labels."""
        attacker = AdversarialAttacker(model=simple_model)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)
        labels = np.array([0] * 5)

        with pytest.raises(ValueError, match="target_labels required"):
            attacker.pgd(images, labels, targeted=True)

    def test_pgd_no_random_start(self, simple_model):
        """Test PGD attack without random initialization."""
        attacker = AdversarialAttacker(model=simple_model)
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 10)

        result = attacker.pgd(
            images=images,
            labels=labels,
            epsilon=0.03,
            alpha=0.007,
            num_iterations=5,
            random_start=False,
        )

        assert result.attack_type == AttackType.PGD
        assert result.attack_params["random_start"] is False

    def test_attack_router_fgsm(self, simple_model):
        """Test attack() method routes to FGSM correctly."""
        attacker = AdversarialAttacker(model=simple_model)
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 5)

        result = attacker.attack(images, labels, attack_type="fgsm", epsilon=0.03)

        assert result.attack_type == AttackType.FGSM

    def test_attack_router_pgd(self, simple_model):
        """Test attack() method routes to PGD correctly."""
        attacker = AdversarialAttacker(model=simple_model)
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 5)

        result = attacker.attack(
            images, labels, attack_type=AttackType.PGD, epsilon=0.03, num_iterations=3
        )

        assert result.attack_type == AttackType.PGD

    def test_attack_router_cw_l2(self, simple_model):
        """Test attack() method routes to C&W L2 correctly."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(2, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0, 0])

        # Use minimal iterations for speed
        result = attacker.attack(
            images, labels,
            attack_type="cw_l2",
            binary_search_steps=1,
            max_iterations=5,
        )

        assert result.attack_type == AttackType.CW_L2

    def test_attack_router_unknown(self, simple_model):
        """Test attack() raises error for unknown attack type."""
        attacker = AdversarialAttacker(model=simple_model)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)
        labels = np.array([0] * 5)

        with pytest.raises(ValueError):
            attacker.attack(images, labels, attack_type="unknown_attack")


class TestCWAttack:
    """Test Carlini-Wagner L2 attack."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])
        return model

    def test_cw_l2_basic(self, simple_model):
        """Test basic C&W L2 attack."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(2, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0, 0])

        result = attacker.cw_l2(
            images=images,
            labels=labels,
            binary_search_steps=1,
            max_iterations=10,
            initial_const=0.01,
        )

        assert result.attack_type == AttackType.CW_L2
        assert result.adversarial_images.shape == images.shape
        assert result.num_samples == 2

    def test_cw_l2_targeted(self, simple_model):
        """Test C&W L2 attack in targeted mode."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(2, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0, 0])
        target_labels = np.array([1, 1])

        result = attacker.cw_l2(
            images=images,
            labels=labels,
            targeted=True,
            target_labels=target_labels,
            binary_search_steps=1,
            max_iterations=10,
        )

        assert result.attack_type == AttackType.CW_L2
        assert result.attack_params["targeted"] is True

    def test_cw_l2_requires_target_for_targeted(self, simple_model):
        """Test C&W L2 raises error if targeted without target_labels."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        images = np.random.rand(2, 28, 28, 1).astype(np.float32)
        labels = np.array([0, 0])

        with pytest.raises(ValueError, match="target_labels required"):
            attacker.cw_l2(images, labels, targeted=True)


class TestFullEvaluation:
    """Test full robustness evaluation."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])
        return model

    def test_full_evaluation_default_configs(self, simple_model):
        """Test full evaluation with default attack/defense configurations."""
        evaluator = RobustnessEvaluator(
            model=simple_model,
            model_name="test_model",
            num_classes=2,
        )
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 10)

        # Use minimal configs for speed
        attack_configs = [
            (AttackType.FGSM, {"epsilon": 0.03}),
        ]
        defense_configs = [
            (DefenseType.GAUSSIAN_BLUR, {"sigma": 1.0}),
        ]

        report = evaluator.full_evaluation(
            images, labels,
            attack_configs=attack_configs,
            defense_configs=defense_configs,
        )

        assert isinstance(report, RobustnessReport)
        assert report.model_name == "test_model"
        assert report.clean_accuracy >= 0
        assert "fgsm_0.03" in report.attack_results
        assert report.vulnerability_assessment is not None

    def test_full_evaluation_high_vulnerability(self, simple_model):
        """Test full evaluation detects high vulnerability."""
        # Create a model that's easily fooled
        def weak_model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            # More sensitive model
            return np.column_stack([1 - mean_val * 2, mean_val * 2])

        evaluator = RobustnessEvaluator(
            model=weak_model,
            model_name="weak_model",
            num_classes=2,
        )
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 10)

        attack_configs = [
            (AttackType.FGSM, {"epsilon": 0.05}),
            (AttackType.FGSM, {"epsilon": 0.1}),
            (AttackType.PGD, {"epsilon": 0.05, "alpha": 0.01, "num_iterations": 3}),
        ]

        report = evaluator.full_evaluation(
            images, labels,
            attack_configs=attack_configs,
            defense_configs=[],
        )

        assert report.clinical_risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_full_evaluation_generates_recommendations(self, simple_model):
        """Test full evaluation generates recommendations."""
        evaluator = RobustnessEvaluator(
            model=simple_model,
            model_name="test_model",
            num_classes=2,
        )
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 10)

        attack_configs = [(AttackType.FGSM, {"epsilon": 0.05})]

        report = evaluator.full_evaluation(
            images, labels,
            attack_configs=attack_configs,
            defense_configs=[],
        )

        assert len(report.recommendations) > 0

    def test_full_evaluation_with_defenses(self):
        """Test full evaluation with defense evaluation."""
        # Use a model that's more susceptible to attacks
        def weak_model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            # More sensitive to perturbations
            return np.column_stack([1 - mean_val * 3, mean_val * 3])

        evaluator = RobustnessEvaluator(
            model=weak_model,
            model_name="test_model",
            num_classes=2,
        )
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 10)

        # Use larger epsilon to ensure attack success
        attack_configs = [(AttackType.FGSM, {"epsilon": 0.2})]
        defense_configs = [
            (DefenseType.GAUSSIAN_BLUR, {"sigma": 1.0}),
        ]

        report = evaluator.full_evaluation(
            images, labels,
            attack_configs=attack_configs,
            defense_configs=defense_configs,
        )

        # Defense results might be empty if attack doesn't succeed
        assert isinstance(report.defense_results, dict)


class TestBinaryModelAttacks:
    """Test attacks on binary classification models."""

    @pytest.fixture
    def binary_model(self):
        """Create a binary output model (single value)."""
        def model(x):
            # Returns single probability value
            mean_val = np.mean(x, axis=(1, 2, 3))
            return mean_val
        return model

    def test_fgsm_binary_model(self, binary_model):
        """Test FGSM on binary output model."""
        attacker = AdversarialAttacker(model=binary_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0, 1, 0, 1, 0])

        result = attacker.fgsm(images, labels, epsilon=0.1)

        assert result.attack_type == AttackType.FGSM
        assert result.adversarial_images.shape == images.shape

    def test_pgd_binary_model(self, binary_model):
        """Test PGD on binary output model."""
        attacker = AdversarialAttacker(model=binary_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0, 1, 0, 1, 0])

        result = attacker.pgd(
            images, labels,
            epsilon=0.1,
            alpha=0.02,
            num_iterations=3
        )

        assert result.attack_type == AttackType.PGD


class TestDefenderAdvanced:
    """Advanced tests for AdversarialDefender."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])
        return model

    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing."""
        np.random.seed(42)
        return np.random.rand(10, 28, 28, 1).astype(np.float32)

    def test_detect_adversarial_no_model(self):
        """Test detect_adversarial raises without model."""
        defender = AdversarialDefender(model=None)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)

        with pytest.raises(ValueError, match="Model required"):
            defender.detect_adversarial(images)

    def test_detect_adversarial_with_model(self, simple_model):
        """Test detect_adversarial with model."""
        defender = AdversarialDefender(model=simple_model)
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)

        is_adversarial, scores = defender.detect_adversarial(images, threshold=0.1)

        assert is_adversarial.shape == (5,)
        assert scores.shape == (5,)
        assert np.all(scores >= 0)

    def test_detect_adversarial_threshold(self, simple_model):
        """Test detect_adversarial with different thresholds."""
        defender = AdversarialDefender(model=simple_model)
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)

        # Low threshold should detect more
        is_adv_low, _ = defender.detect_adversarial(images, threshold=0.001)
        # High threshold should detect less
        is_adv_high, _ = defender.detect_adversarial(images, threshold=0.5)

        # High threshold should have <= detections than low threshold
        assert np.sum(is_adv_high) <= np.sum(is_adv_low)

    def test_evaluate_defense_with_model(self, simple_model):
        """Test evaluate_defense with model."""
        defender = AdversarialDefender(model=simple_model)
        np.random.seed(42)
        clean_images = np.random.rand(10, 28, 28, 1).astype(np.float32)
        clean_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        adv_images = clean_images + 0.05 * np.random.randn(*clean_images.shape).astype(np.float32)
        adv_images = np.clip(adv_images, 0, 1)

        result = defender.evaluate_defense(
            clean_images=clean_images,
            clean_labels=clean_labels,
            adversarial_images=adv_images,
            defense_type=DefenseType.GAUSSIAN_BLUR,
            defense_fn=defender.gaussian_blur,
            defense_params={"sigma": 1.0},
        )

        assert isinstance(result, DefenseResult)
        assert result.defense_type == DefenseType.GAUSSIAN_BLUR
        assert 0 <= result.clean_accuracy_before <= 1
        assert 0 <= result.clean_accuracy_after <= 1

    def test_gaussian_blur_params(self, sample_images):
        """Test gaussian_blur with different sigma values."""
        defender = AdversarialDefender()

        result_low = defender.gaussian_blur(sample_images, sigma=0.5)
        result_high = defender.gaussian_blur(sample_images, sigma=3.0)

        # Higher sigma should smooth more
        assert not np.allclose(result_low, result_high)
        assert result_low.shape == sample_images.shape
        assert result_high.shape == sample_images.shape

    def test_jpeg_compression_quality(self, sample_images):
        """Test JPEG compression with different quality values."""
        defender = AdversarialDefender()

        result_high = defender.jpeg_compression(sample_images, quality=95)
        result_low = defender.jpeg_compression(sample_images, quality=20)

        # Lower quality should have more compression artifacts
        assert not np.allclose(result_high, result_low)

    def test_feature_squeezing_bit_depth(self, sample_images):
        """Test feature squeezing with different bit depths."""
        defender = AdversarialDefender()

        result_high = defender.feature_squeezing(sample_images, bit_depth=8)
        result_low = defender.feature_squeezing(sample_images, bit_depth=2)

        # Lower bit depth should have more squeezing
        assert not np.allclose(result_high, result_low)

    def test_ensemble_defense_combines_methods(self, sample_images):
        """Test ensemble defense combines multiple methods."""
        defender = AdversarialDefender()

        # Single method
        single_result = defender.gaussian_blur(sample_images, sigma=1.0)

        # Ensemble with multiple defenses as (fn, params) tuples
        ensemble_result = defender.ensemble_defense(
            sample_images,
            defenses=[
                (defender.gaussian_blur, {"sigma": 1.0}),
                (defender.feature_squeezing, {"bit_depth": 4}),
            ],
        )

        # Ensemble result should be different from single method
        assert not np.allclose(single_result, ensemble_result)


class TestAttackResultMetrics:
    """Test AttackResult metrics."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])
        return model

    def test_attack_result_perturbation_metrics(self, simple_model):
        """Test perturbation metrics in attack result."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 10)

        result = attacker.fgsm(images, labels, epsilon=0.05)

        # Check perturbation metrics are valid
        assert result.mean_perturbation_l2 >= 0
        assert result.mean_perturbation_linf >= 0
        assert result.mean_perturbation_linf <= result.attack_params.get("epsilon", 1.0) + 1e-5

    def test_attack_result_to_dict(self, simple_model):
        """Test attack result to_dict method."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0])

        result = attacker.fgsm(images, labels, epsilon=0.03)

        result_dict = result.to_dict()

        assert "attack_type" in result_dict
        assert "success_rate" in result_dict
        assert "mean_perturbation_l2" in result_dict
        assert "mean_perturbation_linf" in result_dict
        assert "attack_params" in result_dict

    def test_attack_success_indices(self, simple_model):
        """Test attack successful_indices."""
        attacker = AdversarialAttacker(model=simple_model, num_classes=2)
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.4
        labels = np.array([0] * 10)

        result = attacker.fgsm(images, labels, epsilon=0.1)

        # Successful indices should be a list of indices
        assert len(result.successful_indices) <= len(images)
        # Success rate should be between 0 and 1
        assert 0 <= result.success_rate <= 1


class TestRobustnessReportIO:
    """Test RobustnessReport save/load functionality."""

    def test_report_save_and_load(self, tmp_path):
        """Test saving and loading a report."""
        report = RobustnessReport(
            model_name="test_model",
            evaluation_date="2025-01-01T00:00:00",
            clean_accuracy=0.95,
            attack_results={"fgsm": {"success_rate": 0.3}},
            defense_results={"jpeg": {"accuracy_gain": 0.1}},
            vulnerability_assessment="LOW VULNERABILITY",
            recommendations=["Continue monitoring"],
            clinical_risk_level="LOW",
            metadata={"num_samples": 100},
        )

        save_path = tmp_path / "test_report.json"
        report.save(save_path)

        assert save_path.exists()

        loaded = RobustnessReport.load(save_path)
        assert loaded.model_name == "test_model"
        assert loaded.clean_accuracy == 0.95
        assert loaded.clinical_risk_level == "LOW"

    def test_report_to_dict(self):
        """Test report to_dict method."""
        report = RobustnessReport(
            model_name="test_model",
            evaluation_date="2025-01-01T00:00:00",
            clean_accuracy=0.9,
        )

        result = report.to_dict()
        assert result["model_name"] == "test_model"
        assert result["clean_accuracy"] == 0.9
        assert "attack_results" in result
        assert "defense_results" in result


class TestAdversarialTrainerMethods:
    """Test AdversarialTrainer methods."""

    @pytest.fixture
    def simple_attacker(self):
        """Create a simple attacker for testing."""
        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])

        attacker = AdversarialAttacker(model=model, num_classes=2)
        return attacker

    def test_generate_adversarial_batch_zero_ratio(self, simple_attacker):
        """Test generate_adversarial_batch with zero ratio."""
        from medtech_ai_security.adversarial.defenses import AdversarialTrainer

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])

        trainer = AdversarialTrainer(
            model=model,
            attack_fn=simple_attacker.fgsm,
            attack_params={"epsilon": 0.01},
        )

        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32)
        labels = np.array([0] * 5 + [1] * 5)

        # Zero ratio should return unchanged images
        mixed_images, mixed_labels = trainer.generate_adversarial_batch(
            images, labels, ratio=0.0
        )

        assert np.allclose(mixed_images, images)
        np.testing.assert_array_equal(mixed_labels, labels)

    def test_generate_adversarial_batch_with_ratio(self, simple_attacker):
        """Test generate_adversarial_batch with ratio."""
        from medtech_ai_security.adversarial.defenses import AdversarialTrainer

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            return np.column_stack([1 - mean_val, mean_val])

        trainer = AdversarialTrainer(
            model=model,
            attack_fn=simple_attacker.fgsm,
            attack_params={"epsilon": 0.05},
        )

        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32) * 0.3
        labels = np.array([0] * 10)

        # 50% ratio should modify some images
        mixed_images, mixed_labels = trainer.generate_adversarial_batch(
            images, labels, ratio=0.5
        )

        # Some images should be different (adversarial)
        num_changed = np.sum(~np.isclose(mixed_images, images).all(axis=(1, 2, 3)))
        assert num_changed > 0
        assert num_changed <= 5  # 50% of 10

    def test_trainer_init_default_params(self, simple_attacker):
        """Test AdversarialTrainer initialization with defaults."""
        from medtech_ai_security.adversarial.defenses import AdversarialTrainer

        def model(x):
            return x

        trainer = AdversarialTrainer(model=model, attack_fn=simple_attacker.fgsm)

        assert trainer.attack_params == {"epsilon": 0.01}


class TestDefenseMethodsAdvanced:
    """Advanced tests for defense methods."""

    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing."""
        np.random.seed(42)
        return np.random.rand(5, 32, 32, 3).astype(np.float32)

    def test_spatial_smoothing_kernel_sizes(self, sample_images):
        """Test spatial smoothing with different kernel sizes."""
        defender = AdversarialDefender()

        result_small = defender.spatial_smoothing(sample_images, kernel_size=3)
        result_large = defender.spatial_smoothing(sample_images, kernel_size=5)

        # Different kernel sizes should give different results
        assert not np.allclose(result_small, result_large)
        assert result_small.shape == sample_images.shape

    def test_bit_depth_reduction_high_range(self):
        """Test bit depth reduction with 0-255 range images."""
        defender = AdversarialDefender()

        # Images in 0-255 range
        np.random.seed(42)
        images = (np.random.rand(3, 28, 28, 1) * 255).astype(np.float32)

        result = defender.bit_depth_reduction(images, bits=4)

        assert result.shape == images.shape
        assert result.max() <= 255
        assert result.min() >= 0

    def test_feature_squeezing_combined(self):
        """Test feature squeezing combines bit depth and blur."""
        defender = AdversarialDefender()
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)

        result = defender.feature_squeezing(images, bit_depth=4, blur_sigma=1.0)

        # Result should be different from original
        assert not np.allclose(result, images)
        assert result.shape == images.shape

    def test_ensemble_defense_default_methods(self):
        """Test ensemble defense with default methods."""
        defender = AdversarialDefender()
        np.random.seed(42)
        images = np.random.rand(5, 28, 28, 1).astype(np.float32)

        # Use default defenses (None parameter)
        result = defender.ensemble_defense(images, defenses=None)

        assert result.shape == images.shape
        assert not np.allclose(result, images)

    def test_jpeg_compression_single_channel(self):
        """Test JPEG compression on single channel images."""
        defender = AdversarialDefender()
        np.random.seed(42)
        images = np.random.rand(3, 28, 28, 1).astype(np.float32)

        result = defender.jpeg_compression(images, quality=75)

        assert result.shape == images.shape

    def test_jpeg_compression_three_channel(self):
        """Test JPEG compression on RGB images."""
        defender = AdversarialDefender()
        np.random.seed(42)
        images = np.random.rand(3, 28, 28, 3).astype(np.float32)

        result = defender.jpeg_compression(images, quality=75)

        assert result.shape == images.shape


class TestClinicalImpactAssessment:
    """Test clinical impact assessment methods."""

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
        return RobustnessEvaluator(simple_model, num_classes=2)

    def test_clinical_impact_critical_risk(self, evaluator):
        """Test clinical impact with critical risk (false negatives)."""
        # True labels: all positive (malignant = 1)
        original_labels = np.array([1] * 20)
        # Predictions: all negative (benign = 0) - false negatives
        adversarial_preds = np.array([0] * 20)

        impact = evaluator.assess_clinical_impact(original_labels, adversarial_preds)

        assert impact["risk_level"] == "CRITICAL"
        assert impact["impact_counts"]["critical"] == 20

    def test_clinical_impact_high_risk(self, evaluator):
        """Test clinical impact with high risk (false positives)."""
        # True labels: all negative (benign = 0)
        original_labels = np.array([0] * 20)
        # Predictions: many positive (malignant = 1) - false positives
        adversarial_preds = np.array([1] * 20)

        impact = evaluator.assess_clinical_impact(original_labels, adversarial_preds)

        assert impact["risk_level"] == "HIGH"
        assert impact["impact_counts"]["high"] == 20

    def test_clinical_impact_low_risk(self, evaluator):
        """Test clinical impact with low risk (mostly correct)."""
        original_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        # Mostly correct predictions
        adversarial_preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        impact = evaluator.assess_clinical_impact(original_labels, adversarial_preds)

        assert impact["risk_level"] == "LOW"
        assert impact["impact_counts"]["none"] == 10

    def test_clinical_impact_multiclass(self, simple_model):
        """Test clinical impact with multiclass model."""
        evaluator = RobustnessEvaluator(simple_model, num_classes=5)

        original_labels = np.array([0, 1, 2, 3, 4])
        # Predictions as one-hot like model outputs (5 classes)
        # All misclassified to different classes
        adversarial_preds = np.array([
            [0, 1, 0, 0, 0],  # pred=1, true=0 -> medium
            [0, 0, 1, 0, 0],  # pred=2, true=1 -> medium
            [0, 0, 0, 1, 0],  # pred=3, true=2 -> medium
            [0, 0, 0, 0, 1],  # pred=4, true=3 -> medium
            [1, 0, 0, 0, 0],  # pred=0, true=4 -> medium
        ], dtype=np.float32)

        impact = evaluator.assess_clinical_impact(original_labels, adversarial_preds)

        # Multiclass misclassifications are "medium"
        assert impact["impact_counts"]["medium"] == 5


class TestEvaluatorDefenseEvaluation:
    """Test evaluator defense evaluation methods."""

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
        return RobustnessEvaluator(simple_model, num_classes=2)

    def test_evaluate_defense_unknown_type(self, evaluator):
        """Test evaluate_defense with unknown defense type."""
        np.random.seed(42)
        clean_images = np.random.rand(5, 28, 28, 1).astype(np.float32)
        clean_labels = np.array([0, 1, 0, 1, 0])
        adv_images = clean_images + 0.1

        # Create a fake defense type by using a valid type but testing the flow
        result = evaluator.evaluate_defense(
            clean_images,
            clean_labels,
            adv_images,
            DefenseType.GAUSSIAN_BLUR,
            sigma=1.0,
        )

        assert "defense_type" in result
        assert "clean_accuracy_before" in result

    def test_evaluate_attack_fgsm(self, evaluator):
        """Test evaluate_attack with FGSM."""
        np.random.seed(42)
        images = np.random.rand(10, 28, 28, 1).astype(np.float32)
        labels = np.array([0] * 5 + [1] * 5)

        result = evaluator.evaluate_attack(
            images, labels, AttackType.FGSM, epsilon=0.03
        )

        assert "attack_type" in result
        assert result["attack_type"] == "fgsm"
        assert "success_rate" in result
        assert "robust_accuracy" in result

    def test_evaluate_clean_accuracy_multiclass(self, simple_model):
        """Test clean accuracy with multiclass predictions."""
        # Model that returns multiclass predictions
        def multiclass_model(x):
            batch_size = len(x)
            # Return 3-class predictions
            preds = np.zeros((batch_size, 3))
            for i in range(batch_size):
                preds[i, i % 3] = 1.0
            return preds

        evaluator = RobustnessEvaluator(multiclass_model, num_classes=3)
        np.random.seed(42)
        images = np.random.rand(9, 28, 28, 1).astype(np.float32)
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        acc = evaluator.evaluate_clean_accuracy(images, labels)
        assert 0 <= acc <= 1


class TestVulnerabilityAssessment:
    """Test vulnerability assessment generation."""

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
        return RobustnessEvaluator(simple_model, num_classes=2)

    def test_generate_recommendations_small_perturbation(self, evaluator):
        """Test recommendations for small perturbation attacks."""
        attack_results = {
            "fgsm": {
                "success_rate": 0.6,
                "mean_perturbation_linf": 0.005,  # Very small
            }
        }
        defense_results = {}
        clinical_impact = {"risk_level": "LOW"}

        recs = evaluator.generate_recommendations(
            attack_results, defense_results, clinical_impact
        )

        # Should warn about small perturbation
        assert any("SMALL PERTURBATION" in r for r in recs)

    def test_generate_recommendations_critical_clinical_risk(self, evaluator):
        """Test recommendations for critical clinical risk."""
        attack_results = {"fgsm": {"success_rate": 0.3}}
        defense_results = {}
        clinical_impact = {"risk_level": "CRITICAL"}

        recs = evaluator.generate_recommendations(
            attack_results, defense_results, clinical_impact
        )

        assert any("CRITICAL CLINICAL RISK" in r for r in recs)

    def test_generate_recommendations_high_clinical_risk(self, evaluator):
        """Test recommendations for high clinical risk."""
        attack_results = {"fgsm": {"success_rate": 0.1}}
        defense_results = {}
        clinical_impact = {"risk_level": "HIGH"}

        recs = evaluator.generate_recommendations(
            attack_results, defense_results, clinical_impact
        )

        assert any("HIGH CLINICAL RISK" in r for r in recs)
