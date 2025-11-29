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
    DefenseType,
    DefenseResult,
)
from medtech_ai_security.adversarial.evaluator import (
    RobustnessReport,
    RobustnessEvaluator,
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
