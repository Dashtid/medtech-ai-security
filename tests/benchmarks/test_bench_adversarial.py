"""Benchmark tests for adversarial ML module.

Run with:
    pytest tests/benchmarks/test_bench_adversarial.py --benchmark-only
    pytest tests/benchmarks/test_bench_adversarial.py --benchmark-compare
"""

import numpy as np
import pytest

from medtech_ai_security.adversarial import (
    AdversarialAttacker,
    AdversarialDefender,
)


@pytest.fixture
def sample_images():
    """Generate sample image batch for benchmarking."""
    np.random.seed(42)
    # 32 images, 28x28, single channel (MNIST-like)
    return np.random.rand(32, 28, 28, 1).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample labels for benchmarking."""
    np.random.seed(42)
    return np.random.randint(0, 10, size=32)


@pytest.fixture
def simple_model():
    """Create a simple callable model for benchmarking."""

    def model(x):
        # Simple thresholding model - returns 10-class probabilities
        if len(x.shape) == 4:
            mean_val = np.mean(x, axis=(1, 2, 3))
        else:
            mean_val = np.mean(x, axis=tuple(range(1, len(x.shape))))
        # Create 10-class probabilities
        probs = np.zeros((len(mean_val), 10))
        for i, m in enumerate(mean_val):
            predicted_class = int(m * 10) % 10
            probs[i, predicted_class] = 0.9
            probs[i, (predicted_class + 1) % 10] = 0.1
        return probs

    return model


@pytest.fixture
def binary_model():
    """Create a simple binary classifier model."""

    def model(x):
        if len(x.shape) == 4:
            mean_val = np.mean(x, axis=(1, 2, 3))
        else:
            mean_val = np.mean(x, axis=tuple(range(1, len(x.shape))))
        probs = np.column_stack([1 - mean_val, mean_val])
        return probs

    return model


@pytest.fixture
def attacker(simple_model):
    """Create an adversarial attacker instance."""
    return AdversarialAttacker(model=simple_model, num_classes=10)


@pytest.fixture
def binary_attacker(binary_model):
    """Create a binary adversarial attacker instance."""
    return AdversarialAttacker(model=binary_model, num_classes=2)


@pytest.fixture
def defender():
    """Create an adversarial defender instance."""
    return AdversarialDefender()


class TestAttackBenchmarks:
    """Benchmark suite for adversarial attack methods."""

    def test_bench_fgsm_attack(self, benchmark, attacker, sample_images, sample_labels):
        """Benchmark FGSM attack generation."""

        def run_fgsm():
            return attacker.fgsm(sample_images, sample_labels, epsilon=0.1)

        result = benchmark(run_fgsm)
        assert result.adversarial_images is not None

    def test_bench_pgd_attack(self, benchmark, attacker, sample_images, sample_labels):
        """Benchmark PGD attack generation."""

        def run_pgd():
            return attacker.pgd(sample_images, sample_labels, epsilon=0.1, num_iterations=10)

        result = benchmark(run_pgd)
        assert result.adversarial_images is not None

    def test_bench_pgd_attack_many_steps(self, benchmark, attacker, sample_images, sample_labels):
        """Benchmark PGD attack with more iterations."""

        def run_pgd_many():
            return attacker.pgd(sample_images, sample_labels, epsilon=0.1, num_iterations=40)

        result = benchmark(run_pgd_many)
        assert result.adversarial_images is not None

    def test_bench_cw_attack(self, benchmark, binary_attacker, sample_images):
        """Benchmark Carlini-Wagner attack."""
        # Use smaller batch for CW as it's slower
        images = sample_images[:8]
        labels = np.random.randint(0, 2, size=8)

        def run_cw():
            return binary_attacker.cw_l2(images, labels, max_iterations=50)

        result = benchmark(run_cw)
        assert result.adversarial_images is not None

    def test_bench_deepfool_attack(self, benchmark, binary_attacker, sample_images):
        """Benchmark DeepFool attack."""
        images = sample_images[:8]
        labels = np.random.randint(0, 2, size=8)

        def run_deepfool():
            return binary_attacker.deepfool(images, labels, max_iterations=20)

        result = benchmark(run_deepfool)
        assert result.adversarial_images is not None


class TestDefenseBenchmarks:
    """Benchmark suite for adversarial defense methods."""

    @pytest.fixture
    def adversarial_images(self, sample_images):
        """Generate pre-computed adversarial images."""
        noise = np.random.randn(*sample_images.shape).astype(np.float32) * 0.1
        return np.clip(sample_images + noise, 0, 1)

    def test_bench_jpeg_defense(self, benchmark, defender, adversarial_images):
        """Benchmark JPEG compression defense."""

        def run_jpeg():
            return defender.jpeg_compression(adversarial_images, quality=75)

        result = benchmark(run_jpeg)
        assert result.shape == adversarial_images.shape

    def test_bench_gaussian_blur_defense(self, benchmark, defender, adversarial_images):
        """Benchmark Gaussian blur defense."""

        def run_blur():
            return defender.gaussian_blur(adversarial_images, sigma=1.0)

        result = benchmark(run_blur)
        assert result.shape == adversarial_images.shape

    def test_bench_bit_depth_reduction(self, benchmark, defender, adversarial_images):
        """Benchmark bit depth reduction defense."""

        def run_bit_depth():
            return defender.bit_depth_reduction(adversarial_images, bits=4)

        result = benchmark(run_bit_depth)
        assert result.shape == adversarial_images.shape

    def test_bench_feature_squeezing(self, benchmark, defender, adversarial_images):
        """Benchmark feature squeezing defense."""

        def run_squeeze():
            return defender.feature_squeezing(adversarial_images, bit_depth=4)

        result = benchmark(run_squeeze)
        assert result.shape == adversarial_images.shape

    def test_bench_combined_defense(self, benchmark, defender, adversarial_images):
        """Benchmark combined defense pipeline."""

        def run_ensemble():
            return defender.ensemble_defense(adversarial_images)

        result = benchmark(run_ensemble)
        assert result.shape == adversarial_images.shape


class TestEvaluationBenchmarks:
    """Benchmark suite for robustness evaluation."""

    def test_bench_full_evaluation_small(self, benchmark, attacker, sample_images, sample_labels):
        """Benchmark full robustness evaluation on small dataset."""
        images = sample_images[:16]
        labels = sample_labels[:16]

        def run_eval():
            return attacker.fgsm(images, labels, epsilon=0.1)

        result = benchmark(run_eval)
        assert result.success_rate >= 0.0

    def test_bench_full_evaluation_multiple_attacks(
        self, benchmark, attacker, sample_images, sample_labels
    ):
        """Benchmark evaluation with multiple attacks."""
        images = sample_images[:16]
        labels = sample_labels[:16]

        def run_multiple():
            fgsm_result = attacker.fgsm(images, labels, epsilon=0.1)
            pgd_result = attacker.pgd(images, labels, epsilon=0.1, num_iterations=5)
            return fgsm_result, pgd_result

        result = benchmark(run_multiple)
        assert len(result) == 2


class TestMedicalImageBenchmarks:
    """Benchmark suite for medical imaging specific operations."""

    @pytest.fixture
    def medical_images(self):
        """Generate synthetic medical images (CT scan-like)."""
        np.random.seed(42)
        # Simulate CT scan slices: 16 images, 64x64, single channel
        images = np.random.rand(16, 64, 64, 1).astype(np.float32)
        # Add realistic texture patterns
        for i in range(len(images)):
            images[i] += 0.1 * np.sin(np.linspace(0, 10, 64)).reshape(1, 64, 1)
        return np.clip(images, 0, 1)

    @pytest.fixture
    def medical_model(self):
        """Create a simple model for medical images (binary classification)."""

        def model(x):
            mean_val = np.mean(x, axis=(1, 2, 3))
            probs = np.column_stack([1 - mean_val, mean_val])
            return probs

        return model

    def test_bench_medical_fgsm(self, benchmark, medical_images, medical_model):
        """Benchmark FGSM on medical images."""
        labels = np.random.randint(0, 2, size=16)
        attacker = AdversarialAttacker(model=medical_model, num_classes=2)

        def run_medical_fgsm():
            return attacker.fgsm(medical_images, labels, epsilon=0.05)

        result = benchmark(run_medical_fgsm)
        assert result.adversarial_images is not None

    def test_bench_perturbation_visibility(self, benchmark, medical_images, medical_model):
        """Benchmark perturbation visibility metrics computation."""
        labels = np.random.randint(0, 2, size=16)
        attacker = AdversarialAttacker(model=medical_model, num_classes=2)

        def compute_visibility():
            result = attacker.fgsm(medical_images, labels, epsilon=0.05)
            perturbation = result.adversarial_images - medical_images
            l2_norm = np.linalg.norm(perturbation.reshape(len(medical_images), -1), axis=1)
            linf_norm = np.max(np.abs(perturbation.reshape(len(medical_images), -1)), axis=1)
            return l2_norm, linf_norm

        l2, linf = benchmark(compute_visibility)
        assert len(l2) == 16
        assert len(linf) == 16
