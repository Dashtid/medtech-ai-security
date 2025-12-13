"""Benchmark tests for adversarial ML module.

Run with:
    pytest tests/benchmarks/test_bench_adversarial.py --benchmark-only
    pytest tests/benchmarks/test_bench_adversarial.py --benchmark-compare
"""

import numpy as np
import pytest


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
def large_images():
    """Generate larger image batch for stress testing."""
    np.random.seed(42)
    # 100 images, 224x224, RGB (ImageNet-like)
    return np.random.rand(100, 224, 224, 3).astype(np.float32)


@pytest.fixture
def large_labels():
    """Generate labels for large image batch."""
    np.random.seed(42)
    return np.random.randint(0, 1000, size=100)


class TestAttackBenchmarks:
    """Benchmark suite for adversarial attack methods."""

    @pytest.fixture
    def classifier_model(self):
        """Load or create a classifier model for benchmarking."""
        from medtech_ai_security.adversarial.evaluator import create_simple_classifier

        model = create_simple_classifier(input_shape=(28, 28, 1), num_classes=10)
        return model

    def test_bench_fgsm_attack(self, benchmark, classifier_model, sample_images, sample_labels):
        """Benchmark FGSM attack generation."""
        from medtech_ai_security.adversarial.evaluator import fgsm_attack

        result = benchmark(
            fgsm_attack,
            classifier_model,
            sample_images,
            sample_labels,
            epsilon=0.03
        )
        assert result.shape == sample_images.shape

    def test_bench_pgd_attack(self, benchmark, classifier_model, sample_images, sample_labels):
        """Benchmark PGD attack generation."""
        from medtech_ai_security.adversarial.evaluator import pgd_attack

        result = benchmark(
            pgd_attack,
            classifier_model,
            sample_images,
            sample_labels,
            epsilon=0.1,
            steps=10,
            step_size=0.01
        )
        assert result.shape == sample_images.shape

    def test_bench_pgd_attack_many_steps(self, benchmark, classifier_model, sample_images, sample_labels):
        """Benchmark PGD attack with more iterations."""
        from medtech_ai_security.adversarial.evaluator import pgd_attack

        result = benchmark(
            pgd_attack,
            classifier_model,
            sample_images,
            sample_labels,
            epsilon=0.1,
            steps=40,
            step_size=0.005
        )
        assert result.shape == sample_images.shape

    def test_bench_cw_attack(self, benchmark, classifier_model, sample_images, sample_labels):
        """Benchmark Carlini-Wagner attack."""
        from medtech_ai_security.adversarial.evaluator import cw_attack

        # C&W is expensive, use smaller batch
        small_images = sample_images[:8]
        small_labels = sample_labels[:8]

        result = benchmark(
            cw_attack,
            classifier_model,
            small_images,
            small_labels,
            confidence=0,
            iterations=100
        )
        assert result.shape == small_images.shape

    def test_bench_deepfool_attack(self, benchmark, classifier_model, sample_images, sample_labels):
        """Benchmark DeepFool attack."""
        from medtech_ai_security.adversarial.evaluator import deepfool_attack

        # DeepFool is expensive, use smaller batch
        small_images = sample_images[:8]

        result = benchmark(
            deepfool_attack,
            classifier_model,
            small_images,
            max_iterations=50
        )
        assert result.shape == small_images.shape


class TestDefenseBenchmarks:
    """Benchmark suite for adversarial defense methods."""

    @pytest.fixture
    def adversarial_images(self, sample_images):
        """Generate pre-computed adversarial images."""
        # Add small perturbation
        noise = np.random.randn(*sample_images.shape).astype(np.float32) * 0.1
        return np.clip(sample_images + noise, 0, 1)

    def test_bench_jpeg_defense(self, benchmark, adversarial_images):
        """Benchmark JPEG compression defense."""
        from medtech_ai_security.adversarial.evaluator import jpeg_defense

        result = benchmark(jpeg_defense, adversarial_images, quality=75)
        assert result.shape == adversarial_images.shape

    def test_bench_gaussian_blur_defense(self, benchmark, adversarial_images):
        """Benchmark Gaussian blur defense."""
        from medtech_ai_security.adversarial.evaluator import gaussian_blur_defense

        result = benchmark(gaussian_blur_defense, adversarial_images, sigma=1.0)
        assert result.shape == adversarial_images.shape

    def test_bench_bit_depth_reduction(self, benchmark, adversarial_images):
        """Benchmark bit depth reduction defense."""
        from medtech_ai_security.adversarial.evaluator import bit_depth_defense

        result = benchmark(bit_depth_defense, adversarial_images, bits=4)
        assert result.shape == adversarial_images.shape

    def test_bench_feature_squeezing(self, benchmark, adversarial_images):
        """Benchmark feature squeezing defense."""
        from medtech_ai_security.adversarial.evaluator import feature_squeezing_defense

        result = benchmark(feature_squeezing_defense, adversarial_images, bit_depth=4)
        assert result.shape == adversarial_images.shape

    def test_bench_combined_defense(self, benchmark, adversarial_images):
        """Benchmark combined defense pipeline."""
        from medtech_ai_security.adversarial.evaluator import combined_defense

        result = benchmark(
            combined_defense,
            adversarial_images,
            jpeg_quality=75,
            blur_sigma=0.5
        )
        assert result.shape == adversarial_images.shape


class TestEvaluationBenchmarks:
    """Benchmark suite for robustness evaluation."""

    @pytest.fixture
    def classifier_model(self):
        """Load or create a classifier model for benchmarking."""
        from medtech_ai_security.adversarial.evaluator import create_simple_classifier

        model = create_simple_classifier(input_shape=(28, 28, 1), num_classes=10)
        return model

    def test_bench_full_evaluation_small(self, benchmark, classifier_model, sample_images, sample_labels):
        """Benchmark full robustness evaluation on small dataset."""
        from medtech_ai_security.adversarial.evaluator import RobustnessEvaluator

        evaluator = RobustnessEvaluator(classifier_model)

        result = benchmark(
            evaluator.evaluate,
            sample_images[:16],
            sample_labels[:16],
            attacks=["fgsm"]
        )
        assert "clean_accuracy" in result

    def test_bench_full_evaluation_multiple_attacks(self, benchmark, classifier_model, sample_images, sample_labels):
        """Benchmark evaluation with multiple attacks."""
        from medtech_ai_security.adversarial.evaluator import RobustnessEvaluator

        evaluator = RobustnessEvaluator(classifier_model)

        result = benchmark(
            evaluator.evaluate,
            sample_images[:16],
            sample_labels[:16],
            attacks=["fgsm", "pgd"]
        )
        assert "fgsm" in result["attacks"]
        assert "pgd" in result["attacks"]

    def test_bench_gradient_computation(self, benchmark, classifier_model, sample_images, sample_labels):
        """Benchmark gradient computation for attacks."""
        from medtech_ai_security.adversarial.evaluator import compute_gradients

        result = benchmark(
            compute_gradients,
            classifier_model,
            sample_images,
            sample_labels
        )
        assert result.shape == sample_images.shape


class TestMedicalImageBenchmarks:
    """Benchmark suite for medical imaging specific operations."""

    @pytest.fixture
    def medical_images(self):
        """Generate synthetic medical images (CT scan-like)."""
        np.random.seed(42)
        # Simulate CT scan slices: 16 images, 512x512, single channel
        images = np.random.rand(16, 512, 512, 1).astype(np.float32)
        # Add realistic texture patterns
        for i in range(len(images)):
            images[i] += 0.1 * np.sin(np.linspace(0, 10, 512)).reshape(1, 512, 1)
        return np.clip(images, 0, 1)

    def test_bench_medical_fgsm(self, benchmark, medical_images):
        """Benchmark FGSM on medical images."""
        from medtech_ai_security.adversarial.evaluator import (
            create_medical_classifier,
            fgsm_attack
        )

        model = create_medical_classifier(input_shape=(512, 512, 1), num_classes=2)
        labels = np.array([0, 1] * 8)

        result = benchmark(
            fgsm_attack,
            model,
            medical_images[:8],
            labels[:8],
            epsilon=0.01  # Smaller epsilon for medical images
        )
        assert result.shape == medical_images[:8].shape

    def test_bench_perturbation_visibility(self, benchmark, medical_images):
        """Benchmark perturbation visibility metrics."""
        from medtech_ai_security.adversarial.evaluator import compute_visibility_metrics

        # Create adversarial version
        adv_images = medical_images + 0.05 * np.random.randn(*medical_images.shape).astype(np.float32)
        adv_images = np.clip(adv_images, 0, 1)

        result = benchmark(
            compute_visibility_metrics,
            medical_images,
            adv_images
        )
        assert "psnr" in result
        assert "ssim" in result
