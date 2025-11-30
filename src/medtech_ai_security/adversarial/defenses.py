"""
Adversarial Defense Methods for Medical AI Models

Implements defense strategies to improve robustness of medical imaging
AI models against adversarial attacks.

Defense Methods:
1. Adversarial Training - Train with adversarial examples
2. Input Preprocessing - JPEG compression, bit-depth reduction
3. Feature Squeezing - Gaussian blur, spatial smoothing
4. Ensemble Defenses - Multiple preprocessing techniques

Medical AI Considerations:
- Defense should not significantly degrade clean accuracy
- Feature squeezing can affect diagnostic quality
- Need balance between robustness and clinical utility

References:
- https://www.nature.com/articles/s41598-025-00890-x (Multi-layered defense 2025)
- https://arxiv.org/html/2506.17133 (Robust training with augmentation 2025)
- https://link.springer.com/article/10.1007/s10462-024-11005-9 (Robustness review)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefenseType(Enum):
    """Types of adversarial defenses."""

    ADVERSARIAL_TRAINING = "adversarial_training"
    JPEG_COMPRESSION = "jpeg_compression"
    BIT_DEPTH_REDUCTION = "bit_depth_reduction"
    GAUSSIAN_BLUR = "gaussian_blur"
    SPATIAL_SMOOTHING = "spatial_smoothing"
    FEATURE_SQUEEZING = "feature_squeezing"
    ENSEMBLE = "ensemble"


@dataclass
class DefenseResult:
    """Result of applying a defense."""

    original_images: np.ndarray
    defended_images: np.ndarray
    defense_type: DefenseType
    defense_params: dict
    clean_accuracy_before: float
    clean_accuracy_after: float
    adversarial_accuracy_before: float
    adversarial_accuracy_after: float
    accuracy_drop_clean: float
    accuracy_gain_adversarial: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "defense_type": self.defense_type.value,
            "defense_params": self.defense_params,
            "clean_accuracy_before": float(self.clean_accuracy_before),
            "clean_accuracy_after": float(self.clean_accuracy_after),
            "adversarial_accuracy_before": float(self.adversarial_accuracy_before),
            "adversarial_accuracy_after": float(self.adversarial_accuracy_after),
            "accuracy_drop_clean": float(self.accuracy_drop_clean),
            "accuracy_gain_adversarial": float(self.accuracy_gain_adversarial),
        }


class AdversarialDefender:
    """
    Adversarial defense system for medical AI models.

    Provides multiple defense mechanisms including input preprocessing,
    feature squeezing, and adversarial training utilities.

    Attributes:
        model: Target model to defend
    """

    def __init__(self, model: Callable | None = None):
        """
        Initialize defender.

        Args:
            model: Model to defend (for accuracy evaluation)
        """
        self.model = model

    def jpeg_compression(
        self,
        images: np.ndarray,
        quality: int = 75,
    ) -> np.ndarray:
        """
        Apply JPEG compression defense.

        JPEG compression removes high-frequency components that often
        contain adversarial perturbations.

        Args:
            images: Input images (batch, height, width, channels)
            quality: JPEG quality (1-100, lower = more compression)

        Returns:
            Compressed images
        """
        try:
            import io

            from PIL import Image
        except ImportError:
            logger.warning("PIL not available, skipping JPEG compression")
            return images

        defended = np.zeros_like(images)

        for i in range(len(images)):
            # Convert to PIL Image
            img = images[i]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            if img.shape[-1] == 1:
                img = np.squeeze(img)
                pil_img = Image.fromarray(img, mode="L")
            else:
                pil_img = Image.fromarray(img)

            # Compress and decompress
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            pil_img = Image.open(buffer)

            # Convert back
            img_array = np.array(pil_img)
            if len(img_array.shape) == 2:
                img_array = img_array[..., np.newaxis]

            if images.max() <= 1.0:
                img_array = img_array.astype(np.float32) / 255.0

            defended[i] = img_array

        return defended

    def bit_depth_reduction(
        self,
        images: np.ndarray,
        bits: int = 5,
    ) -> np.ndarray:
        """
        Reduce bit depth of images.

        Lower bit depth removes fine-grained perturbations.

        Args:
            images: Input images
            bits: Target bit depth (1-8)

        Returns:
            Bit-reduced images
        """
        if images.max() <= 1.0:
            # Convert to 0-255 range
            images_255 = images * 255
            reduced = np.round(images_255 / (256 / (2**bits))) * (256 / (2**bits))
            return np.clip(reduced / 255.0, 0, 1).astype(np.float32)
        else:
            reduced = np.round(images / (256 / (2**bits))) * (256 / (2**bits))
            return np.clip(reduced, 0, 255).astype(images.dtype)

    def gaussian_blur(
        self,
        images: np.ndarray,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        Apply Gaussian blur defense.

        Blurring smooths out high-frequency adversarial perturbations.

        Args:
            images: Input images
            sigma: Gaussian blur standard deviation

        Returns:
            Blurred images
        """
        defended = np.zeros_like(images)

        for i in range(len(images)):
            for c in range(images.shape[-1]):
                defended[i, :, :, c] = ndimage.gaussian_filter(
                    images[i, :, :, c], sigma=sigma
                )

        return defended.astype(np.float32)

    def spatial_smoothing(
        self,
        images: np.ndarray,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """
        Apply spatial smoothing (median filter) defense.

        Median filtering is effective against salt-and-pepper noise
        and some adversarial perturbations.

        Args:
            images: Input images
            kernel_size: Size of median filter kernel

        Returns:
            Smoothed images
        """
        defended = np.zeros_like(images)

        for i in range(len(images)):
            for c in range(images.shape[-1]):
                defended[i, :, :, c] = ndimage.median_filter(
                    images[i, :, :, c], size=kernel_size
                )

        return defended.astype(np.float32)

    def feature_squeezing(
        self,
        images: np.ndarray,
        bit_depth: int = 5,
        blur_sigma: float = 0.5,
    ) -> np.ndarray:
        """
        Apply feature squeezing defense.

        Combines bit-depth reduction and smoothing for robust defense.

        Args:
            images: Input images
            bit_depth: Target bit depth
            blur_sigma: Gaussian blur sigma

        Returns:
            Squeezed images
        """
        # Apply bit depth reduction
        squeezed = self.bit_depth_reduction(images, bits=bit_depth)

        # Apply Gaussian blur
        squeezed = self.gaussian_blur(squeezed, sigma=blur_sigma)

        return squeezed

    def ensemble_defense(
        self,
        images: np.ndarray,
        defenses: list | None = None,
    ) -> np.ndarray:
        """
        Apply ensemble of defenses.

        Combines multiple preprocessing techniques for stronger defense.

        Args:
            images: Input images
            defenses: List of (defense_fn, params) tuples

        Returns:
            Defended images (average of all defenses)
        """
        if defenses is None:
            defenses = [
                (self.bit_depth_reduction, {"bits": 5}),
                (self.gaussian_blur, {"sigma": 0.5}),
                (self.spatial_smoothing, {"kernel_size": 3}),
            ]

        defended_images = []

        for defense_fn, params in defenses:
            defended = defense_fn(images, **params)
            defended_images.append(defended)

        # Average all defended versions
        ensemble = np.mean(defended_images, axis=0)
        return ensemble.astype(np.float32)

    def detect_adversarial(
        self,
        images: np.ndarray,
        threshold: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect adversarial examples using feature squeezing.

        Compares model predictions on original vs squeezed images.
        Large differences indicate potential adversarial examples.

        Args:
            images: Input images
            threshold: Detection threshold (L1 distance)

        Returns:
            Tuple of (is_adversarial, detection_scores)
        """
        if self.model is None:
            raise ValueError("Model required for adversarial detection")

        # Get predictions on original images
        original_preds = self.model(images)
        if hasattr(original_preds, "numpy"):
            original_preds = original_preds.numpy()

        # Get predictions on squeezed images
        squeezed = self.feature_squeezing(images)
        squeezed_preds = self.model(squeezed)
        if hasattr(squeezed_preds, "numpy"):
            squeezed_preds = squeezed_preds.numpy()

        # Compute L1 distance between predictions
        detection_scores = np.mean(np.abs(original_preds - squeezed_preds), axis=-1)

        # Classify as adversarial if distance exceeds threshold
        is_adversarial = detection_scores > threshold

        return is_adversarial, detection_scores

    def evaluate_defense(
        self,
        clean_images: np.ndarray,
        clean_labels: np.ndarray,
        adversarial_images: np.ndarray,
        defense_type: DefenseType,
        defense_fn: Callable,
        defense_params: dict,
    ) -> DefenseResult:
        """
        Evaluate effectiveness of a defense.

        Measures accuracy on clean and adversarial images before/after defense.

        Args:
            clean_images: Clean test images
            clean_labels: True labels
            adversarial_images: Adversarial examples
            defense_type: Type of defense being evaluated
            defense_fn: Defense function to apply
            defense_params: Parameters for defense function

        Returns:
            DefenseResult with metrics
        """
        if self.model is None:
            raise ValueError("Model required for defense evaluation")

        def compute_accuracy(images: np.ndarray, labels: np.ndarray) -> float:
            preds = self.model(images)
            if hasattr(preds, "numpy"):
                preds = preds.numpy()

            if len(preds.shape) == 1 or preds.shape[-1] == 1:
                pred_classes = (np.squeeze(preds) > 0.5).astype(int)
            else:
                pred_classes = np.argmax(preds, axis=1)

            return float(np.mean(pred_classes == labels))

        # Accuracy before defense
        clean_acc_before = compute_accuracy(clean_images, clean_labels)
        adv_acc_before = compute_accuracy(adversarial_images, clean_labels)

        # Apply defense
        defended_clean = defense_fn(clean_images, **defense_params)
        defended_adv = defense_fn(adversarial_images, **defense_params)

        # Accuracy after defense
        clean_acc_after = compute_accuracy(defended_clean, clean_labels)
        adv_acc_after = compute_accuracy(defended_adv, clean_labels)

        logger.info(
            f"Defense {defense_type.value}: "
            f"clean {clean_acc_before:.2%} -> {clean_acc_after:.2%}, "
            f"adversarial {adv_acc_before:.2%} -> {adv_acc_after:.2%}"
        )

        return DefenseResult(
            original_images=adversarial_images,
            defended_images=defended_adv,
            defense_type=defense_type,
            defense_params=defense_params,
            clean_accuracy_before=clean_acc_before,
            clean_accuracy_after=clean_acc_after,
            adversarial_accuracy_before=adv_acc_before,
            adversarial_accuracy_after=adv_acc_after,
            accuracy_drop_clean=clean_acc_before - clean_acc_after,
            accuracy_gain_adversarial=adv_acc_after - adv_acc_before,
        )


class AdversarialTrainer:
    """
    Adversarial training for model robustness.

    Trains models with adversarial examples mixed into training data
    to improve robustness against attacks.
    """

    def __init__(
        self,
        model,
        attack_fn: Callable,
        attack_params: dict | None = None,
    ):
        """
        Initialize adversarial trainer.

        Args:
            model: Model to train (Keras model)
            attack_fn: Attack function to generate adversarial examples
            attack_params: Parameters for attack function
        """
        self.model = model
        self.attack_fn = attack_fn
        self.attack_params = attack_params or {"epsilon": 0.01}

    def generate_adversarial_batch(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        ratio: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate mixed batch of clean and adversarial examples.

        Args:
            images: Clean images
            labels: True labels
            ratio: Fraction of batch to make adversarial

        Returns:
            Tuple of (mixed_images, labels)
        """
        batch_size = len(images)
        num_adversarial = int(batch_size * ratio)

        if num_adversarial > 0:
            # Select indices for adversarial examples
            adv_indices = np.random.choice(
                batch_size, num_adversarial, replace=False
            )

            # Generate adversarial examples
            adv_images = images[adv_indices]
            adv_labels = labels[adv_indices]

            result = self.attack_fn(adv_images, adv_labels, **self.attack_params)

            # Replace selected images with adversarial versions
            mixed_images = images.copy()
            mixed_images[adv_indices] = result.adversarial_images

            return mixed_images, labels

        return images, labels

    def adversarial_training_step(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        ratio: float = 0.5,
    ):
        """
        Single adversarial training step.

        Args:
            images: Training images
            labels: Training labels
            ratio: Adversarial example ratio

        Returns:
            Training loss
        """
        # Generate mixed batch
        mixed_images, mixed_labels = self.generate_adversarial_batch(
            images, labels, ratio
        )

        # Train step (assumes Keras model)
        try:
            import tensorflow as tf

            with tf.GradientTape() as tape:
                predictions = self.model(mixed_images, training=True)

                if len(predictions.shape) == 1 or predictions.shape[-1] == 1:
                    loss = tf.keras.losses.binary_crossentropy(
                        tf.cast(mixed_labels, tf.float32),
                        tf.squeeze(predictions),
                    )
                else:
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        mixed_labels, predictions
                    )

                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer = self.model.optimizer
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return float(loss.numpy())

        except ImportError:
            logger.error("TensorFlow required for adversarial training")
            raise

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        adversarial_ratio: float = 0.5,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train model with adversarial examples.

        Args:
            x_train: Training images
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            adversarial_ratio: Ratio of adversarial examples
            validation_data: Optional (x_val, y_val) tuple
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        history = {"loss": [], "val_loss": [], "val_accuracy": []}

        num_batches = len(x_train) // batch_size

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(x_train))
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = start + batch_size

                batch_x = x_shuffled[start:end]
                batch_y = y_shuffled[start:end]

                loss = self.adversarial_training_step(
                    batch_x, batch_y, adversarial_ratio
                )
                epoch_losses.append(loss)

            mean_loss = np.mean(epoch_losses)
            history["loss"].append(mean_loss)

            # Validation
            if validation_data is not None:
                x_val, y_val = validation_data
                val_preds = self.model(x_val)
                if hasattr(val_preds, "numpy"):
                    val_preds = val_preds.numpy()

                if len(val_preds.shape) == 1 or val_preds.shape[-1] == 1:
                    val_classes = (np.squeeze(val_preds) > 0.5).astype(int)
                else:
                    val_classes = np.argmax(val_preds, axis=1)

                val_accuracy = np.mean(val_classes == y_val)
                history["val_accuracy"].append(float(val_accuracy))

                if verbose:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"loss: {mean_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                    )
            else:
                if verbose:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.4f}")

        return history
