"""
Adversarial Attack Methods for Medical AI Models

Implements state-of-the-art adversarial attacks for testing robustness
of medical imaging classification and segmentation models.

Attack Methods:
1. FGSM (Fast Gradient Sign Method) - Goodfellow et al., 2014
   - Single-step attack using gradient sign
   - Fast but less powerful
   - Good for adversarial training

2. PGD (Projected Gradient Descent) - Madry et al., 2017
   - Iterative FGSM with projection
   - Stronger attack, multiple iterations
   - Standard benchmark for robustness

3. C&W (Carlini & Wagner) - Carlini & Wagner, 2017
   - Optimization-based attack
   - Most powerful, finds minimal perturbations
   - Computationally expensive

Medical AI Considerations:
- Medical models are MORE vulnerable than natural image models
- Smaller perturbations needed for misclassification
- Clinical impact of adversarial attacks can be severe

References:
- https://arxiv.org/abs/1412.6572 (FGSM)
- https://arxiv.org/abs/1706.06083 (PGD)
- https://arxiv.org/abs/1608.04644 (C&W)
- https://www.nature.com/articles/s41598-025-00890-x (Medical imaging defense 2025)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of adversarial attacks."""

    FGSM = "fgsm"
    PGD = "pgd"
    CW_L2 = "cw_l2"
    CW_LINF = "cw_linf"


class AttackTarget(Enum):
    """Attack targeting mode."""

    UNTARGETED = "untargeted"  # Cause any misclassification
    TARGETED = "targeted"  # Force specific misclassification


@dataclass
class AttackResult:
    """Result of an adversarial attack."""

    original_images: np.ndarray
    adversarial_images: np.ndarray
    perturbations: np.ndarray
    original_predictions: np.ndarray
    adversarial_predictions: np.ndarray
    original_labels: np.ndarray
    attack_type: AttackType
    attack_params: dict
    success_rate: float
    mean_perturbation_l2: float
    mean_perturbation_linf: float
    num_samples: int
    successful_indices: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "attack_type": self.attack_type.value,
            "attack_params": self.attack_params,
            "success_rate": float(self.success_rate),
            "mean_perturbation_l2": float(self.mean_perturbation_l2),
            "mean_perturbation_linf": float(self.mean_perturbation_linf),
            "num_samples": self.num_samples,
            "num_successful": len(self.successful_indices),
        }


class AdversarialAttacker:
    """
    Adversarial attack generator for medical AI models.

    Implements FGSM, PGD, and C&W attacks with medical imaging
    considerations (smaller perturbations, clinical impact).

    Attributes:
        model: Target model to attack (callable or Keras model)
        loss_fn: Loss function for gradient computation
        clip_min: Minimum pixel value after perturbation
        clip_max: Maximum pixel value after perturbation
    """

    def __init__(
        self,
        model: Callable,
        loss_fn: Callable | None = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        num_classes: int = 2,
    ):
        """
        Initialize adversarial attacker.

        Args:
            model: Target model (callable that takes images, returns logits/probs)
            loss_fn: Loss function (default: cross-entropy)
            clip_min: Minimum valid pixel value
            clip_max: Maximum valid pixel value
            num_classes: Number of output classes
        """
        self.model = model
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.num_classes = num_classes

        # Try to import TensorFlow for gradient computation
        self._tf_available = False
        try:
            import tensorflow as tf

            self.tf = tf
            self._tf_available = True
        except ImportError:
            logger.warning("TensorFlow not available. Using NumPy-based attacks.")

    def _compute_gradient(
        self, images: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of loss w.r.t. input images."""
        if not self._tf_available:
            # Numerical gradient approximation for non-TF models
            return self._numerical_gradient(images, labels)

        # Check if model is a Keras model (has trainable_variables)
        is_keras = hasattr(self.model, "trainable_variables")
        if not is_keras:
            # Use numerical gradient for non-Keras models
            return self._numerical_gradient(images, labels)

        images_tensor = self.tf.constant(images, dtype=self.tf.float32)
        labels_tensor = self.tf.constant(labels, dtype=self.tf.int32)

        with self.tf.GradientTape() as tape:
            tape.watch(images_tensor)
            # Try with training argument (Keras models), fall back without
            try:
                predictions = self.model(images_tensor, training=False)
            except TypeError:
                predictions = self.model(images_tensor)

            # Handle different output formats
            if len(predictions.shape) == 1 or predictions.shape[-1] == 1:
                # Binary classification
                predictions = self.tf.squeeze(predictions)
                labels_float = self.tf.cast(labels_tensor, self.tf.float32)
                loss = self.tf.keras.losses.binary_crossentropy(
                    labels_float, predictions, from_logits=False
                )
            else:
                # Multi-class classification
                loss = self.tf.keras.losses.sparse_categorical_crossentropy(
                    labels_tensor, predictions, from_logits=False
                )

            loss = self.tf.reduce_mean(loss)

        gradient = tape.gradient(loss, images_tensor)
        return gradient.numpy()

    def _numerical_gradient(
        self, images: np.ndarray, labels: np.ndarray, eps: float = 1e-3
    ) -> np.ndarray:
        """
        Compute fast numerical gradient approximation.

        Uses random direction estimation for efficiency - samples random
        directions and estimates gradient along those directions.
        For non-Keras models, this provides a fast approximation.
        """
        batch_size = images.shape[0]
        gradient = np.zeros_like(images)

        # Number of random directions to sample per image
        n_directions = min(100, np.prod(images.shape[1:]))

        for i in range(batch_size):
            # Get base prediction and loss
            pred_base = self.model(images[i:i+1])
            if hasattr(pred_base, "numpy"):
                pred_base = pred_base.numpy()

            if len(pred_base.shape) == 1 or pred_base.shape[-1] == 1:
                pred_val = float(np.squeeze(pred_base))
                loss_base = -np.log(pred_val + 1e-8) if labels[i] == 1 else -np.log(1 - pred_val + 1e-8)
            else:
                loss_base = -np.log(pred_base[0, labels[i]] + 1e-8)

            # Estimate gradient using random perturbations
            for _ in range(n_directions):
                # Random direction
                direction = np.random.randn(*images.shape[1:]).astype(np.float32)
                direction = direction / (np.linalg.norm(direction) + 1e-8)

                # Forward perturbation
                images_plus = images[i:i+1].copy()
                images_plus[0] += eps * direction
                pred_plus = self.model(images_plus)
                if hasattr(pred_plus, "numpy"):
                    pred_plus = pred_plus.numpy()

                if len(pred_plus.shape) == 1 or pred_plus.shape[-1] == 1:
                    pred_val = float(np.squeeze(pred_plus))
                    loss_plus = -np.log(pred_val + 1e-8) if labels[i] == 1 else -np.log(1 - pred_val + 1e-8)
                else:
                    loss_plus = -np.log(pred_plus[0, labels[i]] + 1e-8)

                # Accumulate gradient estimate
                gradient[i] += (loss_plus - loss_base) / eps * direction

            # Average over directions
            gradient[i] /= n_directions

        return gradient

    def fgsm(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epsilon: float = 0.01,
        targeted: bool = False,
        target_labels: np.ndarray | None = None,
    ) -> AttackResult:
        """
        Fast Gradient Sign Method (FGSM) attack.

        Single-step attack that perturbs images in the direction of
        the gradient sign to maximize loss.

        Args:
            images: Input images (batch, height, width, channels)
            labels: True labels
            epsilon: Maximum perturbation magnitude (L-inf norm)
            targeted: If True, minimize loss for target_labels
            target_labels: Target labels for targeted attack

        Returns:
            AttackResult with adversarial examples
        """
        logger.info(f"Running FGSM attack with epsilon={epsilon}")

        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels)

        if targeted and target_labels is None:
            raise ValueError("target_labels required for targeted attack")

        attack_labels = target_labels if targeted else labels

        # Compute gradient
        gradient = self._compute_gradient(images, attack_labels)

        # Create perturbation
        if targeted:
            # Minimize loss for target (subtract gradient sign)
            perturbation = -epsilon * np.sign(gradient)
        else:
            # Maximize loss for true label (add gradient sign)
            perturbation = epsilon * np.sign(gradient)

        # Apply perturbation and clip
        adversarial_images = images + perturbation
        adversarial_images = np.clip(adversarial_images, self.clip_min, self.clip_max)

        # Recompute actual perturbation after clipping
        perturbation = adversarial_images - images

        # Evaluate attack success
        return self._evaluate_attack(
            images,
            adversarial_images,
            perturbation,
            labels,
            AttackType.FGSM,
            {"epsilon": epsilon, "targeted": targeted},
        )

    def pgd(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        num_iterations: int = 40,
        random_start: bool = True,
        targeted: bool = False,
        target_labels: np.ndarray | None = None,
    ) -> AttackResult:
        """
        Projected Gradient Descent (PGD) attack.

        Iterative attack that applies FGSM multiple times with projection
        to stay within epsilon-ball of original image.

        Args:
            images: Input images
            labels: True labels
            epsilon: Maximum perturbation (L-inf bound)
            alpha: Step size per iteration
            num_iterations: Number of attack iterations
            random_start: Start from random point in epsilon-ball
            targeted: Targeted attack mode
            target_labels: Target labels for targeted attack

        Returns:
            AttackResult with adversarial examples
        """
        logger.info(
            f"Running PGD attack with epsilon={epsilon}, "
            f"alpha={alpha}, iterations={num_iterations}"
        )

        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels)

        if targeted and target_labels is None:
            raise ValueError("target_labels required for targeted attack")

        attack_labels = target_labels if targeted else labels

        # Initialize adversarial images
        if random_start:
            # Random start within epsilon-ball
            adversarial_images = images + np.random.uniform(
                -epsilon, epsilon, images.shape
            ).astype(np.float32)
            adversarial_images = np.clip(
                adversarial_images, self.clip_min, self.clip_max
            )
        else:
            adversarial_images = images.copy()

        # Iterative attack
        for iteration in range(num_iterations):
            # Compute gradient at current adversarial images
            gradient = self._compute_gradient(adversarial_images, attack_labels)

            # Update adversarial images
            if targeted:
                adversarial_images = adversarial_images - alpha * np.sign(gradient)
            else:
                adversarial_images = adversarial_images + alpha * np.sign(gradient)

            # Project back to epsilon-ball around original images
            perturbation = adversarial_images - images
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            adversarial_images = images + perturbation

            # Clip to valid pixel range
            adversarial_images = np.clip(
                adversarial_images, self.clip_min, self.clip_max
            )

        # Final perturbation
        perturbation = adversarial_images - images

        return self._evaluate_attack(
            images,
            adversarial_images,
            perturbation,
            labels,
            AttackType.PGD,
            {
                "epsilon": epsilon,
                "alpha": alpha,
                "num_iterations": num_iterations,
                "random_start": random_start,
                "targeted": targeted,
            },
        )

    def cw_l2(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        confidence: float = 0.0,
        learning_rate: float = 0.01,
        binary_search_steps: int = 9,
        max_iterations: int = 1000,
        initial_const: float = 0.001,
        targeted: bool = False,
        target_labels: np.ndarray | None = None,
    ) -> AttackResult:
        """
        Carlini & Wagner L2 attack.

        Optimization-based attack that finds minimal L2 perturbation
        to cause misclassification.

        Args:
            images: Input images
            labels: True labels
            confidence: Confidence margin (kappa in paper)
            learning_rate: Optimization learning rate
            binary_search_steps: Steps for binary search on c
            max_iterations: Max optimization iterations per c
            initial_const: Initial value of c
            targeted: Targeted attack mode
            target_labels: Target labels

        Returns:
            AttackResult with adversarial examples
        """
        logger.info(
            f"Running C&W L2 attack with confidence={confidence}, "
            f"iterations={max_iterations}"
        )

        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels)

        if targeted and target_labels is None:
            raise ValueError("target_labels required for targeted attack")

        attack_labels = target_labels if targeted else labels
        batch_size = images.shape[0]

        # Initialize best adversarial examples
        best_adv = images.copy()
        best_l2 = np.full(batch_size, np.inf)

        # Binary search for optimal c
        c_lower = np.zeros(batch_size)
        c_upper = np.full(batch_size, 1e10)
        c = np.full(batch_size, initial_const)

        for search_step in range(binary_search_steps):
            # Initialize perturbation in tanh space
            # Map images to (-1, 1) range for tanh optimization
            images_tanh = np.arctanh(
                np.clip(images * 2 - 1, -0.999999, 0.999999)
            )
            w = images_tanh.copy()  # Optimization variable

            for iteration in range(max_iterations):
                # Transform w back to image space
                adv_images = (np.tanh(w) + 1) / 2  # Map to (0, 1)
                adv_images = np.clip(adv_images, self.clip_min, self.clip_max)

                # Get predictions
                predictions = self.model(adv_images)
                if hasattr(predictions, "numpy"):
                    predictions = predictions.numpy()

                # Compute L2 distance
                l2_dist = np.sqrt(
                    np.sum((adv_images - images) ** 2, axis=(1, 2, 3))
                )

                # Compute f(x') - the objective function
                if len(predictions.shape) == 1 or predictions.shape[-1] == 1:
                    # Binary classification
                    predictions = np.squeeze(predictions)
                    if targeted:
                        # Want to maximize P(target)
                        f_val = np.maximum(
                            0.5 - predictions if attack_labels == 1 else predictions - 0.5,
                            -confidence,
                        )
                    else:
                        # Want to minimize P(true label)
                        f_val = np.maximum(
                            predictions - 0.5 if labels == 1 else 0.5 - predictions,
                            -confidence,
                        )
                else:
                    # Multi-class: f = max(Z_t - max(Z_i, i!=t), -kappa)
                    if targeted:
                        # Want Z_target > max(Z_other)
                        target_logits = predictions[
                            np.arange(batch_size), attack_labels
                        ]
                        other_max = np.max(
                            predictions
                            - np.eye(self.num_classes)[attack_labels] * 1e10,
                            axis=1,
                        )
                        f_val = np.maximum(other_max - target_logits, -confidence)
                    else:
                        # Want max(Z_other) > Z_true
                        true_logits = predictions[np.arange(batch_size), labels]
                        other_max = np.max(
                            predictions - np.eye(self.num_classes)[labels] * 1e10,
                            axis=1,
                        )
                        f_val = np.maximum(true_logits - other_max, -confidence)

                # Total loss: L2 + c * f
                loss = l2_dist + c * f_val

                # Numerical gradient for w
                gradient = np.zeros_like(w)
                eps = 1e-4
                for i in range(batch_size):
                    for idx in range(min(100, np.prod(w.shape[1:]))):
                        # Sample random indices for efficiency
                        flat_idx = np.random.randint(0, np.prod(w.shape[1:]))
                        idx_tuple = np.unravel_index(flat_idx, w.shape[1:])
                        full_idx = (i,) + idx_tuple

                        w_plus = w.copy()
                        w_plus[full_idx] += eps
                        adv_plus = (np.tanh(w_plus) + 1) / 2
                        pred_plus = self.model(adv_plus[i : i + 1])
                        if hasattr(pred_plus, "numpy"):
                            pred_plus = pred_plus.numpy()

                        l2_plus = np.sqrt(np.sum((adv_plus[i] - images[i]) ** 2))

                        if len(pred_plus.shape) == 1 or pred_plus.shape[-1] == 1:
                            pred_plus = np.squeeze(pred_plus)
                            if targeted:
                                f_plus = max(0.5 - pred_plus if attack_labels[i] == 1 else pred_plus - 0.5, -confidence)
                            else:
                                f_plus = max(pred_plus - 0.5 if labels[i] == 1 else 0.5 - pred_plus, -confidence)
                        else:
                            if targeted:
                                f_plus = max(
                                    np.max(pred_plus[0]) - pred_plus[0, attack_labels[i]],
                                    -confidence,
                                )
                            else:
                                f_plus = max(
                                    pred_plus[0, labels[i]] - np.max(pred_plus[0]),
                                    -confidence,
                                )

                        loss_plus = l2_plus + c[i] * f_plus
                        gradient[full_idx] = (loss_plus - loss[i]) / eps

                # Update w
                w = w - learning_rate * gradient

                # Update best adversarial if successful
                for i in range(batch_size):
                    if f_val[i] <= 0 and l2_dist[i] < best_l2[i]:
                        best_l2[i] = l2_dist[i]
                        best_adv[i] = adv_images[i]

            # Update c via binary search
            for i in range(batch_size):
                if best_l2[i] < np.inf:
                    # Attack succeeded, try smaller c
                    c_upper[i] = min(c_upper[i], c[i])
                    c[i] = (c_lower[i] + c_upper[i]) / 2
                else:
                    # Attack failed, try larger c
                    c_lower[i] = max(c_lower[i], c[i])
                    if c_upper[i] < 1e10:
                        c[i] = (c_lower[i] + c_upper[i]) / 2
                    else:
                        c[i] *= 10

        perturbation = best_adv - images

        return self._evaluate_attack(
            images,
            best_adv,
            perturbation,
            labels,
            AttackType.CW_L2,
            {
                "confidence": confidence,
                "learning_rate": learning_rate,
                "binary_search_steps": binary_search_steps,
                "max_iterations": max_iterations,
                "targeted": targeted,
            },
        )

    def _evaluate_attack(
        self,
        original_images: np.ndarray,
        adversarial_images: np.ndarray,
        perturbations: np.ndarray,
        labels: np.ndarray,
        attack_type: AttackType,
        attack_params: dict,
    ) -> AttackResult:
        """Evaluate attack success and compute metrics."""
        # Get predictions
        original_preds = self.model(original_images)
        adversarial_preds = self.model(adversarial_images)

        if hasattr(original_preds, "numpy"):
            original_preds = original_preds.numpy()
            adversarial_preds = adversarial_preds.numpy()

        # Convert probabilities to class predictions
        if len(original_preds.shape) == 1 or original_preds.shape[-1] == 1:
            _original_classes = (np.squeeze(original_preds) > 0.5).astype(int)  # noqa: F841
            adversarial_classes = (np.squeeze(adversarial_preds) > 0.5).astype(int)
        else:
            _original_classes = np.argmax(original_preds, axis=1)  # noqa: F841
            adversarial_classes = np.argmax(adversarial_preds, axis=1)

        # Compute success rate (misclassification rate for untargeted)
        successful = adversarial_classes != labels
        success_rate = np.mean(successful)
        successful_indices = np.where(successful)[0].tolist()

        # Compute perturbation metrics
        l2_norms = np.sqrt(
            np.sum(perturbations**2, axis=tuple(range(1, perturbations.ndim)))
        )
        linf_norms = np.max(
            np.abs(perturbations), axis=tuple(range(1, perturbations.ndim))
        )

        mean_l2 = np.mean(l2_norms)
        mean_linf = np.mean(linf_norms)

        logger.info(
            f"Attack complete: success_rate={success_rate:.2%}, "
            f"mean_L2={mean_l2:.4f}, mean_Linf={mean_linf:.4f}"
        )

        return AttackResult(
            original_images=original_images,
            adversarial_images=adversarial_images,
            perturbations=perturbations,
            original_predictions=original_preds,
            adversarial_predictions=adversarial_preds,
            original_labels=labels,
            attack_type=attack_type,
            attack_params=attack_params,
            success_rate=float(success_rate),
            mean_perturbation_l2=float(mean_l2),
            mean_perturbation_linf=float(mean_linf),
            num_samples=len(labels),
            successful_indices=successful_indices,
        )

    def attack(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        attack_type: str | AttackType = AttackType.PGD,
        **kwargs,
    ) -> AttackResult:
        """
        Run specified attack on images.

        Args:
            images: Input images
            labels: True labels
            attack_type: Type of attack (fgsm, pgd, cw_l2)
            **kwargs: Attack-specific parameters

        Returns:
            AttackResult
        """
        if isinstance(attack_type, str):
            attack_type = AttackType(attack_type.lower())

        if attack_type == AttackType.FGSM:
            return self.fgsm(images, labels, **kwargs)
        elif attack_type == AttackType.PGD:
            return self.pgd(images, labels, **kwargs)
        elif attack_type == AttackType.CW_L2:
            return self.cw_l2(images, labels, **kwargs)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
