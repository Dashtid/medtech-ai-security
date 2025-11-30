"""
Autoencoder-based Anomaly Detection for Medical Device Traffic

Uses deep learning autoencoders to detect anomalous network traffic patterns
in DICOM and HL7 protocols. The model learns normal traffic patterns and
flags deviations as potential security threats.

Architecture:
- Encoder: Compresses 16 input features to latent space
- Decoder: Reconstructs original features from latent space
- Anomaly Score: Reconstruction error (MSE) indicates anomaly likelihood

Training:
- Train ONLY on normal traffic (unsupervised)
- Threshold set based on normal traffic reconstruction error distribution
- Traffic exceeding threshold flagged as anomalous

Usage:
    from medtech_ai_security.anomaly import AnomalyDetector

    detector = AnomalyDetector()
    detector.fit(normal_traffic_features)

    # Detect anomalies
    results = detector.detect(new_traffic)
    for result in results:
        if result.is_anomaly:
            print(f"ALERT: {result.anomaly_score:.4f}")
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of anomaly detection for a single sample."""

    sample_index: int
    reconstruction_error: float
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    confidence: float
    feature_contributions: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sample_index": self.sample_index,
            "reconstruction_error": float(self.reconstruction_error),
            "anomaly_score": float(self.anomaly_score),
            "threshold": float(self.threshold),
            "is_anomaly": self.is_anomaly,
            "confidence": float(self.confidence),
            "feature_contributions": (
                self.feature_contributions.tolist()
                if self.feature_contributions is not None
                else None
            ),
        }


class Autoencoder:
    """
    Simple autoencoder implemented with NumPy for portability.

    Architecture:
    - Input: 16 features
    - Encoder: 16 -> 12 -> 8 -> 4 (latent)
    - Decoder: 4 -> 8 -> 12 -> 16
    - Activation: ReLU (hidden), Sigmoid (output)
    """

    def __init__(
        self,
        input_dim: int = 16,
        latent_dim: int = 4,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 0.001,
        seed: int = 42,
    ):
        """
        Initialize autoencoder.

        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space (bottleneck)
            hidden_dims: Hidden layer dimensions (default: [12, 8])
            learning_rate: Learning rate for gradient descent
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [12, 8]
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed)

        # Build encoder dimensions: input -> hidden -> latent
        encoder_dims = [input_dim] + self.hidden_dims + [latent_dim]
        # Build decoder dimensions: latent -> hidden (reversed) -> input
        decoder_dims = [latent_dim] + self.hidden_dims[::-1] + [input_dim]

        # Initialize weights with Xavier initialization
        self.encoder_weights = []
        self.encoder_biases = []
        for i in range(len(encoder_dims) - 1):
            fan_in, fan_out = encoder_dims[i], encoder_dims[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.encoder_weights.append(
                self.rng.randn(fan_in, fan_out).astype(np.float32) * scale
            )
            self.encoder_biases.append(np.zeros(fan_out, dtype=np.float32))

        self.decoder_weights = []
        self.decoder_biases = []
        for i in range(len(decoder_dims) - 1):
            fan_in, fan_out = decoder_dims[i], decoder_dims[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.decoder_weights.append(
                self.rng.randn(fan_in, fan_out).astype(np.float32) * scale
            )
            self.decoder_biases.append(np.zeros(fan_out, dtype=np.float32))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(np.float32)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with numerical stability."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        s = self._sigmoid(x)
        return s * (1 - s)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        h = x
        for i, (w, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            h = h @ w + b
            if i < len(self.encoder_weights) - 1:
                h = self._relu(h)
        return h

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representation to output."""
        h = z
        for i, (w, b) in enumerate(zip(self.decoder_weights, self.decoder_biases)):
            h = h @ w + b
            if i < len(self.decoder_weights) - 1:
                h = self._relu(h)
            else:
                h = self._sigmoid(h)  # Output activation
        return h

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list, list]:
        """
        Forward pass through autoencoder.

        Returns:
            Tuple of (output, encoder_activations, decoder_activations)
        """
        # Encoder forward pass
        encoder_activations = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            z = h @ w + b
            if i < len(self.encoder_weights) - 1:
                h = self._relu(z)
            else:
                h = z  # Latent space (no activation)
            encoder_activations.append(h)

        # Decoder forward pass
        decoder_activations = [h]
        for i, (w, b) in enumerate(zip(self.decoder_weights, self.decoder_biases)):
            z = h @ w + b
            if i < len(self.decoder_weights) - 1:
                h = self._relu(z)
            else:
                h = self._sigmoid(z)
            decoder_activations.append(h)

        return h, encoder_activations, decoder_activations

    def backward(
        self,
        x: np.ndarray,
        encoder_activations: list,
        decoder_activations: list,
        output: np.ndarray,
    ) -> tuple[list, list, list, list]:
        """
        Backward pass to compute gradients.

        Returns:
            Tuple of (encoder_weight_grads, encoder_bias_grads,
                      decoder_weight_grads, decoder_bias_grads)
        """
        batch_size = x.shape[0]

        # Output error (MSE derivative)
        delta = (output - x) * self._sigmoid_derivative(output)

        # Decoder gradients (backward)
        decoder_weight_grads = []
        decoder_bias_grads = []
        for i in range(len(self.decoder_weights) - 1, -1, -1):
            dw = decoder_activations[i].T @ delta / batch_size
            db = delta.mean(axis=0)
            decoder_weight_grads.insert(0, dw)
            decoder_bias_grads.insert(0, db)

            if i > 0:
                delta = (delta @ self.decoder_weights[i].T) * self._relu_derivative(
                    decoder_activations[i]
                )

        # Pass error to encoder
        delta = delta @ self.decoder_weights[0].T

        # Encoder gradients (backward)
        encoder_weight_grads = []
        encoder_bias_grads = []
        for i in range(len(self.encoder_weights) - 1, -1, -1):
            if i < len(self.encoder_weights) - 1:
                delta = delta * self._relu_derivative(encoder_activations[i + 1])

            dw = encoder_activations[i].T @ delta / batch_size
            db = delta.mean(axis=0)
            encoder_weight_grads.insert(0, dw)
            encoder_bias_grads.insert(0, db)

            if i > 0:
                delta = delta @ self.encoder_weights[i].T

        return (
            encoder_weight_grads,
            encoder_bias_grads,
            decoder_weight_grads,
            decoder_bias_grads,
        )

    def _clip_gradients(self, grads: list, max_norm: float = 1.0) -> list:
        """Clip gradients to prevent exploding gradients."""
        clipped = []
        for g in grads:
            norm = np.linalg.norm(g)
            if norm > max_norm:
                g = g * max_norm / norm
            clipped.append(g)
        return clipped

    def train_step(self, x: np.ndarray) -> float:
        """
        Single training step.

        Returns:
            Mean squared error loss
        """
        # Forward pass
        output, enc_acts, dec_acts = self.forward(x)

        # Compute loss
        loss = np.mean((output - x) ** 2)

        # Backward pass
        enc_w_grads, enc_b_grads, dec_w_grads, dec_b_grads = self.backward(
            x, enc_acts, dec_acts, output
        )

        # Clip gradients to prevent numerical overflow
        enc_w_grads = self._clip_gradients(enc_w_grads)
        enc_b_grads = self._clip_gradients(enc_b_grads)
        dec_w_grads = self._clip_gradients(dec_w_grads)
        dec_b_grads = self._clip_gradients(dec_b_grads)

        # Update weights (gradient descent)
        for i in range(len(self.encoder_weights)):
            self.encoder_weights[i] -= self.learning_rate * enc_w_grads[i]
            self.encoder_biases[i] -= self.learning_rate * enc_b_grads[i]

        for i in range(len(self.decoder_weights)):
            self.decoder_weights[i] -= self.learning_rate * dec_w_grads[i]
            self.decoder_biases[i] -= self.learning_rate * dec_b_grads[i]

        return loss

    def fit(
        self,
        x: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Train the autoencoder.

        Args:
            x: Training data (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_split: Fraction for validation
            early_stopping_patience: Epochs without improvement before stopping
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        # Split validation set
        n_val = int(len(x) * validation_split)
        indices = self.rng.permutation(len(x))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        x_train, x_val = x[train_idx], x[val_idx]

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):
            # Shuffle training data
            shuffle_idx = self.rng.permutation(len(x_train))
            x_shuffled = x_train[shuffle_idx]

            # Mini-batch training
            train_losses = []
            for i in range(0, len(x_shuffled), batch_size):
                batch = x_shuffled[i : i + batch_size]
                loss = self.train_step(batch)
                train_losses.append(loss)

            train_loss = np.mean(train_losses)
            history["train_loss"].append(train_loss)

            # Validation loss
            val_output, _, _ = self.forward(x_val)
            val_loss = np.mean((val_output - x_val) ** 2)
            history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                best_weights = {
                    "encoder_weights": [w.copy() for w in self.encoder_weights],
                    "encoder_biases": [b.copy() for b in self.encoder_biases],
                    "decoder_weights": [w.copy() for w in self.decoder_weights],
                    "decoder_biases": [b.copy() for b in self.decoder_biases],
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}"
                )

        # Restore best weights
        if best_weights:
            self.encoder_weights = best_weights["encoder_weights"]
            self.encoder_biases = best_weights["encoder_biases"]
            self.decoder_weights = best_weights["decoder_weights"]
            self.decoder_biases = best_weights["decoder_biases"]

        return history

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct input through autoencoder."""
        output, _, _ = self.forward(x)
        return output

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Compute reconstruction error for each sample."""
        reconstructed = self.reconstruct(x)
        return np.mean((x - reconstructed) ** 2, axis=1)

    def save(self, path: Path | str) -> None:
        """Save model weights to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_data = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
        }

        with open(path / "model_config.json", "w") as f:
            json.dump(model_data, f, indent=2)

        np.savez(
            path / "weights.npz",
            **{f"encoder_w_{i}": w for i, w in enumerate(self.encoder_weights)},
            **{f"encoder_b_{i}": b for i, b in enumerate(self.encoder_biases)},
            **{f"decoder_w_{i}": w for i, w in enumerate(self.decoder_weights)},
            **{f"decoder_b_{i}": b for i, b in enumerate(self.decoder_biases)},
        )

        logger.info(f"Saved autoencoder model to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "Autoencoder":
        """Load model weights from disk."""
        path = Path(path)

        with open(path / "model_config.json") as f:
            config = json.load(f)

        model = cls(**config)

        weights = np.load(path / "weights.npz")

        # Load encoder weights
        i = 0
        while f"encoder_w_{i}" in weights:
            model.encoder_weights[i] = weights[f"encoder_w_{i}"]
            model.encoder_biases[i] = weights[f"encoder_b_{i}"]
            i += 1

        # Load decoder weights
        i = 0
        while f"decoder_w_{i}" in weights:
            model.decoder_weights[i] = weights[f"decoder_w_{i}"]
            model.decoder_biases[i] = weights[f"decoder_b_{i}"]
            i += 1

        logger.info(f"Loaded autoencoder model from {path}")
        return model


class AnomalyDetector:
    """
    Anomaly detector for medical device network traffic.

    Uses an autoencoder trained on normal traffic to detect anomalies
    based on reconstruction error. Higher reconstruction error indicates
    the sample deviates from learned normal patterns.

    Attributes:
        autoencoder: Trained autoencoder model
        threshold: Anomaly detection threshold
        mean: Mean of training data (for normalization)
        std: Std of training data (for normalization)
    """

    # Feature names for interpretability
    FEATURE_NAMES = [
        "time_of_day",
        "time_in_hour",
        "source_ip_hash",
        "dest_ip_hash",
        "source_port_norm",
        "dest_port_norm",
        "ae_calling_len",
        "ae_called_len",
        "command_type",
        "message_id_norm",
        "sop_class_hash",
        "dataset_size_norm",
        "transfer_syntax_hash",
        "is_association",
        "is_release",
        "is_abort",
    ]

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 0.001,
        threshold_percentile: float = 95.0,
        seed: int = 42,
    ):
        """
        Initialize anomaly detector.

        Args:
            latent_dim: Autoencoder latent dimension
            hidden_dims: Autoencoder hidden layer dimensions
            learning_rate: Autoencoder learning rate
            threshold_percentile: Percentile for threshold (e.g., 95 = top 5% flagged)
            seed: Random seed
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [12, 8]
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.seed = seed

        self.autoencoder: Autoencoder | None = None
        self.threshold: float = 0.0
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.is_fitted: bool = False

        # Training statistics
        self.train_errors: np.ndarray | None = None
        self.train_history: dict | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize features using training statistics."""
        if self.mean is None or self.std is None:
            raise ValueError("Model not fitted. Call fit() first.")
        # Avoid division by zero
        std = np.where(self.std == 0, 1, self.std)
        return (x - self.mean) / std

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize features."""
        if self.mean is None or self.std is None:
            raise ValueError("Model not fitted. Call fit() first.")
        std = np.where(self.std == 0, 1, self.std)
        return x * std + self.mean

    def fit(
        self,
        x: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> "AnomalyDetector":
        """
        Fit the anomaly detector on NORMAL traffic only.

        Args:
            x: Training features (n_samples, n_features) - NORMAL traffic only
            epochs: Training epochs
            batch_size: Mini-batch size
            validation_split: Validation data fraction
            early_stopping_patience: Early stopping patience
            verbose: Print progress

        Returns:
            self
        """
        logger.info(f"Fitting anomaly detector on {len(x)} samples")

        # Compute normalization statistics
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        # Normalize training data
        x_norm = self._normalize(x)

        # Initialize autoencoder
        self.autoencoder = Autoencoder(
            input_dim=x.shape[1],
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )

        # Train autoencoder
        self.train_history = self.autoencoder.fit(
            x_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose,
        )

        # Compute reconstruction errors on training data
        self.train_errors = self.autoencoder.reconstruction_error(x_norm)

        # Set threshold based on percentile
        self.threshold = float(np.percentile(self.train_errors, self.threshold_percentile))

        self.is_fitted = True

        logger.info(
            f"Training complete. Threshold: {self.threshold:.6f} "
            f"(percentile: {self.threshold_percentile})"
        )

        return self

    def detect(
        self,
        x: np.ndarray,
        return_contributions: bool = False,
    ) -> list[DetectionResult]:
        """
        Detect anomalies in new traffic samples.

        Args:
            x: Features to analyze (n_samples, n_features)
            return_contributions: Include per-feature contribution scores

        Returns:
            List of DetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Normalize
        x_norm = self._normalize(x)

        # Get reconstruction errors
        reconstruction_errors = self.autoencoder.reconstruction_error(x_norm)

        # Calculate anomaly scores (normalized by threshold)
        anomaly_scores = reconstruction_errors / self.threshold

        results = []
        for i in range(len(x)):
            # Compute confidence based on how far from threshold
            confidence = min(1.0, abs(anomaly_scores[i] - 1.0))

            # Feature contributions (which features deviate most)
            contributions = None
            if return_contributions:
                reconstructed = self.autoencoder.reconstruct(x_norm[i : i + 1])
                contributions = np.abs(x_norm[i] - reconstructed[0])

            result = DetectionResult(
                sample_index=i,
                reconstruction_error=float(reconstruction_errors[i]),
                anomaly_score=float(anomaly_scores[i]),
                threshold=self.threshold,
                is_anomaly=bool(reconstruction_errors[i] > self.threshold),
                confidence=confidence,
                feature_contributions=contributions,
            )
            results.append(result)

        return results

    def detect_batch(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch detection for efficiency.

        Args:
            x: Features (n_samples, n_features)

        Returns:
            Tuple of (is_anomaly, anomaly_scores, reconstruction_errors)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        x_norm = self._normalize(x)
        reconstruction_errors = self.autoencoder.reconstruction_error(x_norm)
        anomaly_scores = reconstruction_errors / self.threshold
        is_anomaly = reconstruction_errors > self.threshold

        return is_anomaly, anomaly_scores, reconstruction_errors

    def get_top_anomalies(
        self,
        x: np.ndarray,
        top_k: int = 10,
    ) -> list[DetectionResult]:
        """Get top-k most anomalous samples."""
        results = self.detect(x, return_contributions=True)
        results.sort(key=lambda r: r.anomaly_score, reverse=True)
        return results[:top_k]

    def explain_anomaly(self, result: DetectionResult) -> dict:
        """
        Explain why a sample was flagged as anomalous.

        Args:
            result: DetectionResult with feature_contributions

        Returns:
            Dictionary with explanation
        """
        if result.feature_contributions is None:
            return {"error": "No feature contributions available"}

        # Get top contributing features
        contributions = result.feature_contributions
        top_indices = np.argsort(contributions)[::-1][:5]

        explanation = {
            "is_anomaly": result.is_anomaly,
            "anomaly_score": result.anomaly_score,
            "threshold": result.threshold,
            "top_deviating_features": [
                {
                    "feature": self.FEATURE_NAMES[i] if i < len(self.FEATURE_NAMES) else f"feature_{i}",
                    "contribution": float(contributions[i]),
                    "index": int(i),
                }
                for i in top_indices
            ],
        }

        return explanation

    def evaluate(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """
        Evaluate detector performance.

        Args:
            x: Test features
            y_true: True labels (0=normal, 1=anomaly)

        Returns:
            Dictionary with metrics
        """
        is_anomaly, scores, _ = self.detect_batch(x)
        y_pred = is_anomaly.astype(int)

        # Basic metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true)

        # AUC approximation (simple trapezoid rule)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_true = y_true[sorted_indices]
        tpr_values = np.cumsum(sorted_true) / np.sum(y_true) if np.sum(y_true) > 0 else np.zeros(len(y_true))
        fpr_values = np.cumsum(1 - sorted_true) / np.sum(1 - y_true) if np.sum(1 - y_true) > 0 else np.zeros(len(y_true))
        # Use trapezoid (numpy >= 2.0) or fallback to trapz for older versions
        trapezoid_func = getattr(np, "trapezoid", np.trapz)
        auc = trapezoid_func(tpr_values, fpr_values)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "threshold": float(self.threshold),
        }

    def save(self, path: Path | str) -> None:
        """Save detector to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save autoencoder
        if self.autoencoder:
            self.autoencoder.save(path / "autoencoder")

        # Save detector state
        state = {
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "threshold_percentile": self.threshold_percentile,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted,
            "seed": self.seed,
        }

        with open(path / "detector_config.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save normalization statistics
        if self.mean is not None:
            np.savez(
                path / "normalization.npz",
                mean=self.mean,
                std=self.std,
            )

        # Save training statistics
        if self.train_errors is not None:
            np.save(path / "train_errors.npy", self.train_errors)

        if self.train_history:
            with open(path / "train_history.json", "w") as f:
                json.dump(self.train_history, f, indent=2)

        logger.info(f"Saved anomaly detector to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "AnomalyDetector":
        """Load detector from disk."""
        path = Path(path)

        with open(path / "detector_config.json") as f:
            config = json.load(f)

        detector = cls(
            latent_dim=config["latent_dim"],
            hidden_dims=config["hidden_dims"],
            learning_rate=config["learning_rate"],
            threshold_percentile=config["threshold_percentile"],
            seed=config["seed"],
        )

        detector.threshold = config["threshold"]
        detector.is_fitted = config["is_fitted"]

        # Load autoencoder
        if (path / "autoencoder").exists():
            detector.autoencoder = Autoencoder.load(path / "autoencoder")

        # Load normalization statistics
        if (path / "normalization.npz").exists():
            norm = np.load(path / "normalization.npz")
            detector.mean = norm["mean"]
            detector.std = norm["std"]

        # Load training statistics
        if (path / "train_errors.npy").exists():
            detector.train_errors = np.load(path / "train_errors.npy")

        if (path / "train_history.json").exists():
            with open(path / "train_history.json") as f:
                detector.train_history = json.load(f)

        logger.info(f"Loaded anomaly detector from {path}")
        return detector


def main():
    """CLI entry point for anomaly detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and run anomaly detection on medical device traffic"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train anomaly detector")
    train_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data directory",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default="models/anomaly_detector",
        help="Output path for trained model",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    train_parser.add_argument(
        "--latent-dim",
        type=int,
        default=4,
        help="Latent dimension",
    )
    train_parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Threshold percentile",
    )

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect anomalies")
    detect_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    detect_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data to analyze",
    )
    detect_parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (optional)",
    )
    detect_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top anomalies to show",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate detector")
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    eval_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to labeled test data",
    )

    args = parser.parse_args()

    if args.command == "train":
        # Load training data (normal traffic only)
        data_path = Path(args.data)
        features = np.load(data_path / "features.npy")
        labels = np.load(data_path / "labels.npy")

        # Filter normal traffic only for training
        normal_mask = labels == 0
        normal_features = features[normal_mask]

        print(f"\n{'=' * 60}")
        print("TRAINING ANOMALY DETECTOR")
        print("=" * 60)
        print(f"Total samples: {len(features)}")
        print(f"Normal samples (for training): {len(normal_features)}")
        print(f"Attack samples: {(labels == 1).sum()}")

        # Train detector
        detector = AnomalyDetector(
            latent_dim=args.latent_dim,
            threshold_percentile=args.threshold_percentile,
        )
        detector.fit(
            normal_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # Evaluate on full dataset
        metrics = detector.evaluate(features, labels)

        print(f"\n{'=' * 60}")
        print("EVALUATION ON TRAINING DATA")
        print("=" * 60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")

        # Save model
        detector.save(args.output)
        print(f"\nModel saved to: {args.output}")

    elif args.command == "detect":
        # Load model
        detector = AnomalyDetector.load(args.model)

        # Load data
        data_path = Path(args.data)
        features = np.load(data_path / "features.npy")

        print(f"\n{'=' * 60}")
        print("ANOMALY DETECTION")
        print("=" * 60)
        print(f"Analyzing {len(features)} samples...")

        # Get top anomalies
        top_anomalies = detector.get_top_anomalies(features, top_k=args.top_k)

        print(f"\nTop {args.top_k} anomalies:")
        print("-" * 60)
        for result in top_anomalies:
            explanation = detector.explain_anomaly(result)
            print(f"\nSample {result.sample_index}:")
            print(f"  Anomaly Score: {result.anomaly_score:.4f}")
            print(f"  Is Anomaly: {result.is_anomaly}")
            print("  Top Deviating Features:")
            for feat in explanation.get("top_deviating_features", [])[:3]:
                print(f"    - {feat['feature']}: {feat['contribution']:.4f}")

        # Summary
        is_anomaly, scores, errors = detector.detect_batch(features)
        n_anomalies = is_anomaly.sum()
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)
        print(f"Total anomalies detected: {n_anomalies} ({100*n_anomalies/len(features):.1f}%)")
        print(f"Mean anomaly score: {scores.mean():.4f}")
        print(f"Max anomaly score: {scores.max():.4f}")
        print(f"Threshold: {detector.threshold:.6f}")

        # Save results if output specified
        if args.output:
            results = [r.to_dict() for r in detector.detect(features)]
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif args.command == "evaluate":
        # Load model and data
        detector = AnomalyDetector.load(args.model)
        data_path = Path(args.data)
        features = np.load(data_path / "features.npy")
        labels = np.load(data_path / "labels.npy")

        metrics = detector.evaluate(features, labels)

        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Samples: {len(features)} (Normal: {(labels==0).sum()}, Attack: {(labels==1).sum()})")
        print("\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
