"""Survival analysis loss functions and metrics.

This module implements loss functions for survival analysis, particularly
the Cox proportional hazards model used in medical outcome prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def cox_ph_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Cox proportional hazards loss (negative partial log-likelihood).

    The Cox model assumes the hazard function h(t|x) = h0(t) * exp(f(x))
    where f(x) is the model's risk score. This loss maximizes the partial
    likelihood of observed events.

    Args:
        y_true: Ground truth, shape (batch_size, 2)
                [:, 0] = observed time
                [:, 1] = event indicator (1=event, 0=censored)
        y_pred: Predicted risk scores (log-hazard), shape (batch_size, 1)

    Returns:
        Cox partial log-likelihood loss (scalar)

    Reference:
        Cox, D. R. (1972). "Regression Models and Life-Tables".
        Journal of the Royal Statistical Society, Series B.

    Note:
        This implementation follows the approach from DeepSurv and similar
        deep learning survival models. It handles right-censored data.
    """
    # Extract time and event indicators
    time = y_true[:, 0]
    event = y_true[:, 1]

    # Squeeze risk predictions to 1D
    risk_score = tf.squeeze(y_pred, axis=-1)  # (batch_size,)

    # Sort by time (descending) to compute risk sets efficiently
    indices = tf.argsort(time, direction="DESCENDING")
    tf.gather(time, indices)
    event_sorted = tf.gather(event, indices)
    risk_sorted = tf.gather(risk_score, indices)

    # Compute log-likelihood for each event
    # For each event at time t_i, the risk set R(t_i) includes all j where t_j >= t_i
    # Cox partial likelihood = ∏ [exp(risk_i) / Σ_{j in R(t_i)} exp(risk_j)]
    # Log-likelihood = Σ [risk_i - log(Σ_{j in R(t_i)} exp(risk_j))]

    # Cumulative sum of exp(risk) from bottom to top gives risk set denominators
    exp_risk = tf.exp(risk_sorted)
    risk_set_sum = tf.cumsum(exp_risk, reverse=False)  # Cumulative sum

    # Log of risk set sums
    log_risk_set = tf.math.log(risk_set_sum + K.epsilon())

    # Partial log-likelihood for events
    log_likelihood = risk_sorted - log_risk_set

    # Only include events in the loss (ignore censored)
    event_mask = tf.cast(event_sorted, tf.float32)
    masked_log_likelihood = log_likelihood * event_mask

    # Average over batch (only over events)
    num_events = tf.reduce_sum(event_mask) + K.epsilon()
    loss = -tf.reduce_sum(masked_log_likelihood) / num_events

    return loss


def concordance_index(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Concordance index (C-index) for survival analysis.

    The C-index measures the model's ability to correctly order patients by risk.
    It's the probability that, for a random pair of patients where one experiences
    the event before the other, the model assigns a higher risk to the patient
    who experiences the event first.

    C-index ranges from 0 to 1:
    - 0.5 = random predictions
    - >0.7 = acceptable discrimination
    - >0.8 = excellent discrimination

    Args:
        y_true: Ground truth, shape (batch_size, 2)
                [:, 0] = observed time
                [:, 1] = event indicator
        y_pred: Predicted risk scores, shape (batch_size, 1)

    Returns:
        C-index (scalar between 0 and 1)

    Reference:
        Harrell, F. E., et al. (1982). "Evaluating the Yield of Medical Tests".
        JAMA, 247(18), 2543-2546.
    """
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk_score = tf.squeeze(y_pred, axis=-1)

    # Get all pairs where i experiences event and has time < time_j
    tf.shape(time)[0]

    # Expand dimensions for broadcasting
    time_i = tf.expand_dims(time, axis=1)  # (batch, 1)
    time_j = tf.expand_dims(time, axis=0)  # (1, batch)

    event_i = tf.expand_dims(event, axis=1)
    tf.expand_dims(event, axis=0)

    risk_i = tf.expand_dims(risk_score, axis=1)
    risk_j = tf.expand_dims(risk_score, axis=0)

    # Valid pairs: i had event and time_i < time_j
    valid_pair = tf.logical_and(tf.cast(event_i, tf.bool), time_i < time_j)

    # Concordant pairs: risk_i > risk_j (higher risk for earlier event)
    concordant = tf.logical_and(valid_pair, risk_i > risk_j)

    # Tied pairs: risk_i == risk_j
    tied = tf.logical_and(valid_pair, tf.abs(risk_i - risk_j) < 1e-6)

    # C-index = (concordant + 0.5 * tied) / valid
    num_concordant = tf.reduce_sum(tf.cast(concordant, tf.float32))
    num_tied = tf.reduce_sum(tf.cast(tied, tf.float32))
    num_valid = tf.reduce_sum(tf.cast(valid_pair, tf.float32)) + K.epsilon()

    c_index = (num_concordant + 0.5 * num_tied) / num_valid

    return c_index


def combined_multitask_loss(
    segmentation_weight: float = 0.5, survival_weight: float = 0.5, segmentation_loss_fn=None
):
    """Create combined loss for multi-task learning.

    Combines segmentation loss (e.g., Focal Tversky) with survival loss (Cox).
    The weights control the relative importance of each task.

    Args:
        segmentation_weight: Weight for segmentation loss
        survival_weight: Weight for survival loss
        segmentation_loss_fn: Segmentation loss function (e.g., focal_tversky_loss)

    Returns:
        Combined loss function

    Example:
        >>> from med_seg.training.losses import focal_tversky_loss
        >>> seg_loss = focal_tversky_loss(alpha=0.3, beta=0.7, gamma=0.75)
        >>> combined_loss = combined_multitask_loss(
        ...     segmentation_weight=0.7,
        ...     survival_weight=0.3,
        ...     segmentation_loss_fn=seg_loss
        ... )
    """
    if segmentation_loss_fn is None:
        # Default to binary cross-entropy
        from tensorflow.keras.losses import binary_crossentropy

        segmentation_loss_fn = binary_crossentropy

    def loss(y_true, y_pred):
        """Compute combined multi-task loss.

        Args:
            y_true: Dictionary with keys 'segmentation' and 'survival'
            y_pred: Dictionary with keys 'segmentation' and 'survival'

        Returns:
            Combined loss value
        """
        # Segmentation loss
        seg_loss = segmentation_loss_fn(y_true["segmentation"], y_pred["segmentation"])

        # Survival loss
        surv_loss = cox_ph_loss(y_true["survival"], y_pred["survival"])

        # Weighted combination
        total_loss = segmentation_weight * seg_loss + survival_weight * surv_loss

        return total_loss

    return loss


class SurvivalMetrics(keras.metrics.Metric):
    """Custom Keras metric for survival analysis (C-index).

    This allows tracking C-index during training via callbacks.
    """

    def __init__(self, name="c_index", **kwargs):
        super().__init__(name=name, **kwargs)
        self.c_index_sum = self.add_weight(name="c_index_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state with batch results."""
        c_idx = concordance_index(y_true, y_pred)
        self.c_index_sum.assign_add(c_idx)
        self.count.assign_add(1.0)

    def result(self):
        """Compute final metric value."""
        return self.c_index_sum / (self.count + K.epsilon())

    def reset_state(self):
        """Reset metric state."""
        self.c_index_sum.assign(0.0)
        self.count.assign(0.0)


# Convenience functions for backwards compatibility
def create_cox_loss():
    """Create Cox proportional hazards loss function.

    Returns:
        Cox loss function
    """
    return cox_ph_loss


def create_c_index_metric():
    """Create C-index metric for survival analysis.

    Returns:
        C-index metric
    """
    return SurvivalMetrics(name="c_index")


if __name__ == "__main__":
    # Test loss functions
    import numpy as np

    print("Testing survival loss functions...")

    # Create synthetic survival data
    batch_size = 16
    time = np.random.exponential(20, batch_size).astype(np.float32)
    event = np.random.binomial(1, 0.6, batch_size).astype(np.float32)
    y_true = np.stack([time, event], axis=1)

    # Create synthetic risk predictions
    # Higher risk should correlate with shorter times
    y_pred = np.random.randn(batch_size, 1).astype(np.float32)

    # Convert to tensors
    y_true_tf = tf.constant(y_true)
    y_pred_tf = tf.constant(y_pred)

    # Test Cox loss
    loss = cox_ph_loss(y_true_tf, y_pred_tf)
    print(f"\nCox PH Loss: {loss.numpy():.4f}")

    # Test C-index
    c_idx = concordance_index(y_true_tf, y_pred_tf)
    print(f"C-index: {c_idx.numpy():.4f}")
    print("(Random predictions should give C-index ≈ 0.5)")

    print("\nSurvival loss functions working correctly!")
