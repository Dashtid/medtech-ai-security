"""Evaluation metrics for medical image segmentation."""

import tensorflow as tf
from tensorflow.keras import backend as K


def precision(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Calculate precision metric.

    Precision measures the proportion of positive predictions that are correct.
    Formula: Precision = TP / (TP + FP)

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities
        threshold: Threshold for binarizing predictions

    Returns:
        Precision score
    """
    # Binarize predictions
    y_pred_binary = K.cast(K.greater(y_pred, threshold), K.floatx())

    # Calculate true positives and predicted positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_binary, 0, 1)))

    # Calculate precision
    precision_score = true_positives / (predicted_positives + K.epsilon())

    return precision_score


def recall(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Calculate recall (sensitivity) metric.

    Recall measures the proportion of actual positives that are correctly identified.
    Formula: Recall = TP / (TP + FN)

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities
        threshold: Threshold for binarizing predictions

    Returns:
        Recall score
    """
    # Binarize predictions
    y_pred_binary = K.cast(K.greater(y_pred, threshold), K.floatx())

    # Calculate true positives and actual positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # Calculate recall
    recall_score = true_positives / (possible_positives + K.epsilon())

    return recall_score


def specificity(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Calculate specificity metric.

    Specificity measures the proportion of actual negatives that are correctly identified.
    Formula: Specificity = TN / (TN + FP)

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities
        threshold: Threshold for binarizing predictions

    Returns:
        Specificity score
    """
    # Binarize predictions
    y_pred_binary = K.cast(K.greater(y_pred, threshold), K.floatx())

    # Calculate true negatives and actual negatives
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred_binary), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))

    # Calculate specificity
    specificity_score = true_negatives / (possible_negatives + K.epsilon())

    return specificity_score


def iou_score(
    y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5, smooth: float = 1e-7
) -> tf.Tensor:
    """Calculate Intersection over Union (IoU) score.

    Also known as the Jaccard Index, IoU measures the overlap between
    predicted and ground truth masks.
    Formula: IoU = |X ∩ Y| / |X ∪ Y|

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU score
    """
    # Binarize predictions
    y_pred_binary = K.cast(K.greater(y_pred, threshold), K.floatx())

    # Flatten tensors
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred_binary)

    # Calculate intersection and union
    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection

    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)

    return iou


def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Calculate F1 score (equivalent to DICE for binary segmentation).

    F1 score is the harmonic mean of precision and recall.
    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities
        threshold: Threshold for binarizing predictions

    Returns:
        F1 score
    """
    prec = precision(y_true, y_pred, threshold)
    rec = recall(y_true, y_pred, threshold)

    f1 = 2 * (prec * rec) / (prec + rec + K.epsilon())

    return f1


def hausdorff_distance_approx(
    y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5
) -> tf.Tensor:
    """Approximate Hausdorff distance for segmentation.

    Hausdorff distance measures the maximum distance between boundaries
    of predicted and ground truth segmentations. This is an approximate
    implementation suitable for use as a metric during training.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities
        threshold: Threshold for binarizing predictions

    Returns:
        Approximate Hausdorff distance

    Note:
        This is a simplified approximation. For precise Hausdorff distance
        calculation, use dedicated post-processing functions.
    """
    # Binarize predictions
    y_pred_binary = K.cast(K.greater(y_pred, threshold), K.floatx())

    # This is a placeholder for a more sophisticated implementation
    # For production use, consider using scipy or specialized libraries
    # Here we use a simple edge-based approximation

    # Calculate edges using simple gradient
    true_edges = K.abs(y_true[:, 1:, :, :] - y_true[:, :-1, :, :]) + K.abs(
        y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    )
    pred_edges = K.abs(y_pred_binary[:, 1:, :, :] - y_pred_binary[:, :-1, :, :]) + K.abs(
        y_pred_binary[:, :, 1:, :] - y_pred_binary[:, :, :-1, :]
    )

    # Simplified distance metric
    distance = K.mean(K.abs(true_edges - pred_edges))

    return distance


class SegmentationMetrics:
    """Collection of segmentation metrics as a class.

    This class wraps all metrics and provides a convenient interface
    for computing multiple metrics at once.

    Example:
        >>> metrics = SegmentationMetrics(threshold=0.5)
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss='binary_crossentropy',
        ...     metrics=[metrics.dice, metrics.iou, metrics.precision]
        ... )
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize metrics with specified threshold.

        Args:
            threshold: Threshold for binarizing predictions
        """
        self.threshold = threshold

    def dice(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """DICE coefficient metric."""
        from med_seg.training.losses import dice_coefficient

        return dice_coefficient(y_true, y_pred)

    def iou(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """IoU metric."""
        return iou_score(y_true, y_pred, self.threshold)

    def precision_metric(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Precision metric."""
        return precision(y_true, y_pred, self.threshold)

    def recall_metric(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Recall metric."""
        return recall(y_true, y_pred, self.threshold)

    def f1(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """F1 score metric."""
        return f1_score(y_true, y_pred, self.threshold)
