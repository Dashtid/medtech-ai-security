"""Loss functions for medical image segmentation.

This module implements common loss functions used in medical image
segmentation, particularly the DICE coefficient and its variants.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def dice_coefficient(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-7
) -> tf.Tensor:
    """Compute DICE coefficient for segmentation.

    The DICE coefficient (also known as F1 score or Sørensen–Dice coefficient)
    measures the overlap between predicted and ground truth segmentations.
    It ranges from 0 (no overlap) to 1 (perfect overlap).

    Formula: DICE = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities or binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        DICE coefficient as a scalar tensor

    Reference:
        Dice, Lee R. "Measures of the amount of ecologic association between species."
        Ecology 26.3 (1945): 297-302.
    """
    # Flatten tensors
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Calculate intersection and union
    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat)

    # Compute DICE coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice


def dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-7
) -> tf.Tensor:
    """DICE loss for segmentation (1 - DICE coefficient).

    Minimizing this loss is equivalent to maximizing the DICE coefficient.
    This loss function is particularly useful for imbalanced datasets where
    the foreground class is much smaller than the background.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities
        smooth: Smoothing factor to avoid division by zero

    Returns:
        DICE loss value
    """
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)


def binary_crossentropy_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor
) -> tf.Tensor:
    """Binary cross-entropy loss.

    Standard binary cross-entropy loss, commonly used for binary
    segmentation tasks.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities

    Returns:
        Binary cross-entropy loss value
    """
    return keras.losses.binary_crossentropy(y_true, y_pred)


def combined_loss(
    dice_weight: float = 0.5,
    bce_weight: float = 0.5
):
    """Create a combined DICE and binary cross-entropy loss.

    Combining DICE loss with binary cross-entropy can provide better
    gradient behavior and convergence properties. DICE loss handles
    class imbalance well, while BCE provides stable gradients.

    Args:
        dice_weight: Weight for DICE loss component
        bce_weight: Weight for binary cross-entropy component

    Returns:
        Loss function combining DICE and BCE

    Example:
        >>> loss_fn = combined_loss(dice_weight=0.7, bce_weight=0.3)
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        dice = dice_loss(y_true, y_pred)
        bce = binary_crossentropy_loss(y_true, y_pred)
        return dice_weight * dice + bce_weight * bce

    return loss


def focal_loss(
    alpha: float = 0.25,
    gamma: float = 2.0
):
    """Focal loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses training on hard
    examples. This is particularly useful when there's a severe class
    imbalance, such as in medical image segmentation where the region
    of interest may be very small.

    Args:
        alpha: Weighting factor in range (0, 1) to balance positive/negative examples
        gamma: Focusing parameter for modulating loss (gamma >= 0)

    Returns:
        Focal loss function

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Calculate focal loss
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = y_true * K.pow(1 - y_pred, gamma) + \
                      (1 - y_true) * K.pow(y_pred, gamma)

        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        focal = alpha_factor * focal_weight * bce

        return K.mean(focal)

    return loss


def tversky_loss(
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-7
):
    """Tversky loss - generalization of DICE loss.

    The Tversky index is a generalization of the DICE coefficient that
    allows for adjusting the penalty for false positives and false negatives.
    This is useful when you want to prioritize precision or recall.

    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives (typically alpha + beta = 1)
        smooth: Smoothing factor

    Returns:
        Tversky loss function

    Reference:
        Salehi et al., "Tversky loss function for image segmentation using 3D
        fully convolutional deep networks", MICCAI 2017.
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Flatten tensors
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)

        # Calculate true positives, false positives, false negatives
        true_pos = K.sum(y_true_flat * y_pred_flat)
        false_neg = K.sum(y_true_flat * (1 - y_pred_flat))
        false_pos = K.sum((1 - y_true_flat) * y_pred_flat)

        # Tversky index
        tversky = (true_pos + smooth) / \
                  (true_pos + alpha * false_pos + beta * false_neg + smooth)

        return 1.0 - tversky

    return loss
