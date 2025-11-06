"""Evaluation metrics computation."""

from typing import Dict, List
import numpy as np


def dice_coefficient_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
    """Calculate DICE coefficient for numpy arrays.

    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        smooth: Smoothing factor

    Returns:
        DICE coefficient score
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return float(dice)


def calculate_dice_scores(
    predictions: np.ndarray, ground_truth: np.ndarray, thresholds: List[float] = None
) -> Dict[float, np.ndarray]:
    """Calculate DICE scores at multiple thresholds.

    Args:
        predictions: Predicted probability maps (N, H, W, 1)
        ground_truth: Ground truth masks (N, H, W, 1)
        thresholds: List of thresholds to evaluate

    Returns:
        Dictionary mapping thresholds to DICE scores for each sample
    """
    if thresholds is None:
        thresholds = [i / 10.0 for i in range(1, 10)]  # 0.1 to 0.9

    num_samples = len(predictions)
    results = {}

    for threshold in thresholds:
        dice_scores = np.zeros(num_samples)

        # Binarize predictions and ground truth
        pred_binary = (predictions >= threshold).astype(np.float32)
        gt_binary = (ground_truth >= threshold).astype(np.float32)

        # Calculate DICE for each sample
        for i in range(num_samples):
            dice_scores[i] = dice_coefficient_np(gt_binary[i], pred_binary[i])

        results[threshold] = dice_scores

    return results


def evaluate_segmentation(
    predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate segmentation with multiple metrics.

    Args:
        predictions: Predicted probability maps
        ground_truth: Ground truth masks
        threshold: Threshold for binarization

    Returns:
        Dictionary of metric names and values
    """
    # Binarize
    pred_binary = (predictions >= threshold).astype(np.float32)
    gt_binary = (ground_truth >= threshold).astype(np.float32)

    # Flatten
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()

    # Calculate metrics
    true_pos = np.sum(gt_flat * pred_flat)
    false_pos = np.sum((1 - gt_flat) * pred_flat)
    false_neg = np.sum(gt_flat * (1 - pred_flat))
    true_neg = np.sum((1 - gt_flat) * (1 - pred_flat))

    # DICE
    dice = (2 * true_pos) / (2 * true_pos + false_pos + false_neg + 1e-7)

    # IoU
    iou = true_pos / (true_pos + false_pos + false_neg + 1e-7)

    # Precision
    precision = true_pos / (true_pos + false_pos + 1e-7)

    # Recall
    recall = true_pos / (true_pos + false_neg + 1e-7)

    # Specificity
    specificity = true_neg / (true_neg + false_pos + 1e-7)

    # F1 Score (same as DICE for binary segmentation)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }
