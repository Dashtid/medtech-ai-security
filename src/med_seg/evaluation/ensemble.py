"""Multi-expert ensemble evaluation for medical image segmentation."""

from typing import List, Dict
import numpy as np

from med_seg.evaluation.metrics import calculate_dice_scores


def ensemble_predictions(predictions: List[np.ndarray], method: str = "mean") -> np.ndarray:
    """Ensemble multiple predictions.

    Args:
        predictions: List of prediction arrays from different models/experts
        method: Ensembling method ('mean', 'median', 'max', 'vote')

    Returns:
        Ensembled prediction array
    """
    predictions_array = np.array(predictions)

    if method == "mean":
        return np.mean(predictions_array, axis=0)
    elif method == "median":
        return np.median(predictions_array, axis=0)
    elif method == "max":
        return np.max(predictions_array, axis=0)
    elif method == "vote":
        # Majority voting (assumes binary predictions)
        return (np.mean(predictions_array, axis=0) >= 0.5).astype(np.float32)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def multi_expert_evaluation(
    expert_predictions: List[np.ndarray],
    expert_ground_truths: List[np.ndarray],
    task_id: int = 1,
    thresholds: List[float] = None,
) -> Dict[str, any]:
    """Evaluate multi-expert segmentations as in QUBIQ challenge.

    This function replicates the evaluation strategy where:
    1. Multiple networks are trained (one per expert annotation)
    2. Predictions are averaged (ensembled)
    3. Ground truth masks are also averaged
    4. DICE scores are calculated at multiple thresholds

    Args:
        expert_predictions: List of predictions, one per expert (num_experts, N, H, W, 1)
        expert_ground_truths: List of ground truths, one per expert
        task_id: Task identifier for multi-task datasets
        thresholds: List of threshold values to evaluate

    Returns:
        Dictionary containing:
            - 'average_dice': Overall average DICE score
            - 'dice_matrix': DICE scores per sample and threshold
            - 'best_threshold': Threshold with highest average DICE
    """
    if thresholds is None:
        thresholds = [i / 10.0 for i in range(1, 10)]  # 0.1 to 0.9

    num_experts = len(expert_predictions)
    num_samples = expert_predictions[0].shape[0]

    print(f"[i] Evaluating {num_experts} expert predictions on {num_samples} samples")
    print(f"[i] Task ID: {task_id}")

    # Ensemble predictions (average across experts)
    ensembled_preds = ensemble_predictions(expert_predictions, method="mean")

    # Ensemble ground truths (average across experts)
    ensembled_gt = ensemble_predictions(expert_ground_truths, method="mean")

    # Calculate DICE scores at all thresholds
    dice_results = calculate_dice_scores(ensembled_preds, ensembled_gt, thresholds)

    # Create DICE matrix (thresholds x samples)
    dice_matrix = np.zeros((len(thresholds), num_samples))
    for i, threshold in enumerate(thresholds):
        dice_matrix[i, :] = dice_results[threshold]

    # Calculate average DICE across all thresholds and samples
    average_dice = np.mean(dice_matrix)

    # Find best threshold
    threshold_averages = np.mean(dice_matrix, axis=1)
    best_threshold_idx = np.argmax(threshold_averages)
    best_threshold = thresholds[best_threshold_idx]
    best_dice = threshold_averages[best_threshold_idx]

    print(f"[+] Average DICE score: {average_dice:.4f}")
    print(f"[+] Best threshold: {best_threshold:.2f} (DICE: {best_dice:.4f})")
    print("\n[i] DICE scores per threshold:")
    for threshold, avg_dice in zip(thresholds, threshold_averages):
        print(f"    Threshold {threshold:.1f}: {avg_dice:.4f}")

    return {
        "average_dice": float(average_dice),
        "dice_matrix": dice_matrix,
        "best_threshold": float(best_threshold),
        "best_dice": float(best_dice),
        "threshold_averages": dict(zip(thresholds, threshold_averages.tolist())),
    }
