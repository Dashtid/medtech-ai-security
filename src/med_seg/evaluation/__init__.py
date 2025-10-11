"""Evaluation utilities for medical image segmentation."""

from med_seg.evaluation.metrics import evaluate_segmentation, calculate_dice_scores
from med_seg.evaluation.ensemble import ensemble_predictions, multi_expert_evaluation

__all__ = [
    "evaluate_segmentation",
    "calculate_dice_scores",
    "ensemble_predictions",
    "multi_expert_evaluation",
]
