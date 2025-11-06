"""Visualization utilities for training and results."""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history, save_path: Optional[str] = None, show: bool = True):
    """Plot training history (loss and metrics).

    Args:
        history: Keras History object or dictionary
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if hasattr(history, "history"):
        history = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    axes[0].plot(history["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
        # Mark best epoch
        best_epoch = np.argmin(history["val_loss"])
        best_loss = np.min(history["val_loss"])
        axes[0].plot(
            best_epoch, best_loss, "rx", markersize=12, label="Best Model", markeredgewidth=2
        )

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot main metric (DICE coefficient if available)
    metric_key = None
    for key in history.keys():
        if "dice" in key.lower() and "val" not in key:
            metric_key = key
            break

    if metric_key:
        axes[1].plot(history[metric_key], label=f"Training {metric_key}", linewidth=2)
        val_key = f"val_{metric_key}"
        if val_key in history:
            axes[1].plot(history[val_key], label=f"Validation {metric_key}", linewidth=2)
            best_epoch = np.argmax(history[val_key])
            best_metric = np.max(history[val_key])
            axes[1].plot(
                best_epoch, best_metric, "rx", markersize=12, label="Best Model", markeredgewidth=2
            )

        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel(metric_key.capitalize(), fontsize=12)
        axes[1].set_title(
            f"Training and Validation {metric_key.capitalize()}", fontsize=14, fontweight="bold"
        )
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Training history plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_segmentation_results(
    images: np.ndarray,
    ground_truths: np.ndarray,
    predictions: np.ndarray,
    num_samples: int = 5,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot segmentation results with original images, ground truth, and predictions.

    Args:
        images: Input images (N, H, W, C)
        ground_truths: Ground truth masks (N, H, W, 1)
        predictions: Predicted masks (N, H, W, 1)
        num_samples: Number of samples to plot
        threshold: Threshold for binarizing predictions
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original image
        if images.shape[-1] == 1:
            axes[i, 0].imshow(images[i, :, :, 0], cmap="gray")
        else:
            axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        # Ground truth
        axes[i, 1].imshow(ground_truths[i, :, :, 0], cmap="gray")
        axes[i, 1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

        # Prediction (continuous)
        axes[i, 2].imshow(predictions[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction (Probability)", fontsize=12, fontweight="bold")
        axes[i, 2].axis("off")

        # Prediction (binary)
        pred_binary = (predictions[i, :, :, 0] >= threshold).astype(np.float32)
        axes[i, 3].imshow(pred_binary, cmap="gray")
        axes[i, 3].set_title(f"Prediction (Threshold={threshold})", fontsize=12, fontweight="bold")
        axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Segmentation results plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dice_threshold_analysis(
    thresholds: List[float],
    dice_scores: List[float],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot DICE scores across different thresholds.

    Args:
        thresholds: List of threshold values
        dice_scores: List of corresponding DICE scores
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))

    plt.plot(thresholds, dice_scores, "b-o", linewidth=2, markersize=8)

    # Mark best threshold
    best_idx = np.argmax(dice_scores)
    plt.plot(
        thresholds[best_idx],
        dice_scores[best_idx],
        "r*",
        markersize=20,
        label=f"Best: {thresholds[best_idx]:.2f} (DICE={dice_scores[best_idx]:.4f})",
    )

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("DICE Score", fontsize=12)
    plt.title("DICE Score vs Threshold", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Threshold analysis plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
