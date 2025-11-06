#!/usr/bin/env python
"""Comprehensive evaluation of multi-task model.

This script provides complete evaluation including:
1. Segmentation metrics (DICE, IoU, sensitivity, specificity)
2. Survival prediction metrics (C-index, calibration)
3. Uncertainty calibration analysis
4. Per-patient detailed results
5. Comparison visualizations

Usage:
    python scripts/evaluate_multitask.py \
        --model models/multitask_unet/best_model.keras \
        --data-dir data/synthetic_v2_survival \
        --output results/multitask_evaluation \
        --n-mc-samples 30
"""

import argparse
from pathlib import Path
import sys
import json
import numpy as np
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm

from med_seg.data import PETCTPreprocessor
from med_seg.data.survival_generator import create_survival_generators


def evaluate_segmentation(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate segmentation metrics.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    y_pred_binary = (y_pred > 0.5).astype(float)

    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()

    # True positives, false positives, false negatives, true negatives
    tp = np.sum(y_true_flat * y_pred_flat)
    fp = np.sum((1 - y_true_flat) * y_pred_flat)
    fn = np.sum(y_true_flat * (1 - y_pred_flat))
    tn = np.sum((1 - y_true_flat) * (1 - y_pred_flat))

    # Metrics
    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    sensitivity = (tp + 1e-7) / (tp + fn + 1e-7)
    specificity = (tn + 1e-7) / (tn + fp + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
    }


def calculate_c_index(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    """Calculate concordance index for survival prediction.

    Args:
        times: Observed times
        events: Event indicators (1=event, 0=censored)
        risk_scores: Predicted risk scores

    Returns:
        C-index value
    """
    n = len(times)
    concordant = 0
    total_pairs = 0

    for i in range(n):
        if events[i] == 0:
            continue  # Skip censored

        for j in range(n):
            if times[i] < times[j]:  # i had event before j
                total_pairs += 1
                if risk_scores[i] > risk_scores[j]:  # Higher risk for earlier event
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:  # Tied
                    concordant += 0.5

    if total_pairs == 0:
        return 0.5

    c_index = concordant / total_pairs
    return c_index


def mc_dropout_predictions(model: keras.Model, x: np.ndarray, n_samples: int = 30) -> Dict:
    """Get predictions with uncertainty via MC Dropout.

    Args:
        model: Model with MC Dropout
        x: Input data (batch_size, H, W, C)
        n_samples: Number of MC samples

    Returns:
        Dictionary with mean and std predictions
    """
    seg_preds = []
    surv_preds = []

    for _ in range(n_samples):
        pred = model(x, training=True)
        seg_preds.append(pred["segmentation"].numpy())
        surv_preds.append(pred["survival"].numpy())

    seg_preds = np.array(seg_preds)  # (n_samples, batch, H, W, 1)
    surv_preds = np.array(surv_preds)  # (n_samples, batch, 1)

    return {
        "seg_mean": np.mean(seg_preds, axis=0),
        "seg_std": np.std(seg_preds, axis=0),
        "surv_mean": np.mean(surv_preds, axis=0),
        "surv_std": np.std(surv_preds, axis=0),
    }


def evaluate_model(model: keras.Model, data_gen, n_mc_samples: int = 30) -> Dict:
    """Comprehensive model evaluation.

    Args:
        model: Trained multi-task model
        data_gen: Data generator
        n_mc_samples: Number of MC samples for uncertainty

    Returns:
        Dictionary with all evaluation results
    """
    print("[*] Running evaluation...")

    seg_metrics_list = []
    survival_data = {"times": [], "events": [], "risks": [], "risk_stds": []}
    uncertainty_data = {"seg_stds": [], "seg_errors": []}

    for batch_idx in tqdm(range(len(data_gen)), desc="Evaluating"):
        x_batch, y_batch = data_gen[batch_idx]

        # MC Dropout predictions
        mc_preds = mc_dropout_predictions(model, x_batch, n_mc_samples)

        # Evaluate each sample in batch
        for i in range(len(x_batch)):
            # Segmentation
            y_true_seg = y_batch["segmentation"][i, ..., 0]
            y_pred_seg = mc_preds["seg_mean"][i, ..., 0]
            y_std_seg = mc_preds["seg_std"][i, ..., 0]

            seg_metrics = evaluate_segmentation(y_true_seg, y_pred_seg)
            seg_metrics_list.append(seg_metrics)

            # Uncertainty analysis
            seg_error = np.abs(y_pred_seg - y_true_seg)
            uncertainty_data["seg_stds"].extend(y_std_seg.flatten())
            uncertainty_data["seg_errors"].extend(seg_error.flatten())

            # Survival
            time, event = y_batch["survival"][i]
            risk = mc_preds["surv_mean"][i, 0]
            risk_std = mc_preds["surv_std"][i, 0]

            survival_data["times"].append(time)
            survival_data["events"].append(event)
            survival_data["risks"].append(risk)
            survival_data["risk_stds"].append(risk_std)

    # Aggregate segmentation metrics
    seg_results = {}
    for key in seg_metrics_list[0].keys():
        values = [m[key] for m in seg_metrics_list]
        seg_results[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Survival C-index
    c_index = calculate_c_index(
        np.array(survival_data["times"]),
        np.array(survival_data["events"]),
        np.array(survival_data["risks"]),
    )

    # Uncertainty calibration
    uncertainty_data["seg_stds"] = np.array(uncertainty_data["seg_stds"])
    uncertainty_data["seg_errors"] = np.array(uncertainty_data["seg_errors"])

    # Expected calibration error (ECE) - simplified
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (uncertainty_data["seg_stds"] >= bin_boundaries[i]) & (
            uncertainty_data["seg_stds"] < bin_boundaries[i + 1]
        )
        if np.sum(mask) > 0:
            avg_confidence = np.mean(uncertainty_data["seg_stds"][mask])
            avg_error = np.mean(uncertainty_data["seg_errors"][mask])
            ece += np.abs(avg_confidence - avg_error) * np.sum(mask)
    ece /= len(uncertainty_data["seg_stds"])

    results = {
        "segmentation": seg_results,
        "survival": {
            "c_index": float(c_index),
            "mean_risk": float(np.mean(survival_data["risks"])),
            "std_risk": float(np.std(survival_data["risks"])),
            "mean_uncertainty": float(np.mean(survival_data["risk_stds"])),
        },
        "uncertainty": {
            "ece": float(ece),
            "mean_seg_uncertainty": float(np.mean(uncertainty_data["seg_stds"])),
            "correlation_uncertainty_error": float(
                np.corrcoef(uncertainty_data["seg_stds"], uncertainty_data["seg_errors"])[0, 1]
            ),
        },
    }

    return results, uncertainty_data


def create_visualizations(results: Dict, uncertainty_data: Dict, output_dir: Path):
    """Create evaluation visualizations.

    Args:
        results: Evaluation results dictionary
        uncertainty_data: Uncertainty analysis data
        output_dir: Where to save plots
    """
    print("[*] Creating visualizations...")

    # 1. Metrics summary plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Multi-Task Model Evaluation Summary", fontsize=16, fontweight="bold")

    metrics = ["dice", "iou", "sensitivity", "specificity", "precision", "accuracy"]
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        data = results["segmentation"][metric]

        ax.bar(
            ["Value"], [data["mean"]], yerr=[data["std"]], color="#2E86AB", alpha=0.8, capsize=10
        )
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(
            f"{metric.upper()}: {data['mean']:.4f} ± {data['std']:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "segmentation_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Uncertainty calibration plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot: uncertainty vs error
    ax1.scatter(
        uncertainty_data["seg_stds"],
        uncertainty_data["seg_errors"],
        alpha=0.1,
        s=1,
        color="#2E86AB",
    )
    ax1.set_xlabel("Prediction Uncertainty (Std Dev)", fontsize=11)
    ax1.set_ylabel("Prediction Error", fontsize=11)
    ax1.set_title(
        f"Uncertainty vs Error\n(Correlation: {results['uncertainty']['correlation_uncertainty_error']:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Calibration bins
    n_bins = 10
    bin_boundaries = np.linspace(0, uncertainty_data["seg_stds"].max(), n_bins + 1)
    bin_centers = []
    bin_errors = []
    for i in range(n_bins):
        mask = (uncertainty_data["seg_stds"] >= bin_boundaries[i]) & (
            uncertainty_data["seg_stds"] < bin_boundaries[i + 1]
        )
        if np.sum(mask) > 10:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_errors.append(np.mean(uncertainty_data["seg_errors"][mask]))

    ax2.plot(bin_centers, bin_errors, "o-", color="#2E86AB", linewidth=2, markersize=8)
    ax2.plot([0, max(bin_centers)], [0, max(bin_centers)], "r--", label="Perfect calibration")
    ax2.set_xlabel("Mean Uncertainty in Bin", fontsize=11)
    ax2.set_ylabel("Mean Error in Bin", fontsize=11)
    ax2.set_title(
        f"Calibration Plot\n(ECE: {results['uncertainty']['ece']:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive multi-task model evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument(
        "--output", type=str, default="results/multitask_evaluation", help="Output directory"
    )
    parser.add_argument("--n-mc-samples", type=int, default=30, help="Number of MC Dropout samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    args = parser.parse_args()

    print("\n[+] Comprehensive Multi-Task Model Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_dir}")
    print(f"MC samples: {args.n_mc_samples}")
    print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("[1/4] Loading model...")
    model = keras.models.load_model(args.model, compile=False)

    # Create data generator
    print("[2/4] Creating data generator...")
    preprocessor = PETCTPreprocessor(target_size=(256, 256))
    _, val_gen = create_survival_generators(
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        train_fraction=0.7,
    )

    # Evaluate
    print("[3/4] Evaluating model...")
    results, uncertainty_data = evaluate_model(model, val_gen, args.n_mc_samples)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n[*] Segmentation Metrics:")
    for metric, values in results["segmentation"].items():
        print(f"  {metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")

    print("\n[*] Survival Prediction:")
    print(f"  C-index: {results['survival']['c_index']:.4f}")
    print(f"  Mean risk uncertainty: {results['survival']['mean_uncertainty']:.4f}")

    print("\n[*] Uncertainty Calibration:")
    print(f"  Expected Calibration Error (ECE): {results['uncertainty']['ece']:.4f}")
    print(
        f"  Correlation (uncertainty-error): {results['uncertainty']['correlation_uncertainty_error']:.4f}"
    )

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved results: {results_path}")

    # Create visualizations
    print("\n[4/4] Creating visualizations...")
    create_visualizations(results, uncertainty_data, output_dir)

    print("\n" + "=" * 70)
    print("[+] Evaluation complete!")
    print(f"\n[*] Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
