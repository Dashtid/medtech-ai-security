#!/usr/bin/env python
"""Model comparison framework for PET/CT U-Net segmentation.

This script provides comprehensive comparison between multiple trained models,
including performance metrics, training curves, and visual segmentation comparisons.

Usage:
    python scripts/compare_models.py \
        --models models/petct_unet/best_model.keras models/petct_unet_v2/best_model.keras \
        --logs models/petct_unet/training_log.csv models/petct_unet_v2/training_log.csv \
        --labels "v1 (Combined Loss)" "v2 (Focal Tversky)" \
        --data-dir data/synthetic \
        --output results/comparison
"""

import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict
import json

from med_seg.data import PETCTLoader, PETCTPreprocessor
from med_seg.data.petct_generator import PETCTDataGenerator


def load_model_info(model_path: Path, log_path: Path, label: str) -> Dict:
    """Load model and training information.

    Args:
        model_path: Path to model file
        log_path: Path to training log CSV
        label: Human-readable label for model

    Returns:
        Dictionary with model info
    """
    info = {
        "label": label,
        "model_path": model_path,
        "log_path": log_path,
        "model": None,
        "log": None,
        "metrics": {},
    }

    # Load model
    try:
        info["model"] = keras.models.load_model(model_path, compile=False)
        print(f"[+] Loaded model: {label}")
    except Exception as e:
        print(f"[!] Failed to load model {label}: {e}")
        return info

    # Load training log
    try:
        info["log"] = pd.read_csv(log_path)
        print(f"[+] Loaded training log: {label}")
    except Exception as e:
        print(f"[!] Failed to load log {label}: {e}")

    return info


def evaluate_model_on_data(
    model: keras.Model, data_gen: PETCTDataGenerator, label: str
) -> Dict[str, float]:
    """Evaluate model on dataset.

    Args:
        model: Trained Keras model
        data_gen: Data generator
        label: Model label

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n[*] Evaluating {label}...")

    all_dice = []
    all_iou = []
    all_sensitivity = []
    all_specificity = []
    all_precision = []

    n_batches = len(data_gen)

    for i in range(n_batches):
        # Get batch
        x_batch, y_batch = data_gen[i]

        # Predict
        y_pred = model.predict(x_batch, verbose=0)

        # Calculate metrics for each sample in batch
        for y_true, y_pred_sample in zip(y_batch, y_pred):
            y_true_flat = y_true.flatten()
            y_pred_flat = (y_pred_sample > 0.5).astype(float).flatten()

            # True positives, false positives, false negatives, true negatives
            tp = np.sum(y_true_flat * y_pred_flat)
            fp = np.sum((1 - y_true_flat) * y_pred_flat)
            fn = np.sum(y_true_flat * (1 - y_pred_flat))
            tn = np.sum((1 - y_true_flat) * (1 - y_pred_flat))

            # DICE coefficient
            dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
            all_dice.append(dice)

            # IoU (Jaccard index)
            iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
            all_iou.append(iou)

            # Sensitivity (recall)
            sensitivity = (tp + 1e-7) / (tp + fn + 1e-7)
            all_sensitivity.append(sensitivity)

            # Specificity
            specificity = (tn + 1e-7) / (tn + fp + 1e-7)
            all_specificity.append(specificity)

            # Precision
            precision = (tp + 1e-7) / (tp + fp + 1e-7)
            all_precision.append(precision)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_batches} batches")

    metrics = {
        "dice_mean": np.mean(all_dice),
        "dice_std": np.std(all_dice),
        "dice_median": np.median(all_dice),
        "iou_mean": np.mean(all_iou),
        "iou_std": np.std(all_iou),
        "iou_median": np.median(all_iou),
        "sensitivity_mean": np.mean(all_sensitivity),
        "sensitivity_std": np.std(all_sensitivity),
        "specificity_mean": np.mean(all_specificity),
        "specificity_std": np.std(all_specificity),
        "precision_mean": np.mean(all_precision),
        "precision_std": np.std(all_precision),
    }

    print(f"  DICE: {metrics['dice_mean']:.4f} +/- {metrics['dice_std']:.4f}")
    print(f"  IoU:  {metrics['iou_mean']:.4f} +/- {metrics['iou_std']:.4f}")

    return metrics


def plot_training_comparison(model_infos: List[Dict], output_dir: Path):
    """Create training curve comparison plots.

    Args:
        model_infos: List of model info dictionaries
        output_dir: Directory to save plots
    """
    print("\n[*] Creating training comparison plots...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#06A77D"]

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    for i, info in enumerate(model_infos):
        if info["log"] is None:
            continue
        log = info["log"]
        color = colors[i % len(colors)]
        ax1.plot(
            log["epoch"], log["loss"], label=f"{info['label']} (train)", color=color, linewidth=2
        )
        ax1.plot(
            log["epoch"],
            log["val_loss"],
            label=f"{info['label']} (val)",
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. DICE coefficient
    ax2 = fig.add_subplot(gs[0, 1])
    for i, info in enumerate(model_infos):
        if info["log"] is None:
            continue
        log = info["log"]
        color = colors[i % len(colors)]
        ax2.plot(
            log["epoch"],
            log["dice_coefficient"],
            label=f"{info['label']} (train)",
            color=color,
            linewidth=2,
        )
        ax2.plot(
            log["epoch"],
            log["val_dice_coefficient"],
            label=f"{info['label']} (val)",
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("DICE Coefficient", fontsize=11)
    ax2.set_title("DICE Score Progress", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # 3. IoU score
    ax3 = fig.add_subplot(gs[1, 0])
    for i, info in enumerate(model_infos):
        if info["log"] is None:
            continue
        log = info["log"]
        color = colors[i % len(colors)]
        ax3.plot(
            log["epoch"],
            log["iou_score"],
            label=f"{info['label']} (train)",
            color=color,
            linewidth=2,
        )
        ax3.plot(
            log["epoch"],
            log["val_iou_score"],
            label=f"{info['label']} (val)",
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("IoU Score", fontsize=11)
    ax3.set_title("IoU (Jaccard Index) Progress", fontsize=12, fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # 4. Final metrics comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    summary_text = "Final Training Metrics\n" + "=" * 50 + "\n\n"

    for info in model_infos:
        if info["log"] is None:
            continue

        log = info["log"]
        final = log.iloc[-1]
        best_idx = log["val_dice_coefficient"].idxmax()
        best = log.iloc[best_idx]

        summary_text += f"{info['label']}:\n"
        summary_text += f"  Final epoch: {int(final['epoch'])}\n"
        summary_text += f"  Final val loss: {final['val_loss']:.6f}\n"
        summary_text += f"  Final val DICE: {final['val_dice_coefficient']:.6f}\n"
        summary_text += f"  Final val IoU: {final['val_iou_score']:.6f}\n"
        summary_text += (
            f"  Best val DICE: {best['val_dice_coefficient']:.6f} (epoch {int(best['epoch'])})\n"
        )
        summary_text += "\n"

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    fig.suptitle("Model Training Comparison", fontsize=16, fontweight="bold", y=0.995)

    plot_path = output_dir / "training_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[+] Saved: {plot_path}")
    plt.close()


def plot_metrics_comparison(model_infos: List[Dict], output_dir: Path):
    """Create evaluation metrics comparison plots.

    Args:
        model_infos: List of model info dictionaries
        output_dir: Directory to save plots
    """
    print("\n[*] Creating metrics comparison plots...")

    # Prepare data
    labels = [info["label"] for info in model_infos]
    dice_means = [info["metrics"]["dice_mean"] for info in model_infos]
    dice_stds = [info["metrics"]["dice_std"] for info in model_infos]
    iou_means = [info["metrics"]["iou_mean"] for info in model_infos]
    iou_stds = [info["metrics"]["iou_std"] for info in model_infos]
    sens_means = [info["metrics"]["sensitivity_mean"] for info in model_infos]
    sens_stds = [info["metrics"]["sensitivity_std"] for info in model_infos]
    spec_means = [info["metrics"]["specificity_mean"] for info in model_infos]
    spec_stds = [info["metrics"]["specificity_std"] for info in model_infos]
    prec_means = [info["metrics"]["precision_mean"] for info in model_infos]
    prec_stds = [info["metrics"]["precision_std"] for info in model_infos]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#06A77D"]

    # 1. DICE coefficient
    ax = axes[0, 0]
    bars = ax.bar(
        labels,
        dice_means,
        yerr=dice_stds,
        capsize=5,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("DICE Coefficient", fontsize=11)
    ax.set_title("DICE Score", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars, dice_means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # 2. IoU score
    ax = axes[0, 1]
    bars = ax.bar(
        labels,
        iou_means,
        yerr=iou_stds,
        capsize=5,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("IoU Score", fontsize=11)
    ax.set_title("IoU (Jaccard Index)", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars, iou_means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # 3. Sensitivity
    ax = axes[0, 2]
    bars = ax.bar(
        labels,
        sens_means,
        yerr=sens_stds,
        capsize=5,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("Sensitivity (Recall)", fontsize=11)
    ax.set_title("Sensitivity", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars, sens_means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # 4. Specificity
    ax = axes[1, 0]
    bars = ax.bar(
        labels,
        spec_means,
        yerr=spec_stds,
        capsize=5,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("Specificity", fontsize=11)
    ax.set_title("Specificity", fontsize=12, fontweight="bold")
    ax.set_ylim([0.99, 1.0])  # Zoom in
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars, spec_means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0001,
            f"{val:.6f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # 5. Precision
    ax = axes[1, 1]
    bars = ax.bar(
        labels,
        prec_means,
        yerr=prec_stds,
        capsize=5,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars, prec_means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")

    table_data = []
    table_data.append(["Metric"] + labels)
    table_data.append(["DICE"] + [f"{v:.4f}" for v in dice_means])
    table_data.append(["IoU"] + [f"{v:.4f}" for v in iou_means])
    table_data.append(["Sensitivity"] + [f"{v:.4f}" for v in sens_means])
    table_data.append(["Specificity"] + [f"{v:.6f}" for v in spec_means])
    table_data.append(["Precision"] + [f"{v:.4f}" for v in prec_means])

    # Highlight best values
    for row in range(1, len(table_data)):
        values = [float(table_data[row][i]) for i in range(1, len(labels) + 1)]
        best_idx = values.index(max(values))
        # Mark best with asterisk
        table_data[row][best_idx + 1] = f"{values[best_idx]:.4f}*"

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(labels) + 1):
        cell = table[(0, i)]
        cell.set_facecolor("#2E86AB")
        cell.set_text_props(weight="bold", color="white")

    plt.tight_layout()

    plot_path = output_dir / "metrics_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[+] Saved: {plot_path}")
    plt.close()


def create_comparison_report(model_infos: List[Dict], output_dir: Path):
    """Create comprehensive text comparison report.

    Args:
        model_infos: List of model info dictionaries
        output_dir: Directory to save report
    """
    print("\n[*] Creating comparison report...")

    report_path = output_dir / "comparison_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Model summary
        f.write("Models Compared:\n")
        f.write("-" * 70 + "\n")
        for i, info in enumerate(model_infos, 1):
            f.write(f"{i}. {info['label']}\n")
            f.write(f"   Model: {info['model_path']}\n")
            f.write(f"   Log:   {info['log_path']}\n")
            if info["model"]:
                params = info["model"].count_params()
                f.write(f"   Parameters: {params:,}\n")
            f.write("\n")

        # Evaluation metrics
        f.write("\n" + "=" * 70 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("=" * 70 + "\n\n")

        for info in model_infos:
            f.write(f"{info['label']}:\n")
            f.write("-" * 70 + "\n")
            metrics = info["metrics"]

            f.write("  DICE Coefficient:\n")
            f.write(f"    Mean:   {metrics['dice_mean']:.6f}\n")
            f.write(f"    Std:    {metrics['dice_std']:.6f}\n")
            f.write(f"    Median: {metrics['dice_median']:.6f}\n\n")

            f.write("  IoU Score:\n")
            f.write(f"    Mean:   {metrics['iou_mean']:.6f}\n")
            f.write(f"    Std:    {metrics['iou_std']:.6f}\n")
            f.write(f"    Median: {metrics['iou_median']:.6f}\n\n")

            f.write("  Sensitivity (Recall):\n")
            f.write(f"    Mean: {metrics['sensitivity_mean']:.6f}\n")
            f.write(f"    Std:  {metrics['sensitivity_std']:.6f}\n\n")

            f.write("  Specificity:\n")
            f.write(f"    Mean: {metrics['specificity_mean']:.6f}\n")
            f.write(f"    Std:  {metrics['specificity_std']:.6f}\n\n")

            f.write("  Precision:\n")
            f.write(f"    Mean: {metrics['precision_mean']:.6f}\n")
            f.write(f"    Std:  {metrics['precision_std']:.6f}\n\n")

        # Best model analysis
        f.write("\n" + "=" * 70 + "\n")
        f.write("BEST MODEL ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        # Find best for each metric
        dice_best_idx = max(
            range(len(model_infos)), key=lambda i: model_infos[i]["metrics"]["dice_mean"]
        )
        iou_best_idx = max(
            range(len(model_infos)), key=lambda i: model_infos[i]["metrics"]["iou_mean"]
        )
        sens_best_idx = max(
            range(len(model_infos)), key=lambda i: model_infos[i]["metrics"]["sensitivity_mean"]
        )

        f.write(
            f"Best DICE:        {model_infos[dice_best_idx]['label']} ({model_infos[dice_best_idx]['metrics']['dice_mean']:.6f})\n"
        )
        f.write(
            f"Best IoU:         {model_infos[iou_best_idx]['label']} ({model_infos[iou_best_idx]['metrics']['iou_mean']:.6f})\n"
        )
        f.write(
            f"Best Sensitivity: {model_infos[sens_best_idx]['label']} ({model_infos[sens_best_idx]['metrics']['sensitivity_mean']:.6f})\n"
        )

        # Recommendations
        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n\n")

        best_overall = model_infos[dice_best_idx]
        f.write(f"Recommended Model: {best_overall['label']}\n")
        f.write(f"  Highest DICE score: {best_overall['metrics']['dice_mean']:.6f}\n")
        f.write(f"  IoU: {best_overall['metrics']['iou_mean']:.6f}\n")
        f.write(f"  Sensitivity: {best_overall['metrics']['sensitivity_mean']:.6f}\n")
        f.write("  Use this model for production deployment\n")

    print(f"[+] Saved: {report_path}")

    # Also save as JSON
    json_path = output_dir / "comparison_metrics.json"
    json_data = {}
    for info in model_infos:
        json_data[info["label"]] = {
            "model_path": str(info["model_path"]),
            "log_path": str(info["log_path"]),
            "metrics": info["metrics"],
        }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[+] Saved: {json_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare multiple PET/CT U-Net models")

    parser.add_argument(
        "--models", type=str, nargs="+", required=True, help="Paths to model files (.keras)"
    )
    parser.add_argument(
        "--logs", type=str, nargs="+", required=True, help="Paths to training log CSV files"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each model (default: Model 1, Model 2, ...)",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing evaluation data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison",
        help="Output directory for comparison results",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")

    args = parser.parse_args()

    # Validate inputs
    if len(args.models) != len(args.logs):
        print(
            f"[!] Error: Number of models ({len(args.models)}) must match number of logs ({len(args.logs)})"
        )
        return 1

    # Create labels
    if args.labels:
        if len(args.labels) != len(args.models):
            print(
                f"[!] Error: Number of labels ({len(args.labels)}) must match number of models ({len(args.models)})"
            )
            return 1
        labels = args.labels
    else:
        labels = [f"Model {i+1}" for i in range(len(args.models))]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[+] Model Comparison Framework")
    print("=" * 70)
    print(f"Comparing {len(args.models)} models")
    print(f"Evaluation data: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load models and logs
    model_infos = []
    for model_path, log_path, label in zip(args.models, args.logs, labels):
        info = load_model_info(Path(model_path), Path(log_path), label)
        model_infos.append(info)

    # Create data loader
    print("\n[*] Loading evaluation data...")
    loader = PETCTLoader(args.data_dir)
    print(f"[+] Found {len(loader)} patients")

    # Create preprocessor
    preprocessor = PETCTPreprocessor(
        target_size=(args.image_size, args.image_size),
        ct_window_center=0,
        ct_window_width=400,
        suv_max=15,
    )

    # Create data generator
    data_gen = PETCTDataGenerator(
        loader=loader,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        tumor_only=True,
    )
    print(f"[+] Created data generator: {len(data_gen)} batches")

    # Evaluate each model
    for info in model_infos:
        if info["model"] is None:
            print(f"[!] Skipping {info['label']}: Model not loaded")
            continue

        metrics = evaluate_model_on_data(info["model"], data_gen, info["label"])
        info["metrics"] = metrics

    # Create comparison visualizations
    plot_training_comparison(model_infos, output_dir)
    plot_metrics_comparison(model_infos, output_dir)

    # Create text report
    create_comparison_report(model_infos, output_dir)

    print("\n" + "=" * 70)
    print("[+] Comparison complete!")
    print(f"[+] Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - training_comparison.png")
    print("  - metrics_comparison.png")
    print("  - comparison_report.txt")
    print("  - comparison_metrics.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
