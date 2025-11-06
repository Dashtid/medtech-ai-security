#!/usr/bin/env python
"""Real-time training monitor for PET/CT U-Net models.

This script monitors training progress in real-time by reading the training CSV log
and displaying live plots of loss and metrics. It can also compare multiple training
runs side-by-side.

Usage:
    # Monitor current training
    python scripts/monitor_training.py --log models/petct_unet_v2/training_log.csv

    # Compare two training runs
    python scripts/monitor_training.py \
        --log models/petct_unet_v2/training_log.csv \
        --compare models/petct_unet/training_log.csv \
        --labels "v2 (Focal Tversky)" "v1 (Combined Loss)"

    # Monitor with auto-refresh every 10 seconds
    python scripts/monitor_training.py --log models/petct_unet_v2/training_log.csv --refresh 10
"""

import argparse
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional
import numpy as np


def load_training_log(log_path: Path) -> Optional[pd.DataFrame]:
    """Load training log CSV file.

    Args:
        log_path: Path to training_log.csv

    Returns:
        DataFrame with training metrics or None if file doesn't exist
    """
    if not log_path.exists():
        return None

    try:
        df = pd.read_csv(log_path)
        return df
    except Exception as e:
        print(f"[!] Error reading {log_path}: {e}")
        return None


def plot_training_progress(
    logs: List[pd.DataFrame], labels: List[str], save_path: Optional[Path] = None
):
    """Create comprehensive training progress visualization.

    Args:
        logs: List of training log DataFrames
        labels: Labels for each log
        save_path: Optional path to save figure
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Define colors for up to 4 models
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#06A77D"]

    # 1. Loss curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = colors[i % len(colors)]
        ax1.plot(log["epoch"], log["loss"], label=f"{label} (train)", color=color, linewidth=2)
        ax1.plot(
            log["epoch"],
            log["val_loss"],
            label=f"{label} (val)",
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

    # 2. DICE coefficient (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = colors[i % len(colors)]
        ax2.plot(
            log["epoch"],
            log["dice_coefficient"],
            label=f"{label} (train)",
            color=color,
            linewidth=2,
        )
        ax2.plot(
            log["epoch"],
            log["val_dice_coefficient"],
            label=f"{label} (val)",
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

    # 3. IoU score (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = colors[i % len(colors)]
        ax3.plot(log["epoch"], log["iou_score"], label=f"{label} (train)", color=color, linewidth=2)
        ax3.plot(
            log["epoch"],
            log["val_iou_score"],
            label=f"{label} (val)",
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

    # 4. Accuracy (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = colors[i % len(colors)]
        ax4.plot(log["epoch"], log["accuracy"], label=f"{label} (train)", color=color, linewidth=2)
        ax4.plot(
            log["epoch"],
            log["val_accuracy"],
            label=f"{label} (val)",
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Accuracy", fontsize=11)
    ax4.set_title("Pixel Accuracy Progress", fontsize=12, fontweight="bold")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.99, 1.0])  # Zoom in on high accuracy range

    # 5. Learning rate (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = colors[i % len(colors)]
        if "lr" in log.columns:
            ax5.plot(log["epoch"], log["lr"], label=label, color=color, linewidth=2)

    ax5.set_xlabel("Epoch", fontsize=11)
    ax5.set_ylabel("Learning Rate", fontsize=11)
    ax5.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    ax5.legend(loc="best", fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale("log")

    # 6. Summary statistics (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    # Create summary table
    summary_text = "Current Training Status\n" + "=" * 50 + "\n\n"

    for i, (log, label) in enumerate(zip(logs, labels)):
        latest = log.iloc[-1]
        epoch = int(latest["epoch"])

        summary_text += f"{label}:\n"
        summary_text += f"  Epoch: {epoch}\n"
        summary_text += f"  Train Loss: {latest['loss']:.4f}\n"
        summary_text += f"  Val Loss: {latest['val_loss']:.4f}\n"
        summary_text += f"  Train DICE: {latest['dice_coefficient']:.4f}\n"
        summary_text += f"  Val DICE: {latest['val_dice_coefficient']:.4f}\n"
        summary_text += f"  Train IoU: {latest['iou_score']:.4f}\n"
        summary_text += f"  Val IoU: {latest['val_iou_score']:.4f}\n"

        # Find best val DICE
        best_idx = log["val_dice_coefficient"].idxmax()
        best_epoch = int(log.loc[best_idx, "epoch"])
        best_dice = log.loc[best_idx, "val_dice_coefficient"]

        summary_text += f"  Best Val DICE: {best_dice:.4f} (epoch {best_epoch})\n"
        summary_text += "\n"

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Main title
    fig.suptitle("PET/CT U-Net Training Monitor", fontsize=16, fontweight="bold", y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[+] Saved plot: {save_path}")

    plt.show(block=False)
    plt.pause(0.1)


def print_summary(logs: List[pd.DataFrame], labels: List[str]):
    """Print text summary of training progress.

    Args:
        logs: List of training log DataFrames
        labels: Labels for each log
    """
    print("\n" + "=" * 70)
    print("TRAINING PROGRESS SUMMARY")
    print("=" * 70)

    for log, label in zip(logs, labels):
        if log is None or len(log) == 0:
            print(f"\n[*] {label}: No data yet")
            continue

        latest = log.iloc[-1]
        epoch = int(latest["epoch"])

        print(f"\n[*] {label}:")
        print(f"    Current epoch: {epoch}")
        print(f"    Training loss: {latest['loss']:.6f}")
        print(f"    Validation loss: {latest['val_loss']:.6f}")
        print(f"    Training DICE: {latest['dice_coefficient']:.6f}")
        print(f"    Validation DICE: {latest['val_dice_coefficient']:.6f}")
        print(f"    Training IoU: {latest['iou_score']:.6f}")
        print(f"    Validation IoU: {latest['val_iou_score']:.6f}")

        # Best validation metrics
        best_idx = log["val_dice_coefficient"].idxmax()
        best_epoch = int(log.loc[best_idx, "epoch"])
        best_dice = log.loc[best_idx, "val_dice_coefficient"]
        best_loss = log.loc[best_idx, "val_loss"]

        print(f"    Best val DICE: {best_dice:.6f} (epoch {best_epoch})")
        print(f"    Best val loss: {best_loss:.6f}")

        # Check for potential issues
        if len(log) >= 3:
            recent_val_loss = log["val_loss"].tail(3).values
            if np.all(np.diff(recent_val_loss) > 0):
                print("    [!] WARNING: Val loss increasing for last 3 epochs")

            if epoch - best_epoch >= 5:
                print(f"    [!] NOTE: No improvement for {epoch - best_epoch} epochs")


def monitor_training(
    log_paths: List[Path],
    labels: List[str],
    refresh_interval: Optional[int] = None,
    output_dir: Optional[Path] = None,
):
    """Monitor training progress with optional auto-refresh.

    Args:
        log_paths: Paths to training log CSV files
        labels: Labels for each training run
        refresh_interval: Auto-refresh interval in seconds (None = single update)
        output_dir: Optional directory to save plots
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0

    try:
        while True:
            iteration += 1

            # Load all logs
            logs = []
            for path in log_paths:
                log = load_training_log(path)
                logs.append(log)

            # Check if any logs loaded
            if all(log is None for log in logs):
                print("[!] No training logs found yet. Waiting...")
                if refresh_interval is None:
                    break
                time.sleep(refresh_interval)
                continue

            # Filter out None logs for plotting
            valid_logs = []
            valid_labels = []
            for log, label in zip(logs, labels):
                if log is not None and len(log) > 0:
                    valid_logs.append(log)
                    valid_labels.append(label)

            if not valid_logs:
                print("[!] No valid training data yet. Waiting...")
                if refresh_interval is None:
                    break
                time.sleep(refresh_interval)
                continue

            # Clear terminal (optional)
            print("\033[2J\033[H", end="")  # ANSI escape codes to clear screen

            # Print text summary
            print_summary(logs, labels)

            # Create plots
            save_path = None
            if output_dir:
                save_path = output_dir / f"training_monitor_iter{iteration:03d}.png"

            plt.close("all")  # Close previous plots
            plot_training_progress(valid_logs, valid_labels, save_path)

            # Check if training is complete
            for log, label in zip(logs, labels):
                if log is None or len(log) == 0:
                    break

            # Exit if not in refresh mode
            if refresh_interval is None:
                print("\n[+] Single update complete. Use --refresh for continuous monitoring.")
                break

            # Wait for next refresh
            print(f"\n[*] Next update in {refresh_interval} seconds... (Ctrl+C to stop)")
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n[*] Monitoring stopped by user")

    plt.show()  # Keep final plot open


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Monitor PET/CT U-Net training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single update
  python scripts/monitor_training.py --log models/petct_unet_v2/training_log.csv

  # Auto-refresh every 10 seconds
  python scripts/monitor_training.py --log models/petct_unet_v2/training_log.csv --refresh 10

  # Compare two models
  python scripts/monitor_training.py \\
    --log models/petct_unet_v2/training_log.csv \\
    --compare models/petct_unet/training_log.csv \\
    --labels "v2 (Focal Tversky)" "v1 (Combined Loss)"
        """,
    )

    parser.add_argument("--log", type=str, required=True, help="Path to training_log.csv")
    parser.add_argument(
        "--compare", type=str, nargs="+", help="Additional training logs to compare"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each training run (default: 'Model 1', 'Model 2', ...)",
    )
    parser.add_argument(
        "--refresh", type=int, help="Auto-refresh interval in seconds (omit for single update)"
    )
    parser.add_argument("--output", type=str, help="Directory to save plots")

    args = parser.parse_args()

    # Build list of log paths
    log_paths = [Path(args.log)]
    if args.compare:
        log_paths.extend([Path(p) for p in args.compare])

    # Build labels
    if args.labels:
        if len(args.labels) != len(log_paths):
            print(
                f"[!] Error: Number of labels ({len(args.labels)}) must match number of logs ({len(log_paths)})"
            )
            return 1
        labels = args.labels
    else:
        labels = [f"Model {i+1}" for i in range(len(log_paths))]

    # Output directory
    output_dir = Path(args.output) if args.output else None

    print("\n[+] PET/CT U-Net Training Monitor")
    print("=" * 70)
    print(f"Monitoring: {len(log_paths)} training run(s)")
    for label, path in zip(labels, log_paths):
        print(f"  {label}: {path}")

    if args.refresh:
        print(f"Auto-refresh: Every {args.refresh} seconds")
    else:
        print("Mode: Single update")

    if output_dir:
        print(f"Saving plots to: {output_dir}")
    print()

    # Start monitoring
    monitor_training(log_paths, labels, args.refresh, output_dir)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
