#!/usr/bin/env python
"""Evaluate trained U-Net model on PET/CT data.

This script loads a trained model and evaluates it on test data,
generating visualizations and comprehensive metrics.

Usage:
    python scripts/evaluate_petct_unet.py --model models/petct_unet/best_model.keras --data-dir data/synthetic --output results/evaluation
"""

import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from tensorflow import keras

from med_seg.data import PETCTLoader, PETCTPreprocessor


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate comprehensive segmentation metrics.

    Args:
        y_true: Ground truth binary mask (H, W) or (N, H, W)
        y_pred: Predicted probability map (H, W) or (N, H, W)
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary with metrics: DICE, IoU, sensitivity, specificity, accuracy, precision
    """
    # Binarize prediction
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    y_true_binary = (y_true > 0.5).astype(np.float32)

    # Flatten arrays
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()

    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    # Calculate metrics
    epsilon = 1e-7  # Avoid division by zero

    # DICE coefficient (F1 score)
    dice = (2 * TP) / (2 * TP + FP + FN + epsilon)

    # IoU (Jaccard index)
    iou = TP / (TP + FP + FN + epsilon)

    # Sensitivity (Recall, True Positive Rate)
    sensitivity = TP / (TP + FN + epsilon)

    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP + epsilon)

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)

    # Precision (Positive Predictive Value)
    precision = TP / (TP + FP + epsilon)

    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN)
    }


def visualize_prediction(ct_slice, pet_slice, ground_truth, prediction, metrics, output_path=None):
    """Create comprehensive visualization with overlays.

    Args:
        ct_slice: CT image (H, W)
        pet_slice: PET/SUV image (H, W)
        ground_truth: Ground truth mask (H, W)
        prediction: Predicted probability map (H, W)
        metrics: Dictionary of metrics
        output_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Input images
    # CT
    axes[0, 0].imshow(ct_slice, cmap='gray')
    axes[0, 0].set_title('CT Input', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # PET/SUV
    axes[0, 1].imshow(pet_slice, cmap='hot')
    axes[0, 1].set_title('PET/SUV Input', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Ground Truth
    axes[0, 2].imshow(ct_slice, cmap='gray')
    # Overlay ground truth in green with transparency
    gt_overlay = np.zeros((*ground_truth.shape, 4))
    gt_overlay[ground_truth > 0.5] = [0, 1, 0, 0.5]  # Green, 50% transparent
    axes[0, 2].imshow(gt_overlay)
    axes[0, 2].set_title('Ground Truth Overlay', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Predictions and comparisons
    # Prediction probability map
    axes[1, 0].imshow(prediction, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction Probability', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    cbar1 = plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar1.set_label('Probability', rotation=270, labelpad=15)

    # Prediction overlay (red)
    axes[1, 1].imshow(ct_slice, cmap='gray')
    pred_binary = prediction > 0.5
    pred_overlay = np.zeros((*prediction.shape, 4))
    pred_overlay[pred_binary] = [1, 0, 0, 0.5]  # Red, 50% transparent
    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Comparison: Ground truth (green) vs Prediction (red)
    axes[1, 2].imshow(ct_slice, cmap='gray')

    # Create comparison overlay
    comparison_overlay = np.zeros((*ground_truth.shape, 4))
    gt_binary = ground_truth > 0.5

    # True Positive: Yellow (overlap)
    tp_mask = gt_binary & pred_binary
    comparison_overlay[tp_mask] = [1, 1, 0, 0.6]  # Yellow

    # False Positive: Red (predicted but not in ground truth)
    fp_mask = (~gt_binary) & pred_binary
    comparison_overlay[fp_mask] = [1, 0, 0, 0.6]  # Red

    # False Negative: Blue (in ground truth but not predicted)
    fn_mask = gt_binary & (~pred_binary)
    comparison_overlay[fn_mask] = [0, 0, 1, 0.6]  # Blue

    axes[1, 2].imshow(comparison_overlay)
    axes[1, 2].set_title('Comparison (TP=Yellow, FP=Red, FN=Blue)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    # Add metrics text
    metrics_text = (
        f"DICE: {metrics['dice']:.4f}\n"
        f"IoU: {metrics['iou']:.4f}\n"
        f"Sensitivity: {metrics['sensitivity']:.4f}\n"
        f"Specificity: {metrics['specificity']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    fig.text(0.98, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')

    plt.tight_layout(rect=[0, 0, 0.95, 1])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate(args):
    """Main evaluation function.

    Args:
        args: Command-line arguments
    """
    print("\n[+] PET/CT U-Net Evaluation")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)

    # Load model
    print("[1/5] Loading model...")
    model = keras.models.load_model(args.model, compile=False)
    print(f"  Model loaded: {args.model}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Create data loader
    print("\n[2/5] Loading data...")
    loader = PETCTLoader(args.data_dir)
    print(f"  Found {len(loader)} patients")

    # Create preprocessor
    preprocessor = PETCTPreprocessor(
        target_size=(args.image_size, args.image_size),
        ct_window_center=0,
        ct_window_width=400,
        suv_max=15
    )
    print(f"  Preprocessor: {preprocessor}")

    # Get tumor slices
    print("\n[3/5] Finding tumor slices...")
    all_slices = []
    for patient_idx in range(len(loader)):
        tumor_slices = loader.get_tumor_slices(patient_idx, axis=0, min_tumor_voxels=10)
        for slice_idx in tumor_slices:
            all_slices.append((patient_idx, slice_idx))

    print(f"  Total tumor slices: {len(all_slices)}")

    # Evaluate on slices
    print("\n[4/5] Running inference and computing metrics...")
    all_metrics = []
    num_visualizations = min(args.num_visualizations, len(all_slices))

    # Select slices for visualization (evenly spaced)
    vis_indices = np.linspace(0, len(all_slices) - 1, num_visualizations, dtype=int)

    for idx, (patient_idx, slice_idx) in enumerate(all_slices):
        # Extract slice
        ct_batch, suv_batch, seg_batch = loader.extract_2d_slices(
            patient_idx, axis=0, slice_indices=[slice_idx]
        )

        ct_slice = ct_batch[0]
        suv_slice = suv_batch[0]
        seg_slice = seg_batch[0]

        # Preprocess
        input_processed, seg_processed = preprocessor.preprocess_2d_slice(
            ct_slice, suv_slice, seg_slice
        )

        # Add batch dimension
        input_batch = np.expand_dims(input_processed, axis=0)

        # Run inference
        prediction = model.predict(input_batch, verbose=0)[0, :, :, 0]

        # Calculate metrics
        metrics = calculate_metrics(seg_processed[:, :, 0], prediction, threshold=0.5)
        metrics['patient_idx'] = patient_idx
        metrics['slice_idx'] = slice_idx
        all_metrics.append(metrics)

        # Visualize selected slices
        if idx in vis_indices:
            vis_idx = np.where(vis_indices == idx)[0][0]
            output_path = output_dir / 'visualizations' / f'prediction_{vis_idx:03d}_patient{patient_idx}_slice{slice_idx}.png'

            visualize_prediction(
                ct_slice=input_processed[:, :, 0],
                pet_slice=input_processed[:, :, 1],
                ground_truth=seg_processed[:, :, 0],
                prediction=prediction,
                metrics=metrics,
                output_path=output_path
            )

            print(f"  [{idx+1}/{len(all_slices)}] Patient {patient_idx}, Slice {slice_idx}: "
                  f"DICE={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")

    # Compute aggregate statistics
    print("\n[5/5] Computing aggregate statistics...")

    metrics_summary = {}
    for metric_name in ['dice', 'iou', 'sensitivity', 'specificity', 'accuracy', 'precision']:
        values = [m[metric_name] for m in all_metrics]
        metrics_summary[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    # Print results
    print("\n" + "="*70)
    print("[+] Evaluation Results")
    print("="*70)
    print(f"Total slices evaluated: {len(all_metrics)}")
    print()

    print("Metric Summary (Mean ± Std):")
    print("-" * 70)
    for metric_name in ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'accuracy']:
        mean = metrics_summary[metric_name]['mean']
        std = metrics_summary[metric_name]['std']
        median = metrics_summary[metric_name]['median']
        print(f"  {metric_name.upper():15s}: {mean:.4f} ± {std:.4f}  (median: {median:.4f})")

    # Save detailed metrics to CSV
    import csv
    csv_path = output_dir / 'metrics_per_slice.csv'

    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['patient_idx', 'slice_idx', 'dice', 'iou', 'sensitivity',
                      'specificity', 'accuracy', 'precision', 'TP', 'FP', 'TN', 'FN']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\n[+] Detailed metrics saved: {csv_path}")

    # Save summary statistics
    summary_path = output_dir / 'metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("PET/CT U-Net Evaluation Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Data: {args.data_dir}\n")
        f.write(f"Total slices: {len(all_metrics)}\n")
        f.write("\n")

        f.write("Metric Statistics:\n")
        f.write("-"*70 + "\n")
        for metric_name in ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'accuracy']:
            stats = metrics_summary[metric_name]
            f.write(f"\n{metric_name.upper()}:\n")
            f.write(f"  Mean:   {stats['mean']:.6f}\n")
            f.write(f"  Std:    {stats['std']:.6f}\n")
            f.write(f"  Median: {stats['median']:.6f}\n")
            f.write(f"  Min:    {stats['min']:.6f}\n")
            f.write(f"  Max:    {stats['max']:.6f}\n")

    print(f"[+] Summary statistics saved: {summary_path}")
    print(f"[+] Visualizations saved: {output_dir / 'visualizations'}")

    # Create distribution plots
    print("\n[*] Creating metric distribution plots...")
    create_distribution_plots(all_metrics, output_dir)

    print("\n" + "="*70)
    print("[+] Evaluation complete!")
    print("="*70)


def create_distribution_plots(all_metrics, output_dir):
    """Create box plots and histograms for metric distributions.

    Args:
        all_metrics: List of metric dictionaries
        output_dir: Output directory
    """
    metrics_to_plot = ['dice', 'iou', 'sensitivity', 'specificity', 'precision']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics_to_plot):
        values = [m[metric_name] for m in all_metrics]

        # Histogram
        axes[idx].hist(values, bins=20, edgecolor='black', alpha=0.7)
        axes[idx].axvline(np.mean(values), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(values):.3f}')
        axes[idx].axvline(np.median(values), color='green', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(values):.3f}')
        axes[idx].set_xlabel(metric_name.upper(), fontweight='bold')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{metric_name.upper()} Distribution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    # Remove unused subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Box plots
    fig, ax = plt.subplots(figsize=(12, 6))

    data = [[m[metric_name] for m in all_metrics] for metric_name in metrics_to_plot]
    labels = [m.upper() for m in metrics_to_plot]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Segmentation Metrics Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'metric_distributions.png'}")
    print(f"  Saved: {output_dir / 'metric_boxplots.png'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net on PET/CT data"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.keras file)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help="Directory containing patient data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image size (must match training)"
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=10,
        help="Number of slices to visualize"
    )

    args = parser.parse_args()

    # Verify model exists
    if not Path(args.model).exists():
        print(f"[!] Error: Model not found: {args.model}")
        return 1

    # Verify data directory exists
    if not Path(args.data_dir).exists():
        print(f"[!] Error: Data directory not found: {args.data_dir}")
        return 1

    try:
        evaluate(args)
        return 0

    except Exception as e:
        print(f"\n[!] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
