#!/usr/bin/env python
"""Inference with uncertainty quantification using Monte Carlo Dropout.

This script demonstrates how to use the trained multi-task model with
uncertainty quantification for reliable medical AI predictions.

Usage:
    python scripts/inference_with_uncertainty.py \
        --model models/multitask_unet/best_model.keras \
        --data-dir data/synthetic_v2_survival \
        --patient patient_001 \
        --n-samples 30 \
        --output results/uncertainty
"""

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from med_seg.data import PETCTLoader, PETCTPreprocessor


def mc_dropout_inference(model: keras.Model,
                        x: np.ndarray,
                        n_samples: int = 30) -> dict:
    """Perform Monte Carlo Dropout inference for uncertainty estimation.

    Args:
        model: Trained model with MC Dropout layers
        x: Input image (1, H, W, C)
        n_samples: Number of forward passes

    Returns:
        Dictionary with predictions and uncertainty estimates
    """
    # Collect predictions from multiple forward passes
    seg_predictions = []
    surv_predictions = []

    for _ in range(n_samples):
        pred = model(x, training=True)  # Keep dropout active
        seg_predictions.append(pred['segmentation'].numpy())
        surv_predictions.append(pred['survival'].numpy())

    seg_predictions = np.array(seg_predictions)  # (n_samples, 1, H, W, 1)
    surv_predictions = np.array(surv_predictions)  # (n_samples, 1, 1)

    # Calculate mean and uncertainty
    results = {
        # Segmentation
        'seg_mean': np.mean(seg_predictions, axis=0)[0],  # (H, W, 1)
        'seg_std': np.std(seg_predictions, axis=0)[0],    # (H, W, 1)
        'seg_samples': seg_predictions[:, 0],             # (n_samples, H, W, 1)

        # Survival
        'surv_mean': np.mean(surv_predictions),
        'surv_std': np.std(surv_predictions),
        'surv_samples': surv_predictions[:, 0, 0],        # (n_samples,)
    }

    return results


def visualize_uncertainty(pet: np.ndarray,
                         ct: np.ndarray,
                         ground_truth: np.ndarray,
                         results: dict,
                         save_path: Path):
    """Create uncertainty visualization.

    Args:
        pet: PET image
        ct: CT image
        ground_truth: Ground truth segmentation
        results: Results from mc_dropout_inference
        save_path: Where to save visualization
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Inputs and predictions
    axes[0, 0].imshow(ct, cmap='gray')
    axes[0, 0].set_title('CT Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pet, cmap='hot')
    axes[0, 1].set_title('PET (SUV)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(ct, cmap='gray')
    axes[0, 2].imshow(ground_truth, cmap='Reds', alpha=0.5)
    axes[0, 2].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(ct, cmap='gray')
    axes[0, 3].imshow(results['seg_mean'][..., 0], cmap='Reds', alpha=0.5)
    axes[0, 3].set_title('Mean Prediction', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')

    # Row 2: Uncertainty analysis
    axes[1, 0].imshow(results['seg_mean'][..., 0], cmap='Reds')
    axes[1, 0].set_title(f"Probability Map", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    im = axes[1, 1].imshow(results['seg_std'][..., 0], cmap='viridis')
    axes[1, 1].set_title(f"Uncertainty (Std Dev)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # Binary prediction with confidence threshold
    binary = (results['seg_mean'][..., 0] > 0.5).astype(float)
    low_confidence = results['seg_std'][..., 0] > 0.1
    binary[low_confidence] = 0.5  # Gray out uncertain regions

    axes[1, 2].imshow(ct, cmap='gray')
    axes[1, 2].imshow(binary, cmap='RdYlGn', alpha=0.5, vmin=0, vmax=1)
    axes[1, 2].set_title('High-Confidence Prediction', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    # Survival prediction distribution
    axes[1, 3].hist(results['surv_samples'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 3].axvline(results['surv_mean'], color='red', linestyle='--',
                      linewidth=2, label=f"Mean: {results['surv_mean']:.3f}")
    axes[1, 3].set_xlabel('Risk Score', fontsize=11)
    axes[1, 3].set_ylabel('Frequency', fontsize=11)
    axes[1, 3].set_title(f"Survival Risk Distribution\n(Std: {results['surv_std']:.3f})",
                        fontsize=12, fontweight='bold')
    axes[1, 3].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[+] Saved visualization: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inference with uncertainty quantification")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--patient", type=str, default="patient_001", help="Patient ID")
    parser.add_argument("--slice-idx", type=int, default=None, help="Slice index (random if not specified)")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of MC samples")
    parser.add_argument("--output", type=str, default="results/uncertainty", help="Output directory")

    args = parser.parse_args()

    print("\n[+] Uncertainty Quantification Demo")
    print("="*70)

    # Load model
    print(f"[*] Loading model: {args.model}")
    model = keras.models.load_model(args.model, compile=False)

    # Load data
    print(f"[*] Loading data: {args.data_dir}")
    loader = PETCTLoader(args.data_dir)
    preprocessor = PETCTPreprocessor(target_size=(256, 256))

    # Get slice
    if args.slice_idx is None:
        # Find tumor slice
        patient_data = loader.patients[args.patient]
        tumor_slices = np.where(patient_data['tumor_slices'])[0]
        args.slice_idx = int(np.random.choice(tumor_slices))

    print(f"[*] Patient: {args.patient}, Slice: {args.slice_idx}")

    pet, ct, seg = loader.get_slice(args.patient, args.slice_idx)
    x, y = preprocessor.preprocess_pair(pet, ct, seg)

    # Prepare input
    x_input = np.expand_dims(x, axis=0)

    # MC Dropout inference
    print(f"[*] Running MC Dropout inference ({args.n_samples} samples)...")
    results = mc_dropout_inference(model, x_input, args.n_samples)

    # Print results
    print(f"\n[*] Segmentation Results:")
    print(f"    Mean DICE: {2 * np.sum(results['seg_mean'][..., 0] * y) / (np.sum(results['seg_mean'][..., 0]) + np.sum(y)):.4f}")
    print(f"    Mean uncertainty: {np.mean(results['seg_std']):.4f}")
    print(f"    Max uncertainty: {np.max(results['seg_std']):.4f}")

    print(f"\n[*] Survival Prediction:")
    print(f"    Risk score: {results['surv_mean']:.4f} ± {results['surv_std']:.4f}")
    print(f"    95% CI: [{results['surv_mean'] - 1.96*results['surv_std']:.4f}, {results['surv_mean'] + 1.96*results['surv_std']:.4f}]")

    # Create visualization
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f'{args.patient}_slice{args.slice_idx}_uncertainty.png'
    visualize_uncertainty(pet, ct, y, results, save_path)

    print(f"\n[+] Uncertainty quantification complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
