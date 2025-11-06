#!/usr/bin/env python3
"""
Professional demo script for multi-task PET/CT tumor analysis.

This script provides a user-friendly CLI interface for running inference
on the trained multi-task model, showing both tumor segmentation and
survival prediction with uncertainty quantification.

Usage:
    python scripts/demo.py --patient patient_001
    python scripts/demo.py --patient patient_005 --n-samples 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.med_seg.data.petct_loader import PETCTLoader
from src.med_seg.data.petct_preprocessor import PETCTPreprocessor


def print_header():
    """Print professional header."""
    print("\n" + "=" * 70)
    print("  MULTI-TASK PET/CT TUMOR ANALYSIS SYSTEM")
    print("  Tumor Segmentation + Survival Prediction + Uncertainty")
    print("=" * 70 + "\n")


def print_section(title: str):
    """Print section header."""
    print(f"\n[{title}]")
    print("-" * 70)


def print_metric(name: str, value, unit: str = ""):
    """Print metric with formatting."""
    if isinstance(value, float):
        print(f"  {name:.<50} {value:.4f} {unit}")
    else:
        print(f"  {name:.<50} {value} {unit}")


def load_patient_data(data_dir: str, patient_id: str, preprocessor):
    """Load and preprocess patient data.

    Args:
        data_dir: Directory containing patient data
        patient_id: Patient ID (e.g., 'patient_001')
        preprocessor: PETCTPreprocessor instance

    Returns:
        Tuple of (preprocessed_slices, tumor_slice_indices)
    """
    loader = PETCTLoader(data_dir)

    # Find patient
    patient_idx = None
    for idx, pdir in enumerate(loader.patient_dirs):
        if pdir.name == patient_id:
            patient_idx = idx
            break

    if patient_idx is None:
        raise ValueError(f"Patient {patient_id} not found in {data_dir}")

    # Load all axial slices
    ct_batch, suv_batch, seg_batch = loader.extract_2d_slices(
        patient_idx, axis="axial", slice_indices=None
    )

    # Find tumor-containing slices
    tumor_slices = []
    for i, seg_slice in enumerate(seg_batch):
        if seg_slice.sum() > 0:
            tumor_slices.append(i)

    if not tumor_slices:
        raise ValueError(f"No tumor found in patient {patient_id}")

    # Preprocess all slices
    inputs, targets = preprocessor.preprocess_batch_2d(ct_batch, suv_batch, seg_batch)

    return inputs, targets, tumor_slices


def run_mc_dropout_inference(model, inputs, n_samples: int = 30):
    """Run Monte Carlo Dropout inference for uncertainty estimation.

    Args:
        model: Trained Keras model
        inputs: Input images (batch, H, W, C)
        n_samples: Number of MC samples

    Returns:
        Tuple of (seg_mean, seg_std, surv_mean, surv_std)
    """
    seg_predictions = []
    surv_predictions = []

    for _ in range(n_samples):
        seg_pred, surv_pred = model(inputs, training=True)
        seg_predictions.append(seg_pred.numpy())
        surv_predictions.append(surv_pred.numpy())

    seg_predictions = np.array(seg_predictions)  # (n_samples, batch, H, W, 1)
    surv_predictions = np.array(surv_predictions)  # (n_samples, batch, 1)

    seg_mean = seg_predictions.mean(axis=0)
    seg_std = seg_predictions.std(axis=0)
    surv_mean = surv_predictions.mean(axis=0)
    surv_std = surv_predictions.std(axis=0)

    return seg_mean, seg_std, surv_mean, surv_std


def analyze_segmentation(seg_pred, seg_gt, seg_std, threshold: float = 0.5):
    """Analyze segmentation results.

    Args:
        seg_pred: Predicted segmentation probabilities
        seg_gt: Ground truth segmentation
        seg_std: Uncertainty (standard deviation)
        threshold: Threshold for binarization

    Returns:
        Dictionary of metrics
    """
    seg_binary = (seg_pred > threshold).astype(np.float32)
    gt_binary = (seg_gt > threshold).astype(np.float32)

    # Compute metrics
    intersection = (seg_binary * gt_binary).sum()
    union = seg_binary.sum() + gt_binary.sum()

    dice = 2.0 * intersection / (union + 1e-7)
    iou = intersection / (union - intersection + 1e-7)

    # Uncertainty stats
    mean_uncertainty = seg_std.mean()
    max_uncertainty = seg_std.max()

    # Uncertainty in tumor region
    tumor_mask = gt_binary > 0.5
    if tumor_mask.sum() > 0:
        tumor_uncertainty = seg_std[tumor_mask].mean()
    else:
        tumor_uncertainty = 0.0

    return {
        "dice": dice,
        "iou": iou,
        "mean_uncertainty": mean_uncertainty,
        "max_uncertainty": max_uncertainty,
        "tumor_uncertainty": tumor_uncertainty,
        "tumor_pixels": int(gt_binary.sum()),
        "predicted_pixels": int(seg_binary.sum()),
    }


def analyze_survival(surv_pred, surv_std):
    """Analyze survival prediction.

    Args:
        surv_pred: Predicted risk score
        surv_std: Uncertainty (standard deviation)

    Returns:
        Dictionary of results
    """
    # Risk score interpretation
    risk_score = surv_pred.item()
    uncertainty = surv_std.item()

    # Confidence interval (95%)
    ci_lower = risk_score - 1.96 * uncertainty
    ci_upper = risk_score + 1.96 * uncertainty

    # Risk category
    if risk_score < -0.5:
        risk_category = "LOW RISK"
    elif risk_score < 0.5:
        risk_category = "MODERATE RISK"
    else:
        risk_category = "HIGH RISK"

    return {
        "risk_score": risk_score,
        "uncertainty": uncertainty,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "risk_category": risk_category,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task PET/CT tumor analysis demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/multitask_unet/best_model.keras",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic_v2_survival",
        help="Directory containing patient data",
    )
    parser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID (e.g., patient_001)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of MC Dropout samples for uncertainty",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold",
    )
    args = parser.parse_args()

    print_header()

    # Load model
    print_section("Loading Model")
    print(f"  Model path: {args.model}")
    model = keras.models.load_model(args.model, compile=False)
    print("  [OK] Model loaded successfully")
    print(f"  Total parameters: {model.count_params():,}")

    # Load patient data
    print_section("Loading Patient Data")
    print(f"  Patient ID: {args.patient}")
    print(f"  Data directory: {args.data_dir}")

    preprocessor = PETCTPreprocessor(target_size=(256, 256), normalize=True)
    inputs, targets, tumor_slices = load_patient_data(
        args.data_dir, args.patient, preprocessor
    )

    print(f"  [OK] Loaded {inputs.shape[0]} slices")
    print(f"  [OK] Found {len(tumor_slices)} tumor-containing slices: {tumor_slices}")

    # Select middle tumor slice for demo
    mid_idx = tumor_slices[len(tumor_slices) // 2]
    demo_input = inputs[mid_idx : mid_idx + 1]  # (1, H, W, C)
    demo_seg_gt = targets[mid_idx : mid_idx + 1]  # (1, H, W, 1)

    print(f"  [OK] Using slice {mid_idx} for demo")

    # Run inference with uncertainty
    print_section("Running Monte Carlo Dropout Inference")
    print(f"  MC samples: {args.n_samples}")
    print("  [Processing...]")

    seg_mean, seg_std, surv_mean, surv_std = run_mc_dropout_inference(
        model, demo_input, args.n_samples
    )

    print("  [OK] Inference complete")

    # Analyze segmentation
    print_section("Tumor Segmentation Results")
    seg_metrics = analyze_segmentation(
        seg_mean[0], demo_seg_gt[0], seg_std[0], args.threshold
    )

    print_metric("DICE Score", seg_metrics["dice"])
    print_metric("IoU Score", seg_metrics["iou"])
    print_metric("Ground Truth Tumor Pixels", seg_metrics["tumor_pixels"], "px")
    print_metric("Predicted Tumor Pixels", seg_metrics["predicted_pixels"], "px")
    print_metric("Mean Uncertainty", seg_metrics["mean_uncertainty"])
    print_metric("Max Uncertainty", seg_metrics["max_uncertainty"])
    print_metric("Tumor Region Uncertainty", seg_metrics["tumor_uncertainty"])

    # Analyze survival
    print_section("Survival Prediction Results")
    surv_results = analyze_survival(surv_mean[0], surv_std[0])

    print_metric("Risk Score", surv_results["risk_score"])
    print_metric("Risk Category", surv_results["risk_category"])
    print_metric("Uncertainty (Std Dev)", surv_results["uncertainty"])
    print_metric("95% CI Lower Bound", surv_results["ci_lower"])
    print_metric("95% CI Upper Bound", surv_results["ci_upper"])

    # Risk interpretation
    print("\n  Risk Interpretation:")
    print(f"    - Risk score range: [{surv_results['ci_lower']:.4f}, "
          f"{surv_results['ci_upper']:.4f}]")
    print(f"    - Prediction confidence: {1.0 / (1.0 + surv_results['uncertainty']):.2%}")

    if surv_results["uncertainty"] > 0.5:
        print("    [!] HIGH UNCERTAINTY - Consider additional testing")
    else:
        print("    [OK] Acceptable uncertainty level")

    # Summary
    print_section("Summary")
    print(f"  Patient: {args.patient}")
    print(f"  Slice analyzed: {mid_idx}")
    print(f"  Segmentation quality: DICE = {seg_metrics['dice']:.4f}")
    print(f"  Survival risk: {surv_results['risk_category']} "
          f"(score = {surv_results['risk_score']:.4f})")
    print(f"  Uncertainty: Segmentation = {seg_metrics['mean_uncertainty']:.4f}, "
          f"Survival = {surv_results['uncertainty']:.4f}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
