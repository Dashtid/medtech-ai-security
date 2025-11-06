#!/usr/bin/env python3
"""Evaluation script for trained medical image segmentation models.

This script evaluates trained models using multi-expert ensemble methodology
and generates comprehensive metrics and visualizations.

Usage:
    python scripts/evaluate.py --config configs/brain_growth.yaml --model-dir models/brain-growth
    python scripts/evaluate.py --config configs/kidney.yaml --model-dir models/kidney --output results/
"""

import argparse
import sys
from pathlib import Path
import json

import numpy as np
from tensorflow import keras

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from med_seg.data import load_dataset
from med_seg.evaluation import multi_expert_evaluation, evaluate_segmentation
from med_seg.training import dice_loss, dice_coefficient, precision, recall
from med_seg.utils import load_config, plot_segmentation_results


def load_expert_models(model_dir: Path, num_experts: int):
    """Load all trained expert models.

    Args:
        model_dir: Directory containing expert model subdirectories
        num_experts: Number of experts to load

    Returns:
        List of loaded Keras models
    """
    models = []
    custom_objects = {
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient,
        "precision": precision,
        "recall": recall,
    }

    for expert_id in range(1, num_experts + 1):
        expert_dir = model_dir / f"expert_{expert_id:02d}"
        model_path = expert_dir / "best_model.keras"

        if not model_path.exists():
            print(f"[!] Warning: Model not found for expert {expert_id} at {model_path}")
            print("[!] Trying final_model.keras instead...")
            model_path = expert_dir / "final_model.keras"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found for expert {expert_id}")

        print(f"[+] Loading model for expert {expert_id} from {model_path}")
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        models.append(model)

    return models


def evaluate_models(config: dict, model_dir: str, output_dir: str = None):
    """Evaluate all trained models.

    Args:
        config: Configuration dictionary
        model_dir: Directory containing trained models
        output_dir: Directory to save evaluation results
    """
    print(f"\n{'='*80}")
    print(f" Evaluating Models: {config['dataset']['name']}")
    print(f"{'='*80}\n")

    # Configuration
    dataset_config = config["dataset"]
    data_config = config["data"]
    eval_config = config.get("evaluation", {})

    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = Path(config["output"]["results_dir"])
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_experts = dataset_config["num_experts"]
    task_id = dataset_config.get("task_id", 1)

    # Load test data for all experts
    print(f"[i] Loading test data for {num_experts} experts...")
    expert_predictions = []
    expert_ground_truths = []

    # Load models
    print(f"[i] Loading trained models from {model_dir}")
    models = load_expert_models(model_dir, num_experts)

    # Load test data and generate predictions for each expert
    for expert_id in range(1, num_experts + 1):
        # Create mask identifier
        mask_id = f"seg{expert_id:02d}"
        if dataset_config.get("num_tasks", 1) > 1:
            mask_id = f"task{task_id:02d}_{mask_id}"

        print(f"\n[i] Processing expert {expert_id}")
        print(f"[i] Mask identifier: {mask_id}")

        # Load test data
        x_test, y_test = load_dataset(
            data_dir=data_config["test_dir"],
            mask_identifier=mask_id,
            image_size=data_config["image_size"],
            num_channels=dataset_config["num_channels"],
            use_intensity_windowing=data_config.get("use_intensity_windowing", False),
        )

        print(f"[+] Loaded {len(x_test)} test samples")

        # Generate predictions
        print("[i] Generating predictions...")
        predictions = models[expert_id - 1].predict(x_test, verbose=0)

        expert_predictions.append(predictions)
        expert_ground_truths.append(y_test)

        print(f"[+] Predictions generated for expert {expert_id}")

    # Perform multi-expert ensemble evaluation
    print(f"\n{'='*80}")
    print(" Multi-Expert Ensemble Evaluation")
    print(f"{'='*80}\n")

    thresholds = eval_config.get("thresholds", [i / 10 for i in range(1, 10)])

    results = multi_expert_evaluation(
        expert_predictions=expert_predictions,
        expert_ground_truths=expert_ground_truths,
        task_id=task_id,
        thresholds=thresholds,
    )

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {results_file}")

    # Save DICE matrix
    dice_matrix_file = output_dir / "dice_matrix.npy"
    np.save(dice_matrix_file, results["dice_matrix"])
    print(f"[+] DICE matrix saved to {dice_matrix_file}")

    # Plot threshold analysis
    print("\n[i] Generating visualizations...")
    plot_save_path = output_dir / "dice_threshold_analysis.png"

    from med_seg.utils.visualization import plot_dice_threshold_analysis

    threshold_list = list(results["threshold_averages"].keys())
    dice_list = list(results["threshold_averages"].values())

    plot_dice_threshold_analysis(
        thresholds=threshold_list, dice_scores=dice_list, save_path=str(plot_save_path), show=False
    )
    print(f"[+] Threshold analysis plot saved to {plot_save_path}")

    # Generate sample predictions visualization
    print("\n[i] Generating sample predictions visualization...")

    # Use first expert's test data and ensemble predictions
    from med_seg.evaluation import ensemble_predictions

    ensembled_preds = ensemble_predictions(expert_predictions, method="mean")
    ensembled_gt = ensemble_predictions(expert_ground_truths, method="mean")

    # Plot first 5 samples
    samples_plot = output_dir / "sample_predictions.png"
    plot_segmentation_results(
        images=x_test,
        ground_truths=ensembled_gt,
        predictions=ensembled_preds,
        num_samples=min(5, len(x_test)),
        threshold=results["best_threshold"],
        save_path=str(samples_plot),
        show=False,
    )
    print(f"[+] Sample predictions saved to {samples_plot}")

    # Per-expert individual metrics at best threshold
    print(f"\n{'='*80}")
    print(" Per-Expert Metrics at Best Threshold")
    print(f"{'='*80}\n")

    best_threshold = results["best_threshold"]
    per_expert_results = []

    for expert_id in range(num_experts):
        metrics = evaluate_segmentation(
            predictions=expert_predictions[expert_id],
            ground_truth=expert_ground_truths[expert_id],
            threshold=best_threshold,
        )
        per_expert_results.append(metrics)

        print(f"Expert {expert_id + 1}:")
        print(f"  DICE:        {metrics['dice']:.4f}")
        print(f"  IoU:         {metrics['iou']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print()

    # Save per-expert results
    per_expert_file = output_dir / "per_expert_metrics.json"
    with open(per_expert_file, "w") as f:
        json.dump(per_expert_results, f, indent=2)
    print(f"[+] Per-expert metrics saved to {per_expert_file}")

    # Print summary
    print(f"\n{'='*80}")
    print(" Evaluation Summary")
    print(f"{'='*80}\n")
    print(f"Dataset:              {dataset_config['name']}")
    print(f"Number of experts:    {num_experts}")
    print(f"Test samples:         {len(x_test)}")
    print(f"Best threshold:       {results['best_threshold']:.2f}")
    print(f"Best DICE:            {results['best_dice']:.4f}")
    print(f"Average DICE:         {results['average_dice']:.4f}")
    print(f"\nResults saved to:     {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained medical image segmentation models"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained models (with expert_XX subdirectories)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: from config)",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")

    args = parser.parse_args()

    # Set GPU
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load configuration
    print(f"[i] Loading configuration from {args.config}")
    config = load_config(args.config)

    # Evaluate models
    try:
        evaluate_models(config, args.model_dir, args.output)
    except Exception as e:
        print(f"[!] Error during evaluation: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("[+] Evaluation complete!")


if __name__ == "__main__":
    main()
