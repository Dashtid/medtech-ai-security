#!/usr/bin/env python
"""Train multi-task U-Net on PET/CT data for tumor segmentation AND survival prediction.

This script trains a multi-task model that simultaneously:
1. Segments tumors from PET/CT images
2. Predicts patient survival from imaging features
3. Quantifies prediction uncertainty via Monte Carlo Dropout

Usage:
    python scripts/train_multitask.py --data-dir data/synthetic_v2_survival --epochs 30
"""

import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorflow import keras

from med_seg.models.multitask_unet import MultiTaskUNet
from med_seg.data import PETCTPreprocessor
from med_seg.data.survival_generator import create_survival_generators
from med_seg.training.losses import focal_tversky_loss, dice_coefficient
from med_seg.training.survival_losses import cox_ph_loss, concordance_index
from med_seg.training.metrics import iou_score


def create_callbacks(output_dir):
    """Create training callbacks.

    Args:
        output_dir: Directory to save checkpoints

    Returns:
        List of Keras callbacks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save best model based on validation segmentation DICE
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_segmentation_dice_coefficient",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Early stopping based on combined validation loss
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1
        ),
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
        # CSV logger
        keras.callbacks.CSVLogger(
            filename=str(output_dir / "training_log.csv"), separator=",", append=False
        ),
    ]

    return callbacks


def train(args):
    """Main training function.

    Args:
        args: Command-line arguments
    """
    print("\n[+] Multi-Task PET/CT U-Net Training")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Segmentation weight: {args.seg_weight}")
    print(f"Survival weight: {args.surv_weight}")
    print()

    # Create preprocessor
    print("[1/7] Creating preprocessor...")
    preprocessor = PETCTPreprocessor(
        target_size=(args.image_size, args.image_size),
        ct_window_center=0,
        ct_window_width=400,
        suv_max=15,
    )

    # Create data generators
    print("\n[2/7] Creating data generators...")
    train_gen, val_gen = create_survival_generators(
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        train_fraction=0.7,
        augment_train=True,
    )

    print(f"  Training batches: {len(train_gen)}")
    print(f"  Validation batches: {len(val_gen)}")

    # Build multi-task model
    print("\n[3/7] Building Multi-Task U-Net...")
    model_builder = MultiTaskUNet(
        input_size=args.image_size,
        input_channels=2,  # PET + CT
        num_classes=1,  # Binary segmentation
        base_filters=args.base_filters,
        depth=args.depth,
        use_batch_norm=True,
        dropout_rate=args.dropout,
        survival_hidden_units=(256, 128, 64),
    )

    model = model_builder.build()

    # Print model summary
    print("\n  Model architecture:")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    # Compile model with multi-task losses
    print("\n[4/7] Compiling model...")

    # Segmentation loss (Focal Tversky)
    seg_loss_fn = focal_tversky_loss(alpha=0.3, beta=0.7, gamma=0.75)

    # Survival loss (Cox proportional hazards)
    surv_loss_fn = cox_ph_loss

    # Loss weights
    losses = {"segmentation": seg_loss_fn, "survival": surv_loss_fn}

    loss_weights = {"segmentation": args.seg_weight, "survival": args.surv_weight}

    # Metrics for each output
    metrics = {
        "segmentation": [dice_coefficient, iou_score, "accuracy"],
        "survival": [concordance_index],
    }

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    print(f"  Optimizer: Adam (lr={args.lr})")
    print(f"  Segmentation loss: Focal Tversky (weight={args.seg_weight})")
    print(f"  Survival loss: Cox PH (weight={args.surv_weight})")
    print("  Segmentation metrics: DICE, IoU, Accuracy")
    print("  Survival metrics: C-index")

    # Create callbacks
    print("\n[5/7] Setting up callbacks...")
    callbacks = create_callbacks(args.output)
    print(f"  Callbacks: {len(callbacks)} configured")

    # Train model
    print("\n[6/7] Training multi-task model...")
    print("=" * 70)

    history = model.fit(
        train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks, verbose=1
    )

    # Save final model
    print("\n" + "=" * 70)
    print("[7/7] Saving models...")

    final_model_path = Path(args.output) / "final_model.keras"
    model.save(final_model_path)
    print(f"[+] Final model saved: {final_model_path}")

    best_model_path = Path(args.output) / "best_model.keras"
    print(f"[+] Best model saved: {best_model_path}")

    # Print final metrics
    print("\n[*] Final training metrics:")
    for key, values in history.history.items():
        if not key.startswith("val_"):
            print(f"    {key}: {values[-1]:.4f}")

    print("\n[*] Final validation metrics:")
    for key, values in history.history.items():
        if key.startswith("val_"):
            print(f"    {key}: {values[-1]:.4f}")

    print("\n[*] View training progress:")
    print(f"    tensorboard --logdir {Path(args.output) / 'logs'}")

    print("\n[+] Multi-task training complete!")
    print("\n[*] Next steps:")
    print("    1. Evaluate uncertainty: python scripts/inference_with_uncertainty.py")
    print("    2. Optimize model: python scripts/optimize_model.py")
    print("    3. Benchmark: python scripts/benchmark_inference.py")

    return model, history


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Multi-Task U-Net (Segmentation + Survival)")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing patient data and survival_data.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/multitask_unet",
        help="Output directory for models and logs",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size (square)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--base-filters", type=int, default=64, help="Number of base filters in U-Net"
    )
    parser.add_argument("--depth", type=int, default=4, help="Depth of U-Net")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for MC Dropout")
    parser.add_argument(
        "--seg-weight", type=float, default=0.6, help="Weight for segmentation loss (0-1)"
    )
    parser.add_argument(
        "--surv-weight", type=float, default=0.4, help="Weight for survival loss (0-1)"
    )

    args = parser.parse_args()

    # Verify data directory and survival data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[!] Error: Data directory not found: {data_dir}")
        return 1

    survival_file = data_dir / "survival_data.json"
    if not survival_file.exists():
        print(f"[!] Error: Survival data not found: {survival_file}")
        print("[*] Generate survival data first:")
        print("    python scripts/generate_survival_data.py --data-dir data/synthetic_v2")
        return 1

    try:
        model, history = train(args)
        print("\n[+] Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
        return 1

    except Exception as e:
        print(f"\n[!] Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
