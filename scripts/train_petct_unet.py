#!/usr/bin/env python
"""Train 2D U-Net on PET/CT data for tumor segmentation.

This script trains a U-Net model on PET/CT data with multi-modal inputs
(PET + CT channels) for binary tumor segmentation.

Usage:
    python scripts/train_petct_unet.py --data-dir data/synthetic --epochs 50
"""

import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import tensorflow as tf
from tensorflow import keras

from med_seg.models import UNet
from med_seg.data import PETCTLoader, PETCTPreprocessor
from med_seg.data.petct_generator import PETCTDataGenerator
from med_seg.training.losses import combined_loss, dice_coefficient, focal_tversky_loss
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
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),

        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / 'logs'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),

        # CSV logger
        keras.callbacks.CSVLogger(
            filename=str(output_dir / 'training_log.csv'),
            separator=',',
            append=False
        )
    ]

    return callbacks


def train(args):
    """Main training function.

    Args:
        args: Command-line arguments
    """
    print("\n[+] PET/CT U-Net Training")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Learning rate: {args.lr}")
    print()

    # Create data loader
    print("[1/6] Loading data...")
    loader = PETCTLoader(args.data_dir)
    print(f"  Found {len(loader)} patients")

    # Get dataset statistics
    stats = loader.get_statistics()
    print(f"  CT range: [{stats['ct_range']['min']:.1f}, {stats['ct_range']['max']:.1f}] HU")
    print(f"  PET range: [{stats['pet_range']['min']:.2f}, {stats['pet_range']['max']:.2f}]")
    print(f"  Avg tumor prevalence: {stats['avg_tumor_prevalence']:.3f}%")

    # Create preprocessor
    preprocessor = PETCTPreprocessor(
        target_size=(args.image_size, args.image_size),
        ct_window_center=0,
        ct_window_width=400,
        suv_max=15
    )
    print(f"  Preprocessor: {preprocessor}")

    # Create data generators
    print("\n[2/6] Creating data generators...")
    train_gen = PETCTDataGenerator(
        loader=loader,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        tumor_only=True
    )

    # Use same data for validation (for synthetic data demo)
    # In production, you'd split into train/val sets
    val_gen = PETCTDataGenerator(
        loader=loader,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        tumor_only=True
    )

    print(f"  Training batches: {len(train_gen)}")
    print(f"  Validation batches: {len(val_gen)}")
    print(f"  Training slices: {len(train_gen.slice_index)}")

    # Compute class weights
    print("\n  Computing class weights...")
    class_weights = train_gen.get_class_weights()
    print(f"  Background weight: {class_weights[0]:.2f}")
    print(f"  Tumor weight: {class_weights[1]:.2f}")

    # Build model
    print("\n[3/6] Building U-Net model...")
    model_builder = UNet(
        input_size=args.image_size,
        input_channels=2,  # PET + CT
        num_classes=1,     # Binary segmentation
        base_filters=args.base_filters,
        depth=args.depth,
        use_batch_norm=True,
        use_dropout=args.dropout > 0,
        dropout_rate=args.dropout
    )

    model = model_builder.build()

    # Print model summary
    print(f"\n  Model architecture:")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    # Compile model
    print("\n[4/6] Compiling model...")

    # Use Focal Tversky loss for better handling of severe class imbalance
    # alpha=0.3, beta=0.7 prioritizes recall (sensitivity) over precision
    # gamma=0.75 focuses on hard examples
    loss_fn = focal_tversky_loss(alpha=0.3, beta=0.7, gamma=0.75)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=[
            dice_coefficient,
            iou_score,
            'accuracy'
        ]
    )

    print(f"  Optimizer: Adam (lr={args.lr})")
    print(f"  Loss: Focal Tversky (alpha=0.3, beta=0.7, gamma=0.75)")
    print(f"  Metrics: DICE, IoU, Accuracy")

    # Create callbacks
    print("\n[5/6] Setting up callbacks...")
    callbacks = create_callbacks(args.output)
    print(f"  Callbacks: {len(callbacks)} configured")
    print(f"    - ModelCheckpoint (save best)")
    print(f"    - EarlyStopping (patience=10)")
    print(f"    - ReduceLROnPlateau (patience=5)")
    print(f"    - TensorBoard logging")
    print(f"    - CSV logging")

    # Train model
    print("\n[6/6] Training model...")
    print("="*70)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    print("\n" + "="*70)
    print("[+] Training complete!")

    final_model_path = Path(args.output) / 'final_model.keras'
    model.save(final_model_path)
    print(f"[+] Final model saved: {final_model_path}")

    best_model_path = Path(args.output) / 'best_model.keras'
    print(f"[+] Best model saved: {best_model_path}")

    # Print final metrics
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_dice = history.history['dice_coefficient'][-1]
    final_val_dice = history.history['val_dice_coefficient'][-1]

    print(f"\n[*] Final metrics:")
    print(f"    Training loss: {final_loss:.4f}")
    print(f"    Validation loss: {final_val_loss:.4f}")
    print(f"    Training DICE: {final_dice:.4f}")
    print(f"    Validation DICE: {final_val_dice:.4f}")

    print(f"\n[*] View training progress:")
    print(f"    tensorboard --logdir {Path(args.output) / 'logs'}")

    return model, history


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train U-Net on PET/CT data"
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
        default="models/petct_unet",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image size (will be square)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--base-filters",
        type=int,
        default=64,
        help="Number of base filters in U-Net"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Depth of U-Net (number of downsampling levels)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (0 to disable)"
    )

    args = parser.parse_args()

    # Verify data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[!] Error: Data directory not found: {data_dir}")
        print(f"[*] Generate synthetic data first:")
        print(f"    python scripts/create_synthetic_petct.py --output {data_dir} --num-patients 3")
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
