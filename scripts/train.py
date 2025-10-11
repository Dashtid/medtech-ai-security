#!/usr/bin/env python3
"""Training script for medical image segmentation models.

This script trains U-Net models for medical image segmentation with multi-expert
ensemble support, following the QUBIQ challenge methodology.

Usage:
    python scripts/train.py --config configs/brain_growth.yaml
    python scripts/train.py --config configs/kidney.yaml --expert 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from med_seg.data import load_dataset, create_data_generator, create_validation_generator
from med_seg.models import UNet, UNetDeep, UNetDeepSpatialDropout, UNetLSTM, UNetDeepLSTM
from med_seg.training import ModelTrainer, dice_loss, dice_coefficient, precision, recall
from med_seg.training.callbacks import get_callbacks
from med_seg.utils import load_config, plot_training_history


def get_model_class(architecture_name: str):
    """Get model class by name."""
    models = {
        'UNet': UNet,
        'UNetDeep': UNetDeep,
        'UNetDeepSpatialDropout': UNetDeepSpatialDropout,
        'UNetLSTM': UNetLSTM,
        'UNetDeepLSTM': UNetDeepLSTM,
    }

    if architecture_name not in models:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    return models[architecture_name]


def train_single_expert(config: dict, expert_id: int):
    """Train a model for a single expert annotation.

    Args:
        config: Configuration dictionary
        expert_id: Expert ID (1-indexed)
    """
    print(f"\n{'='*80}")
    print(f" Training Model for Expert {expert_id}")
    print(f"{'='*80}\n")

    # Dataset configuration
    dataset_config = config['dataset']
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    output_config = config['output']

    # Create mask identifier
    mask_id = f"seg{expert_id:02d}"
    if 'task_id' in dataset_config and dataset_config.get('num_tasks', 1) > 1:
        task_id = dataset_config['task_id']
        mask_id = f"task{task_id:02d}_{mask_id}"

    print(f"[i] Loading data for {dataset_config['name']} - Expert {expert_id}")
    print(f"[i] Mask identifier: {mask_id}")

    # Load training data
    print(f"[+] Loading training data from {data_config['train_dir']}")
    x_train, y_train = load_dataset(
        data_dir=data_config['train_dir'],
        mask_identifier=mask_id,
        image_size=data_config['image_size'],
        num_channels=dataset_config['num_channels'],
        use_intensity_windowing=data_config.get('use_intensity_windowing', False)
    )
    print(f"[+] Loaded {len(x_train)} training samples")

    # Load validation data
    print(f"[+] Loading validation data from {data_config['val_dir']}")
    x_val, y_val = load_dataset(
        data_dir=data_config['val_dir'],
        mask_identifier=mask_id,
        image_size=data_config['image_size'],
        num_channels=dataset_config['num_channels'],
        use_intensity_windowing=data_config.get('use_intensity_windowing', False)
    )
    print(f"[+] Loaded {len(x_val)} validation samples\n")

    # Create data generators
    aug_config = training_config.get('augmentation', {})
    if aug_config.get('enabled', True):
        print("[i] Creating data generators with augmentation")
        train_gen = create_data_generator(
            images=x_train,
            masks=y_train,
            batch_size=training_config['batch_size'],
            seed=expert_id,  # Different seed for each expert
            rotation_range=aug_config.get('rotation_range', 10.0),
            width_shift_range=aug_config.get('width_shift_range', 0.1),
            height_shift_range=aug_config.get('height_shift_range', 0.1),
            horizontal_flip=aug_config.get('horizontal_flip', True),
            vertical_flip=aug_config.get('vertical_flip', False),
            zoom_range=aug_config.get('zoom_range')
        )
    else:
        print("[i] Creating data generators without augmentation")
        train_gen = create_validation_generator(
            images=x_train,
            masks=y_train,
            batch_size=training_config['batch_size']
        )

    val_gen = create_validation_generator(
        images=x_val,
        masks=y_val,
        batch_size=training_config['batch_size']
    )

    # Build model
    print(f"[i] Building {model_config['architecture']} model")
    ModelClass = get_model_class(model_config['architecture'])

    model_builder = ModelClass(
        input_size=data_config['image_size'],
        input_channels=dataset_config['num_channels'],
        base_filters=model_config['base_filters'],
        use_batch_norm=model_config.get('use_batch_norm', True),
        use_dropout=model_config.get('use_dropout', True),
        dropout_rate=model_config.get('dropout_rate', 0.5),
        output_activation=model_config.get('output_activation', 'sigmoid')
    )

    model = model_builder.build()
    print(f"[+] Model built successfully\n")

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        loss_function=dice_loss,
        learning_rate=training_config['learning_rate'],
        metrics=[dice_coefficient, precision, recall]
    )

    # Set up callbacks
    expert_output_dir = Path(output_config['checkpoint_dir']) / f"expert_{expert_id:02d}"
    expert_log_dir = Path(output_config['log_dir']) / f"expert_{expert_id:02d}"

    callbacks = get_callbacks(
        checkpoint_dir=str(expert_output_dir),
        log_dir=str(expert_log_dir),
        monitor='val_loss',
        patience=training_config['callbacks']['early_stopping'].get('patience', 20),
        reduce_lr_patience=training_config['callbacks']['reduce_lr'].get('patience', 10)
    )

    # Train model
    print("[i] Starting training...")
    print(f"[i] Epochs: {training_config['epochs']}")
    print(f"[i] Batch size: {training_config['batch_size']}")
    print(f"[i] Learning rate: {training_config['learning_rate']}\n")

    history = trainer.train(
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=training_config['epochs'],
        steps_per_epoch=len(x_train) // training_config['batch_size'],
        validation_steps=len(x_val) // training_config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model_save_path = expert_output_dir / 'final_model.keras'
    trainer.save_model(str(model_save_path))

    # Plot and save training history
    plot_save_path = expert_output_dir / 'training_history.png'
    plot_training_history(history, save_path=str(plot_save_path), show=False)

    print(f"\n[+] Training complete for Expert {expert_id}")
    print(f"[+] Best model saved to: {expert_output_dir / 'best_model.keras'}")
    print(f"[+] Training history saved to: {plot_save_path}\n")

    # Clear session
    keras.backend.clear_session()

    return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train medical image segmentation models'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--expert',
        type=int,
        default=None,
        help='Train for specific expert only (1-indexed). If not specified, trains for all experts.'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU ID to use'
    )

    args = parser.parse_args()

    # Set GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load configuration
    print(f"[i] Loading configuration from {args.config}")
    config = load_config(args.config)

    # Determine which experts to train
    num_experts = config['dataset']['num_experts']

    if args.expert is not None:
        if args.expert < 1 or args.expert > num_experts:
            print(f"[!] Error: Expert ID must be between 1 and {num_experts}")
            sys.exit(1)
        expert_ids = [args.expert]
        print(f"[i] Training for Expert {args.expert} only\n")
    else:
        expert_ids = range(1, num_experts + 1)
        print(f"[i] Training for all {num_experts} experts\n")

    # Train for each expert
    for expert_id in expert_ids:
        try:
            train_single_expert(config, expert_id)
        except Exception as e:
            print(f"[!] Error training expert {expert_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(" Training Complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
