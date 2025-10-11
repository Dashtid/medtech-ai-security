"""Training callbacks for model training."""

from typing import Optional, List
from pathlib import Path

from tensorflow import keras


def get_callbacks(
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    monitor: str = 'val_loss',
    patience: int = 20,
    reduce_lr_patience: int = 10,
    min_lr: float = 1e-7
) -> List[keras.callbacks.Callback]:
    """Create standard set of training callbacks.

    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        monitor: Metric to monitor for callbacks
        patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        min_lr: Minimum learning rate

    Returns:
        List of Keras callbacks
    """
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=str(Path(checkpoint_dir) / 'best_model.keras'),
            monitor=monitor,
            mode='min' if 'loss' in monitor else 'max',
            save_best_only=True,
            verbose=1
        ),

        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1,
            restore_best_weights=True
        ),

        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=reduce_lr_patience,
            mode='min' if 'loss' in monitor else 'max',
            min_lr=min_lr,
            verbose=1
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),

        # CSV logger
        keras.callbacks.CSVLogger(
            filename=str(Path(log_dir) / 'training_log.csv'),
            separator=',',
            append=False
        )
    ]

    return callbacks
