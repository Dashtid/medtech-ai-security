"""Model training orchestration."""

from typing import Dict, Optional, Any, Iterator, Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from med_seg.training.losses import dice_loss, dice_coefficient
from med_seg.training.metrics import precision, recall


class ModelTrainer:
    """Orchestrates model training with best practices.

    This class provides a high-level interface for training segmentation models
    with proper callbacks, checkpointing, and logging.

    Args:
        model: Keras model to train
        loss_function: Loss function (default: DICE loss)
        learning_rate: Initial learning rate
        metrics: List of metrics to track during training

    Example:
        >>> from med_seg.models import UNet
        >>> model_builder = UNet(input_size=256, base_filters=16)
        >>> model = model_builder.build()
        >>> trainer = ModelTrainer(model, learning_rate=0.0001)
        >>> history = trainer.train(
        ...     train_gen=train_generator,
        ...     val_gen=val_generator,
        ...     epochs=100,
        ...     steps_per_epoch=50
        ... )
    """

    def __init__(
        self,
        model: keras.Model,
        loss_function=None,
        learning_rate: float = 0.0001,
        metrics: Optional[list] = None
    ):
        self.model = model
        self.loss_function = loss_function or dice_loss
        self.learning_rate = learning_rate
        self.metrics = metrics or [dice_coefficient, precision, recall]
        self.history = None

    def compile(self):
        """Compile the model with specified loss and metrics."""
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

    def train(
        self,
        train_gen: Iterator[Tuple[np.ndarray, np.ndarray]],
        val_gen: Optional[Iterator[Tuple[np.ndarray, np.ndarray]]] = None,
        epochs: int = 100,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        callbacks: Optional[list] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """Train the model.

        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            epochs: Number of training epochs
            steps_per_epoch: Steps per epoch
            validation_steps: Validation steps per epoch
            callbacks: List of Keras callbacks
            verbose: Verbosity mode

        Returns:
            Training history object
        """
        # Compile if not already compiled
        if not self.model.optimizer:
            self.compile()

        # Train the model
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def save_model(self, filepath: str):
        """Save the trained model.

        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"[+] Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model.

        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(
            filepath,
            custom_objects={
                'dice_loss': self.loss_function,
                'dice_coefficient': dice_coefficient,
                'precision': precision,
                'recall': recall
            }
        )
        print(f"[+] Model loaded from {filepath}")

    def predict(
        self,
        x: np.ndarray,
        batch_size: int = 1,
        verbose: int = 0
    ) -> np.ndarray:
        """Make predictions on input data.

        Args:
            x: Input data
            batch_size: Batch size for prediction
            verbose: Verbosity mode

        Returns:
            Predicted segmentation masks
        """
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)
