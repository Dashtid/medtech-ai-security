"""Data generator for multi-task learning (segmentation + survival)."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import tensorflow as tf

from .petct_generator import PETCTDataGenerator


class SurvivalDataGenerator(PETCTDataGenerator):
    """Data generator for multi-task learning with survival prediction.

    Extends PETCTDataGenerator to also provide survival labels (time, event).
    """

    def __init__(self,
                 loader,
                 preprocessor,
                 survival_data_path: str,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 augment: bool = False,
                 tumor_only: bool = True):
        """Initialize survival data generator.

        Args:
            loader: PETCTLoader instance
            preprocessor: PETCTPreprocessor instance
            survival_data_path: Path to survival_data.json
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply data augmentation
            tumor_only: Whether to only include tumor-containing slices
        """
        super().__init__(loader, preprocessor, batch_size, shuffle, augment, tumor_only)

        # Load survival data
        self.survival_data = self._load_survival_data(survival_data_path)

    def _load_survival_data(self, path: str) -> Dict:
        """Load survival data from JSON file.

        Args:
            path: Path to survival_data.json

        Returns:
            Dictionary mapping patient_id to survival info
        """
        with open(path, 'r') as f:
            survival_list = json.load(f)

        # Convert list to dict keyed by patient_id
        survival_dict = {
            item['patient_id']: item
            for item in survival_list
        }

        return survival_dict

    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:
        """Get batch of data.

        Returns:
            Tuple of (inputs, targets) where both are dictionaries:
            - inputs: {'input_petct': (batch, H, W, 2)}
            - targets: {
                'segmentation': (batch, H, W, 1),
                'survival': (batch, 2)  # [time, event]
              }
        """
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.slice_index))
        batch_indices = self.slice_index[start_idx:end_idx]

        # Initialize batch arrays
        batch_size_actual = len(batch_indices)
        x_batch = np.zeros((batch_size_actual, self.target_size[0], self.target_size[1], 2))
        y_seg_batch = np.zeros((batch_size_actual, self.target_size[0], self.target_size[1], 1))
        y_surv_batch = np.zeros((batch_size_actual, 2))  # [time, event]

        # Fill batch
        for i, idx in enumerate(batch_indices):
            patient_id, slice_idx = idx

            # Get image and segmentation (from parent class logic)
            pet, ct, seg = self.loader.get_slice(patient_id, slice_idx)
            x, y = self.preprocessor.preprocess_pair(pet, ct, seg)

            # Apply augmentation if enabled
            if self.augment:
                x, y = self._augment(x, y)

            x_batch[i] = x
            y_seg_batch[i] = np.expand_dims(y, axis=-1)

            # Get survival data for this patient
            if patient_id in self.survival_data:
                surv_info = self.survival_data[patient_id]
                time = surv_info['observed_time_months']
                event = 1.0 if surv_info['event_occurred'] else 0.0
                y_surv_batch[i] = [time, event]
            else:
                # If survival data missing, use defaults
                y_surv_batch[i] = [30.0, 0.0]  # 30 months, censored

        # Return as dictionaries for multi-output model
        inputs = x_batch
        targets = {
            'segmentation': y_seg_batch,
            'survival': y_surv_batch
        }

        return inputs, targets


def create_survival_generators(
    data_dir: str,
    preprocessor,
    batch_size: int = 8,
    train_fraction: float = 0.7,
    augment_train: bool = True
) -> Tuple[SurvivalDataGenerator, SurvivalDataGenerator]:
    """Create train and validation survival generators.

    Args:
        data_dir: Directory containing patient data and survival_data.json
        preprocessor: PETCTPreprocessor instance
        batch_size: Batch size
        train_fraction: Fraction of data for training
        augment_train: Whether to augment training data

    Returns:
        (train_generator, val_generator)
    """
    from .petct_loader import PETCTLoader

    # Load data
    loader = PETCTLoader(data_dir)
    survival_path = Path(data_dir) / 'survival_data.json'

    if not survival_path.exists():
        raise FileNotFoundError(f"Survival data not found: {survival_path}")

    # Split patients into train/val
    patient_ids = sorted(list(loader.patients.keys()))
    n_train = int(len(patient_ids) * train_fraction)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:]

    # Create loaders for each split
    train_loader = PETCTLoader(data_dir)
    train_loader.patients = {pid: loader.patients[pid] for pid in train_ids}

    val_loader = PETCTLoader(data_dir)
    val_loader.patients = {pid: loader.patients[pid] for pid in val_ids}

    # Create generators
    train_gen = SurvivalDataGenerator(
        loader=train_loader,
        preprocessor=preprocessor,
        survival_data_path=str(survival_path),
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
        tumor_only=True
    )

    val_gen = SurvivalDataGenerator(
        loader=val_loader,
        preprocessor=preprocessor,
        survival_data_path=str(survival_path),
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        tumor_only=True
    )

    return train_gen, val_gen
