"""Data generator for multi-task learning (segmentation + survival)."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from .petct_generator import PETCTDataGenerator


class SurvivalDataGenerator(PETCTDataGenerator):
    """Data generator for multi-task learning with survival prediction.

    Extends PETCTDataGenerator to also provide survival labels (time, event).
    """

    def __init__(
        self,
        loader,
        preprocessor,
        survival_data_path: str,
        batch_size: int = 8,
        shuffle: bool = True,
        augment: bool = False,
        tumor_only: bool = True,
    ):
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

        # Store target size from preprocessor
        self.target_size = (
            preprocessor.target_size if preprocessor.target_size is not None else (256, 256)
        )

        # Load survival data
        self.survival_data = self._load_survival_data(survival_data_path)

    def _load_survival_data(self, path: str) -> Dict:
        """Load survival data from JSON file.

        Args:
            path: Path to survival_data.json

        Returns:
            Dictionary mapping patient_id to survival info
        """
        with open(path, "r") as f:
            survival_list = json.load(f)

        # Convert list to dict keyed by patient_id
        survival_dict = {item["patient_id"]: item for item in survival_list}

        return survival_dict

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, Dict]:
        """Get batch of data.

        Returns:
            Tuple of (inputs, targets):
            - inputs: (batch, H, W, 2) PET+CT
            - targets: {
                'segmentation': (batch, H, W, 1),
                'survival': (batch, 2)  # [time, event]
              }
        """
        # Get batch indices
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.slice_index))
        batch_indices = self.slice_index[start_idx:end_idx]

        # Load and preprocess slices (use parent class logic)
        ct_slices = []
        suv_slices = []
        seg_slices = []

        for patient_idx, slice_idx in batch_indices:
            # Extract single slice
            ct_batch, suv_batch, seg_batch = self.loader.extract_2d_slices(
                patient_idx, axis=self.axis, slice_indices=[slice_idx]
            )

            ct_slices.append(ct_batch[0])
            suv_slices.append(suv_batch[0])
            seg_slices.append(seg_batch[0])

        # Convert to arrays
        ct_slices = np.array(ct_slices)
        suv_slices = np.array(suv_slices)
        seg_slices = np.array(seg_slices)

        # Preprocess batch
        inputs, seg_targets = self.preprocessor.preprocess_batch_2d(
            ct_slices, suv_slices, seg_slices
        )

        # Apply augmentation if enabled
        if self.augment:
            inputs, seg_targets = self._augment_batch(inputs, seg_targets)

        # Get survival data for each patient in batch
        batch_size_actual = len(batch_indices)
        surv_targets = np.zeros((batch_size_actual, 2))  # [time, event]

        for i, (patient_idx, _) in enumerate(batch_indices):
            # Get patient ID from directory name
            patient_dir = self.loader.patient_dirs[patient_idx]
            patient_id = patient_dir.name  # e.g., "patient_001"

            if patient_id in self.survival_data:
                surv_info = self.survival_data[patient_id]
                time = surv_info["observed_time_months"]
                event = 1.0 if surv_info["event_occurred"] else 0.0
                surv_targets[i] = [time, event]
            else:
                # If survival data missing, use defaults
                surv_targets[i] = [30.0, 0.0]  # 30 months, censored

        # Return inputs and target dictionary
        targets = {"segmentation": seg_targets, "survival": surv_targets}

        return inputs, targets


def create_survival_generators(
    data_dir: str,
    preprocessor,
    batch_size: int = 8,
    train_fraction: float = 0.7,
    augment_train: bool = True,
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
    survival_path = Path(data_dir) / "survival_data.json"

    if not survival_path.exists():
        raise FileNotFoundError(f"Survival data not found: {survival_path}")

    # Split patients into train/val
    patient_dirs = loader.patient_dirs
    n_train = int(len(patient_dirs) * train_fraction)

    train_dirs = patient_dirs[:n_train]
    val_dirs = patient_dirs[n_train:]

    # Create loaders for each split
    train_loader = PETCTLoader(data_dir)
    train_loader.patient_dirs = train_dirs

    val_loader = PETCTLoader(data_dir)
    val_loader.patient_dirs = val_dirs

    # Create generators
    train_gen = SurvivalDataGenerator(
        loader=train_loader,
        preprocessor=preprocessor,
        survival_data_path=str(survival_path),
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
        tumor_only=True,
    )

    val_gen = SurvivalDataGenerator(
        loader=val_loader,
        preprocessor=preprocessor,
        survival_data_path=str(survival_path),
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        tumor_only=True,
    )

    return train_gen, val_gen
