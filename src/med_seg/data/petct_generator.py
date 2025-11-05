"""TensorFlow/Keras data generator for PET/CT datasets.

This module provides a Keras-compatible data generator for efficient
batch loading and preprocessing of PET/CT data during training.
"""

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

from med_seg.data.petct_loader import PETCTLoader
from med_seg.data.petct_preprocessor import PETCTPreprocessor


class PETCTDataGenerator(tf.keras.utils.Sequence):
    """Keras data generator for PET/CT datasets.

    Generates batches of preprocessed PET/CT slices for training.
    Supports data augmentation and balanced sampling.

    Args:
        loader: PETCTLoader instance
        preprocessor: PETCTPreprocessor instance
        batch_size: Batch size
        axis: Slice axis (0=axial, 1=coronal, 2=sagittal)
        shuffle: Shuffle data each epoch
        augment: Apply data augmentation
        tumor_only: Only include slices with tumors
        min_tumor_voxels: Minimum tumor voxels per slice (if tumor_only=True)
    """

    def __init__(
        self,
        loader: PETCTLoader,
        preprocessor: PETCTPreprocessor,
        batch_size: int = 8,
        axis: int = 0,
        shuffle: bool = True,
        augment: bool = False,
        tumor_only: bool = True,
        min_tumor_voxels: int = 10
    ):
        self.loader = loader
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.axis = axis
        self.shuffle = shuffle
        self.augment = augment
        self.tumor_only = tumor_only
        self.min_tumor_voxels = min_tumor_voxels

        # Build index of all slices
        self._build_slice_index()

        # Shuffle on initialization
        if self.shuffle:
            self._shuffle_indices()

    def _build_slice_index(self):
        """Build index of (patient_idx, slice_idx) for all slices."""
        self.slice_index = []

        for patient_idx in range(len(self.loader)):
            if self.tumor_only:
                # Get slices with tumors
                tumor_slices = self.loader.get_tumor_slices(
                    patient_idx,
                    axis=self.axis,
                    min_tumor_voxels=self.min_tumor_voxels
                )
                for slice_idx in tumor_slices:
                    self.slice_index.append((patient_idx, slice_idx))
            else:
                # Get all slices
                data = self.loader.load_patient_3d(patient_idx)
                num_slices = data['ct'].shape[self.axis]
                for slice_idx in range(num_slices):
                    self.slice_index.append((patient_idx, slice_idx))

    def _shuffle_indices(self):
        """Shuffle slice indices."""
        np.random.shuffle(self.slice_index)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.slice_index) / self.batch_size))

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get one batch of data.

        Args:
            batch_idx: Batch index

        Returns:
            Tuple of (inputs, targets)
            - inputs: (batch_size, H, W, 2) with channels [CT, SUV]
            - targets: (batch_size, H, W, 1) binary masks
        """
        # Get slice indices for this batch
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.slice_index))

        batch_indices = self.slice_index[start_idx:end_idx]

        # Load and preprocess slices
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
        inputs, targets = self.preprocessor.preprocess_batch_2d(
            ct_slices, suv_slices, seg_slices
        )

        # Apply augmentation if enabled
        if self.augment:
            inputs, targets = self._augment_batch(inputs, targets)

        return inputs, targets

    def _augment_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to batch.

        Args:
            inputs: Input images (N, H, W, 2)
            targets: Target masks (N, H, W, 1)

        Returns:
            Augmented (inputs, targets)
        """
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            # Random horizontal flip
            if np.random.rand() > 0.5:
                inputs[i] = np.fliplr(inputs[i])
                targets[i] = np.fliplr(targets[i])

            # Random vertical flip
            if np.random.rand() > 0.5:
                inputs[i] = np.flipud(inputs[i])
                targets[i] = np.flipud(targets[i])

            # Random 90-degree rotations
            k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degrees
            if k > 0:
                inputs[i] = np.rot90(inputs[i], k)
                targets[i] = np.rot90(targets[i], k)

        return inputs, targets

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.shuffle:
            self._shuffle_indices()

    def get_class_weights(self) -> dict:
        """Compute class weights for imbalanced data.

        Returns:
            Dictionary with class weights {0: weight_background, 1: weight_tumor}
        """
        total_voxels = 0
        tumor_voxels = 0

        # Sample subset of data to compute weights
        num_samples = min(100, len(self.slice_index))
        sample_indices = np.random.choice(len(self.slice_index), num_samples, replace=False)

        for idx in sample_indices:
            patient_idx, slice_idx = self.slice_index[idx]

            # Load slice
            _, _, seg_batch = self.loader.extract_2d_slices(
                patient_idx, axis=self.axis, slice_indices=[slice_idx]
            )

            seg_slice = seg_batch[0]
            total_voxels += seg_slice.size
            tumor_voxels += np.sum(seg_slice > 0)

        background_voxels = total_voxels - tumor_voxels

        # Compute weights (inverse frequency)
        if tumor_voxels > 0:
            weight_background = 1.0
            weight_tumor = background_voxels / tumor_voxels
        else:
            weight_background = 1.0
            weight_tumor = 1.0

        return {
            0: weight_background,
            1: weight_tumor
        }


def create_petct_generators(
    data_dir: str,
    train_patients: list,
    val_patients: list,
    batch_size: int = 8,
    target_size: Optional[Tuple[int, int]] = (256, 256),
    augment_train: bool = True,
    tumor_only: bool = True
) -> Tuple[PETCTDataGenerator, PETCTDataGenerator]:
    """Create train and validation data generators.

    Args:
        data_dir: Directory containing patient data
        train_patients: List of patient directory names for training
        val_patients: List of patient directory names for validation
        batch_size: Batch size
        target_size: Target image size (H, W)
        augment_train: Apply augmentation to training data
        tumor_only: Only include slices with tumors

    Returns:
        Tuple of (train_generator, val_generator)
    """
    # Create loader (will be filtered by train/val splits)
    from pathlib import Path

    # TODO: Implement train/val split filtering
    # For now, use all data
    loader = PETCTLoader(data_dir)

    # Create preprocessor
    preprocessor = PETCTPreprocessor(
        target_size=target_size,
        ct_window_center=0,
        ct_window_width=400,
        suv_max=15
    )

    # Create generators
    train_gen = PETCTDataGenerator(
        loader=loader,
        preprocessor=preprocessor,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
        tumor_only=tumor_only
    )

    val_gen = PETCTDataGenerator(
        loader=loader,
        preprocessor=preprocessor,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        tumor_only=tumor_only
    )

    return train_gen, val_gen
