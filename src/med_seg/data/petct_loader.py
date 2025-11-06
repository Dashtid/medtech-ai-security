"""PET/CT data loader for multi-modal nuclear medicine imaging.

This module provides specialized data loading for PET/CT datasets with
multi-modal inputs (PET + CT) and segmentation masks.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import SimpleITK as sitk


class PETCTLoader:
    """Loader for PET/CT nuclear medicine datasets.

    Handles loading paired PET and CT volumes with segmentation masks.
    Supports both 2D slice extraction and 3D volume loading.

    Args:
        data_dir: Root directory containing patient folders
        ct_filename: Name of CT file (default: 'CTres.nii.gz')
        pet_filename: Name of PET/SUV file (default: 'SUV.nii.gz')
        seg_filename: Name of segmentation file (default: 'SEG.nii.gz')
    """

    def __init__(
        self,
        data_dir: str,
        ct_filename: str = "CTres.nii.gz",
        pet_filename: str = "SUV.nii.gz",
        seg_filename: str = "SEG.nii.gz",
    ):
        self.data_dir = Path(data_dir)
        self.ct_filename = ct_filename
        self.pet_filename = pet_filename
        self.seg_filename = seg_filename

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # Find all patient directories
        self.patient_dirs = self._find_patient_dirs()

        if not self.patient_dirs:
            raise ValueError(f"No patient directories found in {data_dir}")

    def _find_patient_dirs(self) -> List[Path]:
        """Find all patient directories in data directory.

        Returns:
            List of patient directory paths
        """
        patient_dirs = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith("patient"):
                # Check if required files exist
                ct_file = item / self.ct_filename
                pet_file = item / self.pet_filename
                seg_file = item / self.seg_filename

                if ct_file.exists() and pet_file.exists() and seg_file.exists():
                    patient_dirs.append(item)

        return sorted(patient_dirs)

    def load_patient_3d(self, patient_idx: int) -> Dict[str, np.ndarray]:
        """Load full 3D volumes for one patient.

        Args:
            patient_idx: Index of patient (0 to len-1)

        Returns:
            Dictionary with keys: 'ct', 'pet', 'seg', 'spacing', 'origin'
        """
        if patient_idx < 0 or patient_idx >= len(self.patient_dirs):
            raise IndexError(
                f"Patient index {patient_idx} out of range [0, {len(self.patient_dirs)-1}]"
            )

        patient_dir = self.patient_dirs[patient_idx]

        # Load CT
        ct_path = patient_dir / self.ct_filename
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array = sitk.GetArrayFromImage(ct_image)  # (z, y, x)

        # Load PET/SUV
        pet_path = patient_dir / self.pet_filename
        pet_image = sitk.ReadImage(str(pet_path))
        pet_array = sitk.GetArrayFromImage(pet_image)  # (z, y, x)

        # Load segmentation
        seg_path = patient_dir / self.seg_filename
        seg_image = sitk.ReadImage(str(seg_path))
        seg_array = sitk.GetArrayFromImage(seg_image)  # (z, y, x)

        # Get metadata
        spacing = ct_image.GetSpacing()  # (x, y, z)
        origin = ct_image.GetOrigin()

        return {
            "ct": ct_array.astype(np.float32),
            "pet": pet_array.astype(np.float32),
            "seg": seg_array.astype(np.uint8),
            "spacing": spacing,
            "origin": origin,
            "patient_dir": patient_dir,
        }

    def extract_2d_slices(
        self, patient_idx: int, axis: int = 0, slice_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 2D slices from patient volumes.

        Args:
            patient_idx: Index of patient
            axis: Axis to slice along (0=axial/z, 1=coronal/y, 2=sagittal/x)
            slice_indices: Specific slice indices to extract (None = all slices)

        Returns:
            Tuple of (ct_slices, pet_slices, seg_slices)
            Each is shape (num_slices, height, width)
        """
        data = self.load_patient_3d(patient_idx)

        ct = data["ct"]
        pet = data["pet"]
        seg = data["seg"]

        # Select slices along specified axis
        if slice_indices is None:
            if axis == 0:  # Axial (z-axis)
                ct_slices = ct
                pet_slices = pet
                seg_slices = seg
            elif axis == 1:  # Coronal (y-axis)
                ct_slices = np.transpose(ct, (1, 0, 2))
                pet_slices = np.transpose(pet, (1, 0, 2))
                seg_slices = np.transpose(seg, (1, 0, 2))
            elif axis == 2:  # Sagittal (x-axis)
                ct_slices = np.transpose(ct, (2, 0, 1))
                pet_slices = np.transpose(pet, (2, 0, 1))
                seg_slices = np.transpose(seg, (2, 0, 1))
            else:
                raise ValueError(f"Invalid axis: {axis} (must be 0, 1, or 2)")
        else:
            # Extract specific slices
            if axis == 0:
                ct_slices = ct[slice_indices]
                pet_slices = pet[slice_indices]
                seg_slices = seg[slice_indices]
            elif axis == 1:
                ct_slices = ct[:, slice_indices, :]
                pet_slices = pet[:, slice_indices, :]
                seg_slices = seg[:, slice_indices, :]
                ct_slices = np.transpose(ct_slices, (1, 0, 2))
                pet_slices = np.transpose(pet_slices, (1, 0, 2))
                seg_slices = np.transpose(seg_slices, (1, 0, 2))
            elif axis == 2:
                ct_slices = ct[:, :, slice_indices]
                pet_slices = pet[:, :, slice_indices]
                seg_slices = seg[:, :, slice_indices]
                ct_slices = np.transpose(ct_slices, (2, 0, 1))
                pet_slices = np.transpose(pet_slices, (2, 0, 1))
                seg_slices = np.transpose(seg_slices, (2, 0, 1))

        return ct_slices, pet_slices, seg_slices

    def get_tumor_slices(
        self, patient_idx: int, axis: int = 0, min_tumor_voxels: int = 10
    ) -> List[int]:
        """Find slice indices containing tumors.

        Args:
            patient_idx: Index of patient
            axis: Axis to slice along (0=axial, 1=coronal, 2=sagittal)
            min_tumor_voxels: Minimum tumor voxels required per slice

        Returns:
            List of slice indices with tumors
        """
        data = self.load_patient_3d(patient_idx)
        seg = data["seg"]

        # Count tumor voxels per slice
        if axis == 0:  # Axial
            tumor_counts = np.sum(seg, axis=(1, 2))
        elif axis == 1:  # Coronal
            tumor_counts = np.sum(seg, axis=(0, 2))
        elif axis == 2:  # Sagittal
            tumor_counts = np.sum(seg, axis=(0, 1))
        else:
            raise ValueError(f"Invalid axis: {axis}")

        # Find slices with sufficient tumor content
        tumor_slice_indices = np.where(tumor_counts >= min_tumor_voxels)[0]

        return tumor_slice_indices.tolist()

    def load_all_2d_slices(
        self, axis: int = 0, include_empty: bool = False, min_tumor_voxels: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load all 2D slices from all patients.

        Args:
            axis: Axis to slice along (0=axial, 1=coronal, 2=sagittal)
            include_empty: Include slices without tumors
            min_tumor_voxels: Minimum tumor voxels for non-empty slices

        Returns:
            Tuple of (ct_slices, pet_slices, seg_slices)
            Each is shape (total_slices, height, width)
        """
        all_ct_slices = []
        all_pet_slices = []
        all_seg_slices = []

        for patient_idx in range(len(self.patient_dirs)):
            if include_empty:
                # Get all slices
                ct_slices, pet_slices, seg_slices = self.extract_2d_slices(patient_idx, axis=axis)
            else:
                # Get only slices with tumors
                tumor_indices = self.get_tumor_slices(
                    patient_idx, axis=axis, min_tumor_voxels=min_tumor_voxels
                )
                if len(tumor_indices) == 0:
                    continue

                ct_slices, pet_slices, seg_slices = self.extract_2d_slices(
                    patient_idx, axis=axis, slice_indices=tumor_indices
                )

            all_ct_slices.append(ct_slices)
            all_pet_slices.append(pet_slices)
            all_seg_slices.append(seg_slices)

        # Concatenate all slices
        all_ct_slices = np.concatenate(all_ct_slices, axis=0)
        all_pet_slices = np.concatenate(all_pet_slices, axis=0)
        all_seg_slices = np.concatenate(all_seg_slices, axis=0)

        return all_ct_slices, all_pet_slices, all_seg_slices

    def get_statistics(self) -> Dict:
        """Compute dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "num_patients": len(self.patient_dirs),
            "ct_range": {"min": float("inf"), "max": float("-inf")},
            "pet_range": {"min": float("inf"), "max": float("-inf")},
            "tumor_prevalence": [],
            "volume_shapes": [],
        }

        for patient_idx in range(len(self.patient_dirs)):
            data = self.load_patient_3d(patient_idx)

            ct = data["ct"]
            pet = data["pet"]
            seg = data["seg"]

            # Update ranges
            stats["ct_range"]["min"] = min(stats["ct_range"]["min"], float(ct.min()))
            stats["ct_range"]["max"] = max(stats["ct_range"]["max"], float(ct.max()))
            stats["pet_range"]["min"] = min(stats["pet_range"]["min"], float(pet.min()))
            stats["pet_range"]["max"] = max(stats["pet_range"]["max"], float(pet.max()))

            # Tumor prevalence
            total_voxels = seg.size
            tumor_voxels = np.sum(seg > 0)
            prevalence = tumor_voxels / total_voxels * 100
            stats["tumor_prevalence"].append(prevalence)

            # Volume shapes
            stats["volume_shapes"].append(ct.shape)

        # Compute averages
        stats["avg_tumor_prevalence"] = np.mean(stats["tumor_prevalence"])
        stats["std_tumor_prevalence"] = np.std(stats["tumor_prevalence"])

        return stats

    def __len__(self) -> int:
        """Get number of patients in dataset."""
        return len(self.patient_dirs)

    def __repr__(self) -> str:
        """String representation of loader."""
        return (
            f"PETCTLoader("
            f"data_dir={self.data_dir}, "
            f"num_patients={len(self)}, "
            f"ct={self.ct_filename}, "
            f"pet={self.pet_filename}, "
            f"seg={self.seg_filename})"
        )
