"""Data loading utilities for medical imaging datasets."""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import SimpleITK as sitk


class MedicalImageLoader:
    """Loader for medical imaging datasets.

    Supports multiple formats:
    - 2D images: PNG, JPG, TIFF
    - 3D volumes: NIfTI (.nii, .nii.gz), DICOM
    - Medical formats: MHA, MHD

    Args:
        data_dir: Root directory containing images
        mask_dir: Directory containing segmentation masks (optional)
        image_extension: File extension for images (e.g., '.png', '.nii.gz')
        mask_extension: File extension for masks
    """

    def __init__(
        self,
        data_dir: str,
        mask_dir: Optional[str] = None,
        image_extension: str = ".png",
        mask_extension: str = ".png",
    ):
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_extension = image_extension
        self.mask_extension = mask_extension

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        if self.mask_dir and not self.mask_dir.exists():
            raise ValueError(f"Mask directory does not exist: {mask_dir}")

    def load_2d_image(self, image_path: str) -> np.ndarray:
        """Load a 2D image from file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (H, W) or (H, W, C)
        """
        if image_path.endswith(('.nii', '.nii.gz')):
            # Load NIfTI file
            image = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(image)
            # NIfTI is (D, H, W), take middle slice if 3D
            if image.ndim == 3:
                image = image[image.shape[0] // 2]
        else:
            # Load standard image formats
            image = Image.open(image_path)
            image = np.array(image)

        return image

    def load_2d_mask(self, mask_path: str) -> np.ndarray:
        """Load a 2D segmentation mask from file.

        Args:
            mask_path: Path to mask file

        Returns:
            Mask as numpy array (H, W)
        """
        mask = self.load_2d_image(mask_path)

        # Ensure binary mask (0 or 1)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        return mask

    def get_image_paths(self) -> List[Path]:
        """Get all image file paths in data directory.

        Returns:
            List of image paths
        """
        image_paths = sorted(
            self.data_dir.glob(f"*{self.image_extension}")
        )
        return image_paths

    def get_mask_paths(self) -> Optional[List[Path]]:
        """Get all mask file paths in mask directory.

        Returns:
            List of mask paths or None if no mask directory
        """
        if not self.mask_dir:
            return None

        mask_paths = sorted(
            self.mask_dir.glob(f"*{self.mask_extension}")
        )
        return mask_paths

    def load_dataset_2d(
        self,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load entire 2D dataset into memory.

        Args:
            max_samples: Maximum number of samples to load (None for all)

        Returns:
            Tuple of (images, masks) as numpy arrays
            masks is None if no mask directory specified
        """
        image_paths = self.get_image_paths()

        if max_samples:
            image_paths = image_paths[:max_samples]

        images = []
        masks = []

        for img_path in image_paths:
            # Load image
            image = self.load_2d_image(str(img_path))
            images.append(image)

            # Load corresponding mask if available
            if self.mask_dir:
                mask_path = self.mask_dir / img_path.name.replace(
                    self.image_extension,
                    self.mask_extension
                )
                if mask_path.exists():
                    mask = self.load_2d_mask(str(mask_path))
                    masks.append(mask)

        images = np.array(images)
        masks = np.array(masks) if masks else None

        return images, masks

    def load_3d_volume(self, volume_path: str) -> np.ndarray:
        """Load a 3D medical image volume.

        Args:
            volume_path: Path to 3D volume file

        Returns:
            Volume as numpy array (D, H, W)
        """
        volume = sitk.ReadImage(volume_path)
        volume = sitk.GetArrayFromImage(volume)
        return volume

    def __len__(self) -> int:
        """Get number of images in dataset."""
        return len(self.get_image_paths())

    def __repr__(self) -> str:
        """String representation of loader."""
        return (
            f"MedicalImageLoader("
            f"data_dir={self.data_dir}, "
            f"num_images={len(self)}, "
            f"has_masks={self.mask_dir is not None})"
        )
