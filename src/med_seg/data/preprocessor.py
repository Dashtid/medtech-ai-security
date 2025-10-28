"""Preprocessing utilities for medical images."""

from typing import Tuple, Optional
import numpy as np


class MedicalImagePreprocessor:
    """Preprocessing pipeline for medical images.

    Args:
        target_size: Target image size (height, width)
        normalization_method: Normalization method ('min-max', 'z-score', or None)
        clip_range: Value range to clip before normalization (min, max)
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalization_method: str = "min-max",
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        self.target_size = target_size
        self.normalization_method = normalization_method
        self.clip_range = clip_range

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensities.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        image = image.astype(np.float32)

        # Clip values if specified
        if self.clip_range:
            image = np.clip(image, self.clip_range[0], self.clip_range[1])

        if self.normalization_method == "min-max":
            # Scale to [0, 1]
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        elif self.normalization_method == "z-score":
            # Standardize to mean=0, std=1
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std

        return image

    def resize(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        """Resize image and mask to target size.

        Args:
            image: Input image
            mask: Input mask (optional)

        Returns:
            Tuple of (resized_image, resized_mask)
        """
        if not self.target_size:
            return image, mask

        try:
            from skimage.transform import resize as sk_resize
        except ImportError:
            raise ImportError("scikit-image required for resizing. Install with: pip install scikit-image")

        resized_image = sk_resize(
            image,
            self.target_size,
            preserve_range=True,
            anti_aliasing=True
        ).astype(image.dtype)

        resized_mask = None
        if mask is not None:
            resized_mask = sk_resize(
                mask,
                self.target_size,
                order=0,  # Nearest neighbor for masks
                preserve_range=True,
                anti_aliasing=False
            ).astype(mask.dtype)

        return resized_image, resized_mask

    def preprocess(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply full preprocessing pipeline.

        Args:
            image: Input image
            mask: Input mask (optional)

        Returns:
            Tuple of (preprocessed_image, preprocessed_mask)
        """
        # Resize if target size specified
        if self.target_size:
            image, mask = self.resize(image, mask)

        # Normalize image (not mask)
        image = self.normalize(image)

        # Ensure correct dimensions (H, W, C)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if mask is not None and mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        return image, mask

    def preprocess_batch(
        self,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess a batch of images and masks.

        Args:
            images: Batch of images (N, H, W) or (N, H, W, C)
            masks: Batch of masks (N, H, W) (optional)

        Returns:
            Tuple of (preprocessed_images, preprocessed_masks)
        """
        processed_images = []
        processed_masks = []

        for i in range(len(images)):
            image = images[i]
            mask = masks[i] if masks is not None else None

            proc_img, proc_mask = self.preprocess(image, mask)
            processed_images.append(proc_img)

            if proc_mask is not None:
                processed_masks.append(proc_mask)

        processed_images = np.array(processed_images)
        processed_masks = np.array(processed_masks) if processed_masks else None

        return processed_images, processed_masks

    def __repr__(self) -> str:
        """String representation of preprocessor."""
        return (
            f"MedicalImagePreprocessor("
            f"target_size={self.target_size}, "
            f"normalization={self.normalization_method}, "
            f"clip_range={self.clip_range})"
        )
