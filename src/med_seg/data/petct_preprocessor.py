"""PET/CT preprocessing pipeline for nuclear medicine imaging.

This module provides preprocessing specific to PET/CT data including
intensity windowing, normalization, and multi-modal data preparation.
"""

from typing import Tuple, Optional
import numpy as np


class PETCTPreprocessor:
    """Preprocessing pipeline for PET/CT nuclear medicine data.

    Handles CT windowing, SUV normalization, resizing, and data formatting
    for deep learning models.

    Args:
        target_size: Target image size (height, width) for 2D, or None to keep original
        ct_window_center: CT window center in HU (default: 0 for soft tissue)
        ct_window_width: CT window width in HU (default: 400 for soft tissue)
        suv_max: Maximum SUV for clipping (default: 15)
        normalize_method: Normalization method ('min-max' or 'z-score')
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        ct_window_center: float = 0.0,
        ct_window_width: float = 400.0,
        suv_max: float = 15.0,
        normalize_method: str = "min-max",
    ):
        self.target_size = target_size
        self.ct_window_center = ct_window_center
        self.ct_window_width = ct_window_width
        self.suv_max = suv_max
        self.normalize_method = normalize_method

        # Compute CT window bounds
        self.ct_min = ct_window_center - ct_window_width / 2
        self.ct_max = ct_window_center + ct_window_width / 2

    def window_ct(self, ct: np.ndarray) -> np.ndarray:
        """Apply intensity windowing to CT data.

        Args:
            ct: CT volume or slice in Hounsfield Units

        Returns:
            Windowed CT normalized to [0, 1]
        """
        ct_windowed = np.clip(ct, self.ct_min, self.ct_max)
        ct_normalized = (ct_windowed - self.ct_min) / (self.ct_max - self.ct_min)
        return ct_normalized.astype(np.float32)

    def normalize_suv(self, suv: np.ndarray) -> np.ndarray:
        """Normalize SUV/PET data.

        Args:
            suv: SUV volume or slice

        Returns:
            Normalized SUV in [0, 1]
        """
        suv_clipped = np.clip(suv, 0, self.suv_max)
        suv_normalized = suv_clipped / self.suv_max
        return suv_normalized.astype(np.float32)

    def resize_2d(
        self, ct: np.ndarray, suv: np.ndarray, seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Resize 2D slices to target size.

        Args:
            ct: CT slice (H, W)
            suv: SUV slice (H, W)
            seg: Segmentation slice (H, W), optional

        Returns:
            Tuple of (resized_ct, resized_suv, resized_seg)
        """
        if self.target_size is None:
            return ct, suv, seg

        try:
            from skimage.transform import resize as sk_resize
        except ImportError:
            raise ImportError(
                "scikit-image required for resizing. Install with: pip install scikit-image"
            )

        # Resize CT and SUV with bilinear interpolation
        ct_resized = sk_resize(
            ct, self.target_size, order=1, preserve_range=True, anti_aliasing=True  # Bilinear
        ).astype(ct.dtype)

        suv_resized = sk_resize(
            suv, self.target_size, order=1, preserve_range=True, anti_aliasing=True  # Bilinear
        ).astype(suv.dtype)

        # Resize segmentation with nearest neighbor
        seg_resized = None
        if seg is not None:
            seg_resized = sk_resize(
                seg,
                self.target_size,
                order=0,  # Nearest neighbor
                preserve_range=True,
                anti_aliasing=False,
            ).astype(seg.dtype)

        return ct_resized, suv_resized, seg_resized

    def preprocess_2d_slice(
        self, ct: np.ndarray, suv: np.ndarray, seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess a single 2D slice.

        Args:
            ct: CT slice (H, W)
            suv: SUV slice (H, W)
            seg: Segmentation slice (H, W), optional

        Returns:
            Tuple of (preprocessed_input, preprocessed_seg)
            - preprocessed_input: (H, W, 2) with channels [CT, SUV]
            - preprocessed_seg: (H, W, 1) binary mask or None
        """
        # Resize if needed
        if self.target_size is not None:
            ct, suv, seg = self.resize_2d(ct, suv, seg)

        # Apply windowing and normalization
        ct_norm = self.window_ct(ct)
        suv_norm = self.normalize_suv(suv)

        # Stack CT and SUV as channels (H, W, 2)
        multi_modal_input = np.stack([ct_norm, suv_norm], axis=-1)

        # Process segmentation
        seg_processed = None
        if seg is not None:
            # Ensure binary and add channel dimension
            seg_binary = (seg > 0).astype(np.float32)
            seg_processed = np.expand_dims(seg_binary, axis=-1)  # (H, W, 1)

        return multi_modal_input, seg_processed

    def preprocess_batch_2d(
        self, ct_batch: np.ndarray, suv_batch: np.ndarray, seg_batch: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess a batch of 2D slices.

        Args:
            ct_batch: CT slices (N, H, W)
            suv_batch: SUV slices (N, H, W)
            seg_batch: Segmentation slices (N, H, W), optional

        Returns:
            Tuple of (preprocessed_inputs, preprocessed_segs)
            - preprocessed_inputs: (N, H, W, 2) with channels [CT, SUV]
            - preprocessed_segs: (N, H, W, 1) binary masks or None
        """
        batch_size = ct_batch.shape[0]

        inputs_list = []
        segs_list = []

        for i in range(batch_size):
            ct_slice = ct_batch[i]
            suv_slice = suv_batch[i]
            seg_slice = seg_batch[i] if seg_batch is not None else None

            input_processed, seg_processed = self.preprocess_2d_slice(
                ct_slice, suv_slice, seg_slice
            )

            inputs_list.append(input_processed)
            if seg_processed is not None:
                segs_list.append(seg_processed)

        # Stack into batches
        inputs_batch = np.stack(inputs_list, axis=0)
        segs_batch = np.stack(segs_list, axis=0) if segs_list else None

        return inputs_batch, segs_batch

    def preprocess_3d_volume(
        self, ct: np.ndarray, suv: np.ndarray, seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess a 3D volume.

        Args:
            ct: CT volume (D, H, W)
            suv: SUV volume (D, H, W)
            seg: Segmentation volume (D, H, W), optional

        Returns:
            Tuple of (preprocessed_input, preprocessed_seg)
            - preprocessed_input: (D, H, W, 2) with channels [CT, SUV]
            - preprocessed_seg: (D, H, W, 1) binary mask or None
        """
        # Apply windowing and normalization
        ct_norm = self.window_ct(ct)
        suv_norm = self.normalize_suv(suv)

        # Stack CT and SUV as channels (D, H, W, 2)
        multi_modal_input = np.stack([ct_norm, suv_norm], axis=-1)

        # Process segmentation
        seg_processed = None
        if seg is not None:
            # Ensure binary and add channel dimension
            seg_binary = (seg > 0).astype(np.float32)
            seg_processed = np.expand_dims(seg_binary, axis=-1)  # (D, H, W, 1)

        return multi_modal_input, seg_processed

    def get_ct_window_presets(self) -> dict:
        """Get common CT window presets.

        Returns:
            Dictionary of window presets
        """
        return {
            "soft_tissue": {"center": 0, "width": 400},
            "lung": {"center": -600, "width": 1500},
            "bone": {"center": 300, "width": 1500},
            "brain": {"center": 40, "width": 80},
            "liver": {"center": 60, "width": 150},
            "mediastinum": {"center": 50, "width": 350},
        }

    def set_ct_window(self, preset: str):
        """Set CT window using a preset.

        Args:
            preset: Window preset name (soft_tissue, lung, bone, etc.)
        """
        presets = self.get_ct_window_presets()

        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        window = presets[preset]
        self.ct_window_center = window["center"]
        self.ct_window_width = window["width"]

        # Update window bounds
        self.ct_min = self.ct_window_center - self.ct_window_width / 2
        self.ct_max = self.ct_window_center + self.ct_window_width / 2

    def __repr__(self) -> str:
        """String representation of preprocessor."""
        return (
            f"PETCTPreprocessor("
            f"target_size={self.target_size}, "
            f"ct_window=[{self.ct_min:.0f}, {self.ct_max:.0f}], "
            f"suv_max={self.suv_max}, "
            f"normalize={self.normalize_method})"
        )
