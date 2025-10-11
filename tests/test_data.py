"""Unit tests for data loading and preprocessing."""

import pytest
import numpy as np
from pathlib import Path

from med_seg.data.loader import natural_sort_key
from med_seg.data.preprocessor import MedicalImagePreprocessor


class TestNaturalSort:
    """Test natural sorting functionality."""

    def test_natural_sort_numbers(self):
        """Test natural sorting with numbers."""
        files = ['image10.nii', 'image2.nii', 'image1.nii', 'image20.nii']
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == ['image1.nii', 'image2.nii', 'image10.nii', 'image20.nii']

    def test_natural_sort_mixed(self):
        """Test natural sorting with mixed content."""
        files = ['case10_seg2.nii', 'case2_seg1.nii', 'case10_seg1.nii']
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == ['case2_seg1.nii', 'case10_seg1.nii', 'case10_seg2.nii']


class TestMedicalImagePreprocessor:
    """Test medical image preprocessing."""

    @pytest.fixture
    def sample_image(self):
        """Create sample image for testing."""
        return np.random.rand(128, 128).astype(np.float32) * 100

    @pytest.fixture
    def sample_mask(self):
        """Create sample binary mask."""
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[40:80, 40:80] = 1.0
        return mask

    def test_min_max_normalization(self, sample_image):
        """Test min-max normalization."""
        preprocessor = MedicalImagePreprocessor(
            normalization_method='min-max',
            intensity_range=(0.0, 1.0)
        )

        normalized = preprocessor.normalize(sample_image)

        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
        assert normalized.shape == sample_image.shape

    def test_z_score_normalization(self, sample_image):
        """Test z-score normalization."""
        preprocessor = MedicalImagePreprocessor(
            normalization_method='z-score'
        )

        normalized = preprocessor.normalize(sample_image)

        # Z-score should have approximately 0 mean and 1 std
        assert np.abs(np.mean(normalized)) < 0.1
        assert np.abs(np.std(normalized) - 1.0) < 0.1

    def test_percentile_normalization(self, sample_image):
        """Test percentile normalization."""
        preprocessor = MedicalImagePreprocessor(
            normalization_method='percentile',
            percentile_range=(1.0, 99.0),
            intensity_range=(0.0, 1.0)
        )

        normalized = preprocessor.normalize(sample_image)

        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0

    def test_normalization_with_mask(self, sample_image, sample_mask):
        """Test normalization using ROI mask."""
        preprocessor = MedicalImagePreprocessor(
            normalization_method='min-max',
            intensity_range=(0.0, 1.0)
        )

        normalized = preprocessor.normalize(sample_image, mask=sample_mask)

        # Should normalize based on ROI only
        assert normalized.shape == sample_image.shape

    def test_intensity_windowing(self, sample_image):
        """Test intensity windowing."""
        windowed = MedicalImagePreprocessor.apply_intensity_windowing(
            sample_image,
            window_center=50.0,
            window_width=40.0
        )

        assert np.min(windowed) >= 0.0
        assert np.max(windowed) <= 1.0

    def test_intensity_windowing_auto(self, sample_image):
        """Test automatic intensity windowing."""
        windowed = MedicalImagePreprocessor.apply_intensity_windowing(sample_image)

        assert np.min(windowed) >= 0.0
        assert np.max(windowed) <= 1.0

    def test_pad_to_size(self):
        """Test padding to target size."""
        image = np.ones((100, 100), dtype=np.float32)
        padded = MedicalImagePreprocessor.pad_to_size(
            image,
            target_size=(128, 128),
            mode='constant',
            constant_value=0.0
        )

        assert padded.shape == (128, 128)
        assert padded[64, 64] == 1.0  # Center should be original
        assert padded[0, 0] == 0.0  # Corners should be padded

    def test_pad_to_size_3d(self):
        """Test padding 3D image."""
        image = np.ones((100, 100, 3), dtype=np.float32)
        padded = MedicalImagePreprocessor.pad_to_size(
            image,
            target_size=(128, 128),
            mode='constant'
        )

        assert padded.shape == (128, 128, 3)

    def test_crop_to_size_center(self):
        """Test center cropping."""
        image = np.ones((200, 200), dtype=np.float32)
        cropped = MedicalImagePreprocessor.crop_to_size(
            image,
            target_size=(128, 128),
            center_crop=True
        )

        assert cropped.shape == (128, 128)

    def test_crop_to_size_topleft(self):
        """Test top-left cropping."""
        image = np.ones((200, 200), dtype=np.float32)
        cropped = MedicalImagePreprocessor.crop_to_size(
            image,
            target_size=(128, 128),
            center_crop=False
        )

        assert cropped.shape == (128, 128)

    def test_no_normalization(self, sample_image):
        """Test with normalization disabled."""
        preprocessor = MedicalImagePreprocessor(normalization_method=None)

        result = preprocessor.normalize(sample_image)

        np.testing.assert_array_equal(result, sample_image)
