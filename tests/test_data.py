"""Unit tests for data loading and preprocessing."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

from med_seg.data import MedicalImageLoader, MedicalImagePreprocessor


class TestMedicalImageLoader:
    """Test medical image loading functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with sample images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "images"
            mask_dir = Path(tmpdir) / "masks"
            data_dir.mkdir()
            mask_dir.mkdir()

            # Create sample images
            for i in range(5):
                img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
                mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8)

                Image.fromarray(img).save(data_dir / f"image_{i}.png")
                Image.fromarray(mask * 255).save(mask_dir / f"image_{i}.png")

            yield data_dir, mask_dir

    def test_loader_init(self, temp_data_dir):
        """Test loader initialization."""
        data_dir, mask_dir = temp_data_dir

        loader = MedicalImageLoader(
            data_dir=str(data_dir),
            mask_dir=str(mask_dir)
        )

        assert loader.data_dir == data_dir
        assert loader.mask_dir == mask_dir

    def test_loader_get_image_paths(self, temp_data_dir):
        """Test getting image paths."""
        data_dir, _ = temp_data_dir

        loader = MedicalImageLoader(data_dir=str(data_dir))
        image_paths = loader.get_image_paths()

        assert len(image_paths) == 5

    def test_loader_load_2d_image(self, temp_data_dir):
        """Test loading a single 2D image."""
        data_dir, _ = temp_data_dir

        loader = MedicalImageLoader(data_dir=str(data_dir))
        image_paths = loader.get_image_paths()

        image = loader.load_2d_image(str(image_paths[0]))

        assert isinstance(image, np.ndarray)
        assert image.ndim == 2  # Grayscale image
        assert image.shape == (128, 128)

    def test_loader_load_dataset_2d(self, temp_data_dir):
        """Test loading entire 2D dataset."""
        data_dir, mask_dir = temp_data_dir

        loader = MedicalImageLoader(
            data_dir=str(data_dir),
            mask_dir=str(mask_dir)
        )

        images, masks = loader.load_dataset_2d()

        assert images.shape == (5, 128, 128)
        assert masks.shape == (5, 128, 128)

    def test_loader_max_samples(self, temp_data_dir):
        """Test loading with max_samples parameter."""
        data_dir, mask_dir = temp_data_dir

        loader = MedicalImageLoader(
            data_dir=str(data_dir),
            mask_dir=str(mask_dir)
        )

        images, masks = loader.load_dataset_2d(max_samples=3)

        assert images.shape == (3, 128, 128)
        assert masks.shape == (3, 128, 128)

    def test_loader_invalid_directory(self):
        """Test loader with invalid directory raises error."""
        with pytest.raises(ValueError):
            MedicalImageLoader(data_dir="/nonexistent/directory")


class TestMedicalImagePreprocessor:
    """Test medical image preprocessing functionality."""

    @pytest.fixture
    def sample_image(self):
        """Create sample image for testing."""
        return np.random.rand(128, 128).astype(np.float32) * 255

    @pytest.fixture
    def sample_mask(self):
        """Create sample mask for testing."""
        return np.random.randint(0, 2, (128, 128), dtype=np.uint8)

    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        preprocessor = MedicalImagePreprocessor(
            target_size=(256, 256),
            normalization_method="min-max"
        )

        assert preprocessor.target_size == (256, 256)
        assert preprocessor.normalization_method == "min-max"

    def test_normalize_min_max(self, sample_image):
        """Test min-max normalization."""
        preprocessor = MedicalImagePreprocessor(
            normalization_method="min-max"
        )

        normalized = preprocessor.normalize(sample_image)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32

    def test_normalize_z_score(self, sample_image):
        """Test z-score normalization."""
        preprocessor = MedicalImagePreprocessor(
            normalization_method="z-score"
        )

        normalized = preprocessor.normalize(sample_image)

        # Z-score should have mean ~0 and std ~1
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1

    def test_normalize_clip_range(self, sample_image):
        """Test normalization with clipping."""
        preprocessor = MedicalImagePreprocessor(
            normalization_method="min-max",
            clip_range=(50, 200)
        )

        normalized = preprocessor.normalize(sample_image)

        # All values should be within normalized range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_preprocess_adds_channel_dimension(self, sample_image):
        """Test that preprocessing adds channel dimension."""
        preprocessor = MedicalImagePreprocessor()

        processed_img, _ = preprocessor.preprocess(sample_image)

        assert processed_img.ndim == 3
        assert processed_img.shape[-1] == 1  # Single channel

    def test_preprocess_batch(self, sample_image, sample_mask):
        """Test batch preprocessing."""
        batch_size = 5
        images = np.stack([sample_image] * batch_size)
        masks = np.stack([sample_mask] * batch_size)

        preprocessor = MedicalImagePreprocessor(
            normalization_method="min-max"
        )

        proc_images, proc_masks = preprocessor.preprocess_batch(images, masks)

        assert proc_images.shape == (batch_size, 128, 128, 1)
        assert proc_masks.shape == (batch_size, 128, 128, 1)
        assert proc_images.min() >= 0.0
        assert proc_images.max() <= 1.0
