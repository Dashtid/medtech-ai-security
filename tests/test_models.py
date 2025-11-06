"""Unit tests for U-Net model architecture."""

import pytest
import numpy as np
import tensorflow as tf

from med_seg.models import UNet


class TestUNet:
    """Test U-Net model building and properties."""

    def test_unet_build_default(self):
        """Test building U-Net with default parameters."""
        model_builder = UNet()
        model = model_builder.build()

        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert model.name == "unet"

    def test_unet_input_shape(self):
        """Test U-Net input shape is correct."""
        model_builder = UNet(input_size=128, input_channels=3)
        model = model_builder.build()

        expected_shape = (None, 128, 128, 3)
        assert model.input_shape == expected_shape

    def test_unet_output_shape_binary(self):
        """Test U-Net output shape for binary segmentation."""
        model_builder = UNet(input_size=256, num_classes=1)
        model = model_builder.build()

        expected_shape = (None, 256, 256, 1)
        assert model.output_shape == expected_shape

    def test_unet_output_shape_multiclass(self):
        """Test U-Net output shape for multi-class segmentation."""
        model_builder = UNet(input_size=256, num_classes=5)
        model = model_builder.build()

        expected_shape = (None, 256, 256, 5)
        assert model.output_shape == expected_shape

    def test_unet_prediction(self):
        """Test U-Net can make predictions."""
        model_builder = UNet(input_size=64, base_filters=16)
        model = model_builder.build()

        # Create dummy input
        batch_size = 2
        dummy_input = np.random.rand(batch_size, 64, 64, 1).astype(np.float32)

        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)

        assert prediction.shape == (batch_size, 64, 64, 1)
        assert prediction.min() >= 0.0
        assert prediction.max() <= 1.0  # Sigmoid output

    def test_unet_with_batch_norm(self):
        """Test U-Net with batch normalization enabled."""
        model_builder = UNet(use_batch_norm=True)
        model = model_builder.build()

        # Check that BatchNormalization layers exist
        batch_norm_layers = [
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
        ]
        assert len(batch_norm_layers) > 0

    def test_unet_with_dropout(self):
        """Test U-Net with dropout enabled."""
        model_builder = UNet(use_dropout=True, dropout_rate=0.3)
        model = model_builder.build()

        # Check that Dropout layers exist
        dropout_layers = [
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
        ]
        assert len(dropout_layers) > 0

    def test_unet_depth(self):
        """Test U-Net with different depth values."""
        for depth in [3, 4, 5]:
            model_builder = UNet(depth=depth, base_filters=16)
            model = model_builder.build()

            # Model should build successfully
            assert model is not None
            assert isinstance(model, tf.keras.Model)

    @pytest.mark.parametrize("input_size", [64, 128, 256, 512])
    def test_unet_different_sizes(self, input_size):
        """Test U-Net with different input sizes."""
        model_builder = UNet(input_size=input_size, base_filters=16)
        model = model_builder.build()

        assert model.input_shape == (None, input_size, input_size, 1)
        assert model.output_shape == (None, input_size, input_size, 1)

    def test_unet_parameter_count(self):
        """Test U-Net has reasonable parameter count."""
        model_builder = UNet(input_size=256, base_filters=64)
        model = model_builder.build()

        total_params = model.count_params()

        # U-Net should have millions of parameters
        assert total_params > 1_000_000
        assert total_params < 100_000_000  # But not too many
