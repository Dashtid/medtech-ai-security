"""Unit tests for model architectures."""

import pytest
import numpy as np
from tensorflow import keras

from med_seg.models import (
    UNet,
    UNetDeep,
    UNetDeepSpatialDropout,
    UNetLSTM,
    UNetDeepLSTM
)


class TestUNetModels:
    """Test U-Net model building and basic properties."""

    @pytest.fixture
    def input_params(self):
        """Common input parameters for all models."""
        return {
            'input_size': 256,
            'input_channels': 1,
            'base_filters': 16,
            'use_batch_norm': True,
            'use_dropout': True,
            'dropout_rate': 0.5
        }

    def test_unet_build(self, input_params):
        """Test standard U-Net can be built."""
        model_builder = UNet(**input_params)
        model = model_builder.build()

        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 256, 256, 1)
        assert model.output_shape == (None, 256, 256, 1)

    def test_unet_deep_build(self, input_params):
        """Test deep U-Net can be built."""
        model_builder = UNetDeep(**input_params)
        model = model_builder.build()

        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 256, 256, 1)
        assert model.output_shape == (None, 256, 256, 1)

    def test_unet_deep_spatial_dropout_build(self, input_params):
        """Test deep U-Net with spatial dropout can be built."""
        model_builder = UNetDeepSpatialDropout(**input_params)
        model = model_builder.build()

        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 256, 256, 1)
        assert model.output_shape == (None, 256, 256, 1)

    def test_unet_lstm_build(self, input_params):
        """Test U-Net with LSTM can be built."""
        model_builder = UNetLSTM(**input_params)
        model = model_builder.build()

        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 256, 256, 1)
        assert model.output_shape == (None, 256, 256, 1)

    def test_unet_deep_lstm_build(self, input_params):
        """Test deep U-Net with LSTM can be built."""
        model_builder = UNetDeepLSTM(**input_params)
        model = model_builder.build()

        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 256, 256, 1)
        assert model.output_shape == (None, 256, 256, 1)

    def test_multi_channel_input(self):
        """Test model works with multi-channel input."""
        model_builder = UNet(
            input_size=256,
            input_channels=4,
            base_filters=16
        )
        model = model_builder.build()

        assert model.input_shape == (None, 256, 256, 4)
        assert model.output_shape == (None, 256, 256, 1)

    def test_model_prediction(self, input_params):
        """Test model can make predictions."""
        model_builder = UNet(**input_params)
        model = model_builder.build()

        # Create dummy input
        dummy_input = np.random.rand(2, 256, 256, 1).astype(np.float32)

        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)

        assert prediction.shape == (2, 256, 256, 1)
        assert np.all(prediction >= 0) and np.all(prediction <= 1)  # Sigmoid output

    def test_different_image_sizes(self):
        """Test models work with different image sizes."""
        for size in [128, 256, 512]:
            model_builder = UNet(input_size=size, base_filters=8)
            model = model_builder.build()

            assert model.input_shape == (None, size, size, 1)
            assert model.output_shape == (None, size, size, 1)

    def test_model_without_batch_norm(self):
        """Test model building without batch normalization."""
        model_builder = UNet(
            input_size=256,
            input_channels=1,
            base_filters=16,
            use_batch_norm=False
        )
        model = model_builder.build()

        assert isinstance(model, keras.Model)

    def test_model_without_dropout(self):
        """Test model building without dropout."""
        model_builder = UNet(
            input_size=256,
            input_channels=1,
            base_filters=16,
            use_dropout=False
        )
        model = model_builder.build()

        assert isinstance(model, keras.Model)
