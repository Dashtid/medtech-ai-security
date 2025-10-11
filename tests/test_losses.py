"""Unit tests for loss functions."""

import pytest
import numpy as np
import tensorflow as tf

from med_seg.training.losses import (
    dice_coefficient,
    dice_loss,
    binary_crossentropy_loss,
    combined_loss,
    focal_loss,
    tversky_loss
)


class TestLossFunctions:
    """Test loss functions and metrics."""

    @pytest.fixture
    def perfect_prediction(self):
        """Perfect prediction (DICE = 1.0)."""
        y_true = tf.constant([[[[1.0]], [[1.0]], [[0.0]], [[0.0]]]], dtype=tf.float32)
        y_pred = tf.constant([[[[1.0]], [[1.0]], [[0.0]], [[0.0]]]], dtype=tf.float32)
        return y_true, y_pred

    @pytest.fixture
    def random_prediction(self):
        """Random prediction."""
        np.random.seed(42)
        y_true = tf.constant(np.random.rand(2, 4, 4, 1), dtype=tf.float32)
        y_pred = tf.constant(np.random.rand(2, 4, 4, 1), dtype=tf.float32)
        return y_true, y_pred

    def test_dice_coefficient_perfect(self, perfect_prediction):
        """Test DICE coefficient with perfect prediction."""
        y_true, y_pred = perfect_prediction
        dice = dice_coefficient(y_true, y_pred)

        assert float(dice) > 0.99  # Should be very close to 1.0

    def test_dice_coefficient_range(self, random_prediction):
        """Test DICE coefficient is in valid range [0, 1]."""
        y_true, y_pred = random_prediction
        dice = dice_coefficient(y_true, y_pred)

        assert 0.0 <= float(dice) <= 1.0

    def test_dice_loss_perfect(self, perfect_prediction):
        """Test DICE loss with perfect prediction."""
        y_true, y_pred = perfect_prediction
        loss = dice_loss(y_true, y_pred)

        assert float(loss) < 0.01  # Should be very close to 0.0

    def test_dice_loss_range(self, random_prediction):
        """Test DICE loss is in valid range [0, 1]."""
        y_true, y_pred = random_prediction
        loss = dice_loss(y_true, y_pred)

        assert 0.0 <= float(loss) <= 1.0

    def test_dice_loss_inverse_of_coefficient(self, random_prediction):
        """Test DICE loss = 1 - DICE coefficient."""
        y_true, y_pred = random_prediction

        dice = dice_coefficient(y_true, y_pred)
        loss = dice_loss(y_true, y_pred)

        assert np.isclose(float(dice) + float(loss), 1.0, atol=1e-6)

    def test_binary_crossentropy_loss(self, random_prediction):
        """Test binary cross-entropy loss."""
        y_true, y_pred = random_prediction
        loss = binary_crossentropy_loss(y_true, y_pred)

        assert float(loss) >= 0.0  # BCE is always non-negative

    def test_combined_loss_default_weights(self, random_prediction):
        """Test combined loss with default weights."""
        y_true, y_pred = random_prediction

        loss_fn = combined_loss(dice_weight=0.5, bce_weight=0.5)
        loss = loss_fn(y_true, y_pred)

        assert float(loss) >= 0.0

    def test_combined_loss_custom_weights(self, random_prediction):
        """Test combined loss with custom weights."""
        y_true, y_pred = random_prediction

        loss_fn = combined_loss(dice_weight=0.7, bce_weight=0.3)
        loss = loss_fn(y_true, y_pred)

        assert float(loss) >= 0.0

    def test_focal_loss(self, random_prediction):
        """Test focal loss."""
        y_true, y_pred = random_prediction

        loss_fn = focal_loss(alpha=0.25, gamma=2.0)
        loss = loss_fn(y_true, y_pred)

        assert float(loss) >= 0.0

    def test_tversky_loss_default(self, random_prediction):
        """Test Tversky loss with default parameters."""
        y_true, y_pred = random_prediction

        loss_fn = tversky_loss(alpha=0.5, beta=0.5)
        loss = loss_fn(y_true, y_pred)

        assert 0.0 <= float(loss) <= 1.0

    def test_tversky_loss_recall_focus(self, random_prediction):
        """Test Tversky loss focusing on recall."""
        y_true, y_pred = random_prediction

        # Higher beta emphasizes false negatives (recall)
        loss_fn = tversky_loss(alpha=0.3, beta=0.7)
        loss = loss_fn(y_true, y_pred)

        assert 0.0 <= float(loss) <= 1.0

    def test_tversky_loss_precision_focus(self, random_prediction):
        """Test Tversky loss focusing on precision."""
        y_true, y_pred = random_prediction

        # Higher alpha emphasizes false positives (precision)
        loss_fn = tversky_loss(alpha=0.7, beta=0.3)
        loss = loss_fn(y_true, y_pred)

        assert 0.0 <= float(loss) <= 1.0

    def test_loss_with_zeros(self):
        """Test losses handle all-zero predictions."""
        y_true = tf.zeros((1, 4, 4, 1), dtype=tf.float32)
        y_pred = tf.zeros((1, 4, 4, 1), dtype=tf.float32)

        dice = dice_coefficient(y_true, y_pred)
        loss = dice_loss(y_true, y_pred)

        # Should handle division by zero gracefully
        assert not np.isnan(float(dice))
        assert not np.isnan(float(loss))

    def test_loss_with_ones(self):
        """Test losses handle all-one predictions."""
        y_true = tf.ones((1, 4, 4, 1), dtype=tf.float32)
        y_pred = tf.ones((1, 4, 4, 1), dtype=tf.float32)

        dice = dice_coefficient(y_true, y_pred)
        loss = dice_loss(y_true, y_pred)

        assert float(dice) > 0.99  # Perfect match
        assert float(loss) < 0.01
