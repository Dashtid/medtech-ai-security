"""Multi-task U-Net for segmentation and survival prediction.

This module implements a multi-task U-Net architecture with:
1. Shared encoder (feature extraction)
2. Segmentation decoder (spatial predictions)
3. Survival prediction head (global outcome prediction)
4. Monte Carlo Dropout for uncertainty quantification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


class MCDropout(layers.Dropout):
    """Monte Carlo Dropout layer.

    Unlike standard Dropout, this layer remains active during inference,
    allowing uncertainty quantification through multiple forward passes.
    """

    def call(self, inputs, training=None):
        # Always apply dropout, even at inference time
        return super().call(inputs, training=True)


class MultiTaskUNet:
    """Multi-task U-Net for segmentation and survival prediction.

    Architecture:
        Input (H×W×C)
            ↓
        Shared Encoder (contracting path with skip connections)
            ↓
            ├─→ Segmentation Decoder → Tumor Mask (H×W×1)
            │   (with MC Dropout for uncertainty)
            │
            └─→ Survival Prediction Head → Risk Score (scalar)
                ├─→ Global Average Pooling
                ├─→ Dense layers with dropout
                └─→ Linear output (log-hazard)

    The shared encoder allows the model to learn features useful for both tasks,
    while task-specific heads enable specialization.
    """

    def __init__(
        self,
        input_size: int = 256,
        input_channels: int = 2,
        num_classes: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3,
        survival_hidden_units: Tuple[int, ...] = (256, 128, 64),
    ):
        """Initialize Multi-Task U-Net.

        Args:
            input_size: Input image size (square)
            input_channels: Number of input channels (e.g., 2 for PET+CT)
            num_classes: Number of segmentation classes (1 for binary)
            base_filters: Number of filters in first conv layer
            depth: Number of downsampling/upsampling levels
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for MC Dropout and survival head
            survival_hidden_units: Hidden layer sizes for survival head
        """
        self.input_size = input_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.depth = depth
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.survival_hidden_units = survival_hidden_units

    def _conv_block(self, x: tf.Tensor, filters: int, name_prefix: str) -> tf.Tensor:
        """Convolutional block (Conv → BN → ReLU → Conv → BN → ReLU).

        Args:
            x: Input tensor
            filters: Number of filters
            name_prefix: Prefix for layer names

        Returns:
            Output tensor
        """
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

        x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv2")(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)

        return x

    def build(self) -> keras.Model:
        """Build multi-task U-Net model.

        Returns:
            Keras Model with two outputs: (segmentation_mask, survival_risk)
        """
        # Input
        inputs = layers.Input(
            shape=(self.input_size, self.input_size, self.input_channels), name="input_petct"
        )

        # =====================================================================
        # SHARED ENCODER (Contracting Path)
        # =====================================================================
        encoder_outputs = []
        x = inputs

        for i in range(self.depth):
            filters = self.base_filters * (2**i)
            x = self._conv_block(x, filters, f"encoder_block{i+1}")
            encoder_outputs.append(x)

            if i < self.depth - 1:  # No pooling on last block
                x = layers.MaxPooling2D(2, name=f"encoder_pool{i+1}")(x)

        # Bottleneck
        filters = self.base_filters * (2**self.depth)
        bottleneck = self._conv_block(x, filters, "bottleneck")

        # =====================================================================
        # TASK 1: SEGMENTATION DECODER (Expanding Path with MC Dropout)
        # =====================================================================
        x_seg = bottleneck

        for i in range(self.depth - 1, -1, -1):
            filters = self.base_filters * (2**i)

            # Upsampling (skip for first iteration since bottleneck is same size as last encoder output)
            if i < self.depth - 1:
                x_seg = layers.Conv2DTranspose(
                    filters, 2, strides=2, padding="same", name=f"seg_upsample{i+1}"
                )(x_seg)

            # Skip connection from encoder
            x_seg = layers.Concatenate(name=f"seg_concat{i+1}")([x_seg, encoder_outputs[i]])

            # Convolutional block
            x_seg = self._conv_block(x_seg, filters, f"seg_decoder{i+1}")

            # MC Dropout for uncertainty quantification
            if self.dropout_rate > 0:
                x_seg = MCDropout(self.dropout_rate, name=f"seg_mc_dropout{i+1}")(x_seg)

        # Segmentation output
        segmentation_output = layers.Conv2D(
            self.num_classes, 1, activation="sigmoid", name="segmentation_output"
        )(x_seg)

        # =====================================================================
        # TASK 2: SURVIVAL PREDICTION HEAD
        # =====================================================================
        x_surv = bottleneck

        # Global average pooling to aggregate spatial information
        x_surv = layers.GlobalAveragePooling2D(name="surv_global_pool")(x_surv)

        # Dense layers with dropout
        for i, units in enumerate(self.survival_hidden_units):
            x_surv = layers.Dense(units, name=f"surv_dense{i+1}")(x_surv)
            x_surv = layers.BatchNormalization(name=f"surv_bn{i+1}")(x_surv)
            x_surv = layers.Activation("relu", name=f"surv_relu{i+1}")(x_surv)
            x_surv = layers.Dropout(self.dropout_rate, name=f"surv_dropout{i+1}")(x_surv)

        # Survival output (log-hazard, linear activation)
        # Single neuron predicting risk score for Cox proportional hazards
        survival_output = layers.Dense(1, activation="linear", name="survival_output")(x_surv)

        # =====================================================================
        # CREATE MULTI-OUTPUT MODEL
        # =====================================================================
        model = keras.Model(
            inputs=inputs,
            outputs={"segmentation": segmentation_output, "survival": survival_output},
            name="multitask_unet",
        )

        return model

    def build_single_task_segmentation(self) -> keras.Model:
        """Build single-task segmentation model (for comparison).

        Returns:
            Keras Model with only segmentation output
        """
        # Build multi-task model
        multitask_model = self.build()

        # Extract only segmentation output
        single_task_model = keras.Model(
            inputs=multitask_model.input,
            outputs=multitask_model.outputs["segmentation"],
            name="unet_segmentation_only",
        )

        return single_task_model


def create_multitask_unet(
    input_size: int = 256,
    input_channels: int = 2,
    base_filters: int = 64,
    depth: int = 4,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Convenience function to create multi-task U-Net.

    Args:
        input_size: Input image size
        input_channels: Number of input channels
        base_filters: Base number of filters
        depth: Network depth
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    builder = MultiTaskUNet(
        input_size=input_size,
        input_channels=input_channels,
        base_filters=base_filters,
        depth=depth,
        dropout_rate=dropout_rate,
    )

    model = builder.build()
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating Multi-Task U-Net...")
    model = create_multitask_unet(
        input_size=256, input_channels=2, base_filters=64, depth=4, dropout_rate=0.3
    )

    print("\nModel created successfully!")
    model.summary()

    print("\nOutput shapes:")
    print(f"  Segmentation: {model.outputs['segmentation'].shape}")
    print(f"  Survival: {model.outputs['survival'].shape}")

    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")
