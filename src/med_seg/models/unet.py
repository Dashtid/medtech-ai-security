"""Standard U-Net architecture for 2D medical image segmentation.

Based on:
    Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
    Image Segmentation" (MICCAI 2015)
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class UNet:
    """U-Net model builder for 2D medical image segmentation.

    Args:
        input_size: Size of input images (assumes square images)
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes (1 for binary, >1 for multi-class)
        base_filters: Number of filters in first convolutional layer
        depth: Number of downsampling/upsampling levels
        use_batch_norm: Whether to use batch normalization
        use_dropout: Whether to use dropout
        dropout_rate: Dropout rate if use_dropout is True
    """

    def __init__(
        self,
        input_size: int = 256,
        input_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        use_batch_norm: bool = True,
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
    ):
        self.input_size = input_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.depth = depth
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

    def conv_block(
        self,
        x: tf.Tensor,
        filters: int,
        kernel_size: int = 3,
        activation: str = "relu",
    ) -> tf.Tensor:
        """Double convolution block (Conv -> BN -> Activation) x 2.

        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Convolution kernel size
            activation: Activation function

        Returns:
            Output tensor after double convolution
        """
        x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.Activation(activation)(x)

        x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.Activation(activation)(x)

        return x

    def encoder_block(
        self,
        x: tf.Tensor,
        filters: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encoder block: Conv block -> Max pooling.

        Args:
            x: Input tensor
            filters: Number of filters

        Returns:
            Tuple of (skip connection tensor, pooled tensor)
        """
        skip = self.conv_block(x, filters)
        x = layers.MaxPooling2D(pool_size=(2, 2))(skip)

        if self.use_dropout:
            x = layers.Dropout(self.dropout_rate)(x)

        return skip, x

    def decoder_block(
        self,
        x: tf.Tensor,
        skip: tf.Tensor,
        filters: int,
    ) -> tf.Tensor:
        """Decoder block: Upsampling -> Concatenate -> Conv block.

        Args:
            x: Input tensor from previous layer
            skip: Skip connection from encoder
            filters: Number of filters

        Returns:
            Output tensor after upsampling and convolution
        """
        x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)

        x = layers.concatenate([x, skip])
        x = self.conv_block(x, filters)

        return x

    def build(self) -> keras.Model:
        """Build the complete U-Net model.

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=(self.input_size, self.input_size, self.input_channels))

        # Encoder path
        skip_connections = []
        x = inputs

        for i in range(self.depth):
            filters = self.base_filters * (2**i)
            skip, x = self.encoder_block(x, filters)
            skip_connections.append(skip)

        # Bottleneck
        filters = self.base_filters * (2**self.depth)
        x = self.conv_block(x, filters)

        if self.use_dropout:
            x = layers.Dropout(self.dropout_rate)(x)

        # Decoder path
        for i in range(self.depth - 1, -1, -1):
            filters = self.base_filters * (2**i)
            skip = skip_connections[i]
            x = self.decoder_block(x, skip, filters)

        # Output layer
        if self.num_classes == 1:
            # Binary segmentation
            outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="output")(x)
        else:
            # Multi-class segmentation
            outputs = layers.Conv2D(
                self.num_classes, kernel_size=1, activation="softmax", name="output"
            )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="unet")

        return model
