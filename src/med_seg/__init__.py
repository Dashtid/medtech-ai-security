"""Medical Image Segmentation package.

A professional implementation of U-Net architectures for medical image
segmentation.
"""

__version__ = "1.0.0"
__author__ = "David Dashti"

from med_seg.models.unet import UNet
from med_seg.data.loader import MedicalImageLoader
from med_seg.data.preprocessor import MedicalImagePreprocessor
from med_seg.training.losses import dice_loss, dice_coefficient
from med_seg.training.trainer import ModelTrainer

__all__ = [
    "UNet",
    "MedicalImageLoader",
    "MedicalImagePreprocessor",
    "dice_loss",
    "dice_coefficient",
    "ModelTrainer",
]
