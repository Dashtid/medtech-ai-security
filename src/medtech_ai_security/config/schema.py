"""
Pydantic Schema for YAML Configuration.

Provides type-safe configuration models for adversarial ML experiments
with comprehensive validation and helpful error messages.

Usage:
    from medtech_ai_security.config import AttackConfig, load_attack_config

    config = load_attack_config("attack.yaml")
    print(config.attack_type)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================


class AttackType(str, Enum):
    """Supported adversarial attack types."""

    FGSM = "fgsm"
    PGD = "pgd"
    CW = "cw"
    DEEPFOOL = "deepfool"
    SQUARE = "square"
    AUTOATTACK = "autoattack"
    BOUNDARY = "boundary"
    HOPSKIPJUMP = "hopskipjump"


class DefenseType(str, Enum):
    """Supported defense types."""

    ADVERSARIAL_TRAINING = "adversarial_training"
    RANDOMIZED_SMOOTHING = "randomized_smoothing"
    INPUT_TRANSFORMATION = "input_transformation"
    FEATURE_SQUEEZING = "feature_squeezing"
    JPEG_COMPRESSION = "jpeg_compression"
    GAUSSIAN_BLUR = "gaussian_blur"
    BIT_DEPTH_REDUCTION = "bit_depth_reduction"
    GRADIENT_REGULARIZATION = "gradient_regularization"
    ENSEMBLE = "ensemble"


class NormType(str, Enum):
    """Perturbation norm types."""

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    LINF = "Linf"


class DatasetType(str, Enum):
    """Supported dataset types."""

    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMAGENET = "imagenet"
    MEDICAL = "medical"
    CUSTOM = "custom"


class DeviceType(str, Enum):
    """Compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


# =============================================================================
# Base Configuration
# =============================================================================


class BaseConfig(BaseModel):
    """Base configuration with common settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="unnamed",
        description="Configuration name for identification",
        min_length=1,
        max_length=100,
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the configuration",
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration schema version",
        pattern=r"^\d+\.\d+\.\d+$",
    )


# =============================================================================
# Model Configuration
# =============================================================================


class ModelConfig(BaseModel):
    """Configuration for target model."""

    architecture: str = Field(
        ...,
        description="Model architecture (e.g., 'resnet18', 'vgg16')",
        min_length=1,
    )
    pretrained: bool = Field(
        default=True,
        description="Whether to use pretrained weights",
    )
    num_classes: int = Field(
        default=10,
        description="Number of output classes",
        ge=2,
        le=10000,
    )
    input_shape: list[int] = Field(
        default=[3, 224, 224],
        description="Input tensor shape [C, H, W]",
        min_length=3,
        max_length=3,
    )
    weights_path: str | None = Field(
        default=None,
        description="Path to custom model weights",
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Compute device to use",
    )

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, v: list[int]) -> list[int]:
        """Validate input shape dimensions."""
        if any(dim <= 0 for dim in v):
            raise ValueError("All dimensions must be positive")
        if v[0] not in [1, 3, 4]:  # Grayscale, RGB, or RGBA
            raise ValueError("Channel dimension must be 1, 3, or 4")
        return v


# =============================================================================
# Dataset Configuration
# =============================================================================


class DatasetConfig(BaseModel):
    """Configuration for dataset."""

    type: DatasetType = Field(
        default=DatasetType.CIFAR10,
        description="Dataset type",
    )
    root: str = Field(
        default="./data",
        description="Root directory for dataset",
    )
    download: bool = Field(
        default=True,
        description="Whether to download if not present",
    )
    train: bool = Field(
        default=False,
        description="Whether to use training split",
    )
    subset_size: int | None = Field(
        default=None,
        description="Number of samples to use (None for all)",
        ge=1,
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for data loading",
        ge=1,
        le=1024,
    )
    num_workers: int = Field(
        default=4,
        description="Number of data loading workers",
        ge=0,
        le=32,
    )
    normalize: bool = Field(
        default=True,
        description="Whether to apply normalization",
    )
    custom_path: str | None = Field(
        default=None,
        description="Path to custom dataset (for type='custom')",
    )

    @model_validator(mode="after")
    def validate_custom_dataset(self) -> "DatasetConfig":
        """Validate custom dataset configuration."""
        if self.type == DatasetType.CUSTOM and not self.custom_path:
            raise ValueError("custom_path is required when type is 'custom'")
        return self


# =============================================================================
# Attack Configuration
# =============================================================================


class AttackConfig(BaseConfig):
    """Configuration for adversarial attacks."""

    attack_type: AttackType = Field(
        ...,
        description="Type of adversarial attack",
    )
    epsilon: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.3,
        description="Maximum perturbation magnitude",
    )
    norm: NormType = Field(
        default=NormType.LINF,
        description="Perturbation norm type",
    )
    targeted: bool = Field(
        default=False,
        description="Whether attack is targeted",
    )
    target_class: int | None = Field(
        default=None,
        description="Target class for targeted attacks",
        ge=0,
    )

    # PGD-specific parameters
    iterations: int = Field(
        default=40,
        description="Number of attack iterations (for iterative attacks)",
        ge=1,
        le=1000,
    )
    step_size: float | None = Field(
        default=None,
        description="Step size per iteration (auto-computed if None)",
        ge=0.0,
    )
    random_start: bool = Field(
        default=True,
        description="Whether to use random initialization",
    )
    num_restarts: int = Field(
        default=1,
        description="Number of random restarts",
        ge=1,
        le=100,
    )

    # C&W-specific parameters
    confidence: float = Field(
        default=0.0,
        description="Confidence parameter for C&W attack",
        ge=0.0,
    )
    learning_rate: float = Field(
        default=0.01,
        description="Learning rate for optimization-based attacks",
        gt=0.0,
        le=1.0,
    )
    binary_search_steps: int = Field(
        default=9,
        description="Binary search steps for C&W",
        ge=1,
        le=20,
    )

    # Model and dataset
    model: ModelConfig | None = Field(
        default=None,
        description="Target model configuration",
    )
    dataset: DatasetConfig | None = Field(
        default=None,
        description="Dataset configuration",
    )

    @model_validator(mode="after")
    def validate_targeted(self) -> "AttackConfig":
        """Validate targeted attack configuration."""
        if self.targeted and self.target_class is None:
            raise ValueError("target_class is required for targeted attacks")
        return self

    @model_validator(mode="after")
    def compute_step_size(self) -> "AttackConfig":
        """Auto-compute step size if not provided."""
        if self.step_size is None and self.iterations > 0:
            # Default step size: 2.5 * epsilon / iterations
            object.__setattr__(
                self,
                "step_size",
                2.5 * self.epsilon / self.iterations,
            )
        return self


# =============================================================================
# Defense Configuration
# =============================================================================


class DefenseConfig(BaseConfig):
    """Configuration for adversarial defenses."""

    defense_type: DefenseType = Field(
        ...,
        description="Type of defense",
    )
    enabled: bool = Field(
        default=True,
        description="Whether defense is enabled",
    )

    # Randomized smoothing parameters
    noise_std: float = Field(
        default=0.25,
        description="Standard deviation of Gaussian noise",
        ge=0.0,
        le=1.0,
    )
    num_samples: int = Field(
        default=100,
        description="Number of samples for Monte Carlo estimation",
        ge=1,
        le=10000,
    )

    # Input transformation parameters
    jpeg_quality: int = Field(
        default=75,
        description="JPEG compression quality",
        ge=1,
        le=100,
    )
    blur_sigma: float = Field(
        default=1.0,
        description="Gaussian blur sigma",
        ge=0.0,
        le=10.0,
    )
    bit_depth: int = Field(
        default=4,
        description="Bit depth for reduction",
        ge=1,
        le=8,
    )

    # Adversarial training parameters
    train_epsilon: float = Field(
        default=0.3,
        description="Epsilon for adversarial training",
        ge=0.0,
        le=1.0,
    )
    train_iterations: int = Field(
        default=7,
        description="Attack iterations during training",
        ge=1,
        le=100,
    )
    train_ratio: float = Field(
        default=0.5,
        description="Ratio of adversarial examples in batch",
        ge=0.0,
        le=1.0,
    )

    # Evaluation metrics
    evaluate_accuracy: bool = Field(
        default=True,
        description="Evaluate clean accuracy",
    )
    evaluate_robustness: bool = Field(
        default=True,
        description="Evaluate adversarial robustness",
    )
    evaluation_attacks: list[AttackType] = Field(
        default_factory=lambda: [AttackType.FGSM, AttackType.PGD],
        description="Attacks to use for robustness evaluation",
    )


# =============================================================================
# Output Configuration
# =============================================================================


class OutputConfig(BaseModel):
    """Configuration for output and logging."""

    output_dir: str = Field(
        default="./results",
        description="Directory for output files",
    )
    save_adversarial: bool = Field(
        default=False,
        description="Whether to save adversarial examples",
    )
    save_perturbations: bool = Field(
        default=False,
        description="Whether to save perturbations",
    )
    save_metrics: bool = Field(
        default=True,
        description="Whether to save evaluation metrics",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    tensorboard: bool = Field(
        default=False,
        description="Enable TensorBoard logging",
    )
    wandb: bool = Field(
        default=False,
        description="Enable Weights & Biases logging",
    )
    wandb_project: str | None = Field(
        default=None,
        description="W&B project name",
    )


# =============================================================================
# Experiment Configuration
# =============================================================================


class ExperimentConfig(BaseConfig):
    """Complete experiment configuration."""

    # Reproducibility
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
        ge=0,
    )
    deterministic: bool = Field(
        default=True,
        description="Enable deterministic operations",
    )

    # Components
    model: ModelConfig = Field(
        ...,
        description="Model configuration",
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description="Dataset configuration",
    )
    attacks: list[AttackConfig] = Field(
        default_factory=list,
        description="List of attack configurations",
    )
    defenses: list[DefenseConfig] = Field(
        default_factory=list,
        description="List of defense configurations",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration",
    )

    # Execution settings
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Compute device",
    )
    num_workers: int = Field(
        default=4,
        description="Number of parallel workers",
        ge=0,
        le=32,
    )
    verbose: bool = Field(
        default=True,
        description="Enable verbose output",
    )

    @model_validator(mode="after")
    def validate_experiment(self) -> "ExperimentConfig":
        """Validate experiment configuration."""
        if not self.attacks and not self.defenses:
            raise ValueError("At least one attack or defense must be specified")
        return self

    def get_attack_by_type(self, attack_type: AttackType) -> AttackConfig | None:
        """Get attack configuration by type."""
        for attack in self.attacks:
            if attack.attack_type == attack_type:
                return attack
        return None

    def get_defense_by_type(self, defense_type: DefenseType) -> DefenseConfig | None:
        """Get defense configuration by type."""
        for defense in self.defenses:
            if defense.defense_type == defense_type:
                return defense
        return None
