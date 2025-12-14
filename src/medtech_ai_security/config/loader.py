"""
YAML Configuration Loader.

Provides utilities for loading, validating, and managing YAML configurations
with Pydantic models for type safety.

Usage:
    from medtech_ai_security.config import load_experiment_config

    config = load_experiment_config("experiment.yaml")
    print(f"Running {len(config.attacks)} attacks")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from medtech_ai_security.config.schema import (
    AttackConfig,
    DefenseConfig,
    ExperimentConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigError(Exception):
    """Configuration loading or validation error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigLoader:
    """
    YAML Configuration Loader with validation.

    Provides methods for loading and validating configuration files
    with helpful error messages and template generation.
    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        """
        Initialize the config loader.

        Args:
            config_dir: Default directory for configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()

    def load_yaml(self, path: str | Path) -> dict[str, Any]:
        """
        Load raw YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary with parsed YAML content

        Raises:
            ConfigError: If file not found or YAML parsing fails
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise ConfigError(
                f"Configuration file not found: {file_path}",
                {"path": str(file_path)},
            )

        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                return {}

            if not isinstance(data, dict):
                raise ConfigError(
                    f"YAML file must contain a dictionary, got {type(data).__name__}",
                    {"path": str(file_path)},
                )

            logger.debug(f"Loaded YAML from {file_path}")
            return data

        except yaml.YAMLError as e:
            raise ConfigError(
                f"Failed to parse YAML file: {e}",
                {"path": str(file_path), "error": str(e)},
            ) from e

    def load_attack(self, path: str | Path) -> AttackConfig:
        """
        Load and validate attack configuration.

        Args:
            path: Path to attack config YAML

        Returns:
            Validated AttackConfig

        Raises:
            ConfigError: If validation fails
        """
        data = self.load_yaml(path)
        return self._validate_model(AttackConfig, data, path)

    def load_defense(self, path: str | Path) -> DefenseConfig:
        """
        Load and validate defense configuration.

        Args:
            path: Path to defense config YAML

        Returns:
            Validated DefenseConfig

        Raises:
            ConfigError: If validation fails
        """
        data = self.load_yaml(path)
        return self._validate_model(DefenseConfig, data, path)

    def load_experiment(self, path: str | Path) -> ExperimentConfig:
        """
        Load and validate experiment configuration.

        Args:
            path: Path to experiment config YAML

        Returns:
            Validated ExperimentConfig

        Raises:
            ConfigError: If validation fails
        """
        data = self.load_yaml(path)
        return self._validate_model(ExperimentConfig, data, path)

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve path relative to config_dir if not absolute."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.config_dir / p

    def _validate_model(
        self,
        model_class: type[T],
        data: dict[str, Any],
        path: str | Path,
    ) -> T:
        """
        Validate data against Pydantic model.

        Args:
            model_class: Pydantic model class
            data: Dictionary to validate
            path: Original file path (for error messages)

        Returns:
            Validated model instance

        Raises:
            ConfigError: If validation fails
        """
        try:
            return model_class.model_validate(data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"  {loc}: {msg}")

            error_text = "\n".join(errors)
            raise ConfigError(
                f"Configuration validation failed for {path}:\n{error_text}",
                {"path": str(path), "errors": e.errors()},
            ) from e

    @staticmethod
    def save_yaml(config: Any, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Pydantic model or dictionary to save
            path: Output file path
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(config, "model_dump"):
            data = config.model_dump(mode="json")
        else:
            data = config

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {file_path}")

    @staticmethod
    def generate_attack_template() -> str:
        """Generate attack configuration template."""
        return """# Attack Configuration Template
# MedTech AI Security - Adversarial Attack

name: "fgsm_attack"
description: "Fast Gradient Sign Method attack"
version: "1.0.0"

attack_type: "fgsm"
epsilon: 0.3
norm: "Linf"
targeted: false

# For iterative attacks (PGD, C&W)
iterations: 40
step_size: null  # Auto-computed if null
random_start: true
num_restarts: 1

# For C&W attack
confidence: 0.0
learning_rate: 0.01
binary_search_steps: 9

# Optional: Model configuration
model:
  architecture: "resnet18"
  pretrained: true
  num_classes: 10
  input_shape: [3, 224, 224]
  device: "auto"

# Optional: Dataset configuration
dataset:
  type: "cifar10"
  root: "./data"
  download: true
  train: false
  batch_size: 32
"""

    @staticmethod
    def generate_defense_template() -> str:
        """Generate defense configuration template."""
        return """# Defense Configuration Template
# MedTech AI Security - Adversarial Defense

name: "randomized_smoothing"
description: "Randomized smoothing defense"
version: "1.0.0"

defense_type: "randomized_smoothing"
enabled: true

# Randomized smoothing parameters
noise_std: 0.25
num_samples: 100

# Input transformation parameters
jpeg_quality: 75
blur_sigma: 1.0
bit_depth: 4

# Adversarial training parameters
train_epsilon: 0.3
train_iterations: 7
train_ratio: 0.5

# Evaluation settings
evaluate_accuracy: true
evaluate_robustness: true
evaluation_attacks:
  - "fgsm"
  - "pgd"
"""

    @staticmethod
    def generate_experiment_template() -> str:
        """Generate experiment configuration template."""
        return """# Experiment Configuration Template
# MedTech AI Security - Complete Experiment

name: "adversarial_robustness_evaluation"
description: "Evaluate model robustness against multiple attacks"
version: "1.0.0"

# Reproducibility
seed: 42
deterministic: true

# Model configuration
model:
  architecture: "resnet18"
  pretrained: true
  num_classes: 10
  input_shape: [3, 224, 224]
  device: "auto"

# Dataset configuration
dataset:
  type: "cifar10"
  root: "./data"
  download: true
  train: false
  subset_size: 1000
  batch_size: 32
  num_workers: 4
  normalize: true

# Attack configurations
attacks:
  - name: "fgsm_attack"
    attack_type: "fgsm"
    epsilon: 0.3
    norm: "Linf"

  - name: "pgd_attack"
    attack_type: "pgd"
    epsilon: 0.3
    norm: "Linf"
    iterations: 40
    step_size: 0.01
    random_start: true

  - name: "cw_attack"
    attack_type: "cw"
    epsilon: 0.3
    norm: "L2"
    iterations: 100
    learning_rate: 0.01
    confidence: 0.0

# Defense configurations
defenses:
  - name: "jpeg_defense"
    defense_type: "jpeg_compression"
    enabled: true
    jpeg_quality: 75

  - name: "smoothing_defense"
    defense_type: "randomized_smoothing"
    enabled: true
    noise_std: 0.25
    num_samples: 100

# Output configuration
output:
  output_dir: "./results"
  save_adversarial: false
  save_perturbations: false
  save_metrics: true
  log_level: "INFO"
  tensorboard: false
  wandb: false

# Execution settings
device: "auto"
num_workers: 4
verbose: true
"""


# =============================================================================
# Convenience Functions
# =============================================================================


def load_attack_config(path: str | Path) -> AttackConfig:
    """
    Load attack configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Validated AttackConfig

    Raises:
        ConfigError: If loading or validation fails
    """
    loader = ConfigLoader()
    return loader.load_attack(path)


def load_defense_config(path: str | Path) -> DefenseConfig:
    """
    Load defense configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Validated DefenseConfig

    Raises:
        ConfigError: If loading or validation fails
    """
    loader = ConfigLoader()
    return loader.load_defense(path)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Validated ExperimentConfig

    Raises:
        ConfigError: If loading or validation fails
    """
    loader = ConfigLoader()
    return loader.load_experiment(path)


def validate_config(data: dict[str, Any], config_type: str = "experiment") -> Any:
    """
    Validate configuration dictionary.

    Args:
        data: Configuration dictionary
        config_type: Type of configuration ("attack", "defense", "experiment")

    Returns:
        Validated configuration object

    Raises:
        ConfigError: If validation fails
        ValueError: If config_type is invalid
    """
    config_classes = {
        "attack": AttackConfig,
        "defense": DefenseConfig,
        "experiment": ExperimentConfig,
    }

    if config_type not in config_classes:
        raise ValueError(f"Invalid config_type: {config_type}")

    model_class = config_classes[config_type]

    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        raise ConfigError(
            f"Validation failed: {e}",
            {"errors": e.errors()},
        ) from e
