"""
YAML Configuration System for MedTech AI Security.

This package provides Pydantic-based YAML configuration for:
- Attack configurations (FGSM, PGD, C&W, etc.)
- Defense configurations
- Experiment workflows
- Reproducible research settings
"""

from medtech_ai_security.config.loader import (
    ConfigLoader,
    load_attack_config,
    load_defense_config,
    load_experiment_config,
)
from medtech_ai_security.config.schema import (
    AttackConfig,
    AttackType,
    DatasetConfig,
    DatasetType,
    DefenseConfig,
    DefenseType,
    DeviceType,
    ExperimentConfig,
    ModelConfig,
    NormType,
    OutputConfig,
)

__all__ = [
    # Schema - Enums
    "AttackType",
    "DatasetType",
    "DefenseType",
    "DeviceType",
    "NormType",
    # Schema - Models
    "AttackConfig",
    "DatasetConfig",
    "DefenseConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OutputConfig",
    # Loader
    "ConfigLoader",
    "load_attack_config",
    "load_defense_config",
    "load_experiment_config",
]
