"""
Tests for the YAML Configuration module.

Tests cover:
- Pydantic schema validation
- YAML loading and parsing
- Config template generation
- Error handling
- Type coercion

"""

import tempfile
from pathlib import Path

import pytest
import yaml

from medtech_ai_security.config import (
    AttackConfig,
    AttackType,
    ConfigLoader,
    DatasetConfig,
    DatasetType,
    DefenseConfig,
    DefenseType,
    DeviceType,
    ExperimentConfig,
    ModelConfig,
    NormType,
    OutputConfig,
    load_attack_config,
    load_defense_config,
    load_experiment_config,
)
from medtech_ai_security.config.loader import ConfigError, validate_config


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_attack_yaml():
    """Valid attack configuration YAML."""
    return """
name: "test_fgsm_attack"
description: "Test FGSM attack"
version: "1.0.0"
attack_type: "fgsm"
epsilon: 0.3
norm: "Linf"
targeted: false
iterations: 40
"""


@pytest.fixture
def valid_defense_yaml():
    """Valid defense configuration YAML."""
    return """
name: "test_defense"
description: "Test defense"
version: "1.0.0"
defense_type: "randomized_smoothing"
enabled: true
noise_std: 0.25
num_samples: 100
"""


@pytest.fixture
def valid_experiment_yaml():
    """Valid experiment configuration YAML."""
    return """
name: "test_experiment"
description: "Test experiment"
version: "1.0.0"
seed: 42
deterministic: true

model:
  architecture: "resnet18"
  pretrained: true
  num_classes: 10
  input_shape: [3, 224, 224]
  device: "auto"

dataset:
  type: "cifar10"
  root: "./data"
  download: true
  train: false
  batch_size: 32

attacks:
  - name: "fgsm_test"
    attack_type: "fgsm"
    epsilon: 0.3
    norm: "Linf"

defenses:
  - name: "smoothing_test"
    defense_type: "randomized_smoothing"
    enabled: true
    noise_std: 0.25

output:
  output_dir: "./results"
  save_metrics: true
  log_level: "INFO"
"""


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test configuration enums."""

    def test_attack_type_values(self):
        """Test AttackType enum values."""
        assert AttackType.FGSM.value == "fgsm"
        assert AttackType.PGD.value == "pgd"
        assert AttackType.CW.value == "cw"
        assert AttackType.DEEPFOOL.value == "deepfool"
        assert AttackType.AUTOATTACK.value == "autoattack"

    def test_defense_type_values(self):
        """Test DefenseType enum values."""
        assert DefenseType.ADVERSARIAL_TRAINING.value == "adversarial_training"
        assert DefenseType.RANDOMIZED_SMOOTHING.value == "randomized_smoothing"
        assert DefenseType.INPUT_TRANSFORMATION.value == "input_transformation"
        assert DefenseType.ENSEMBLE.value == "ensemble"

    def test_norm_type_values(self):
        """Test NormType enum values."""
        assert NormType.LINF.value == "Linf"
        assert NormType.L2.value == "L2"
        assert NormType.L1.value == "L1"
        assert NormType.L0.value == "L0"

    def test_dataset_type_values(self):
        """Test DatasetType enum values."""
        assert DatasetType.CIFAR10.value == "cifar10"
        assert DatasetType.CIFAR100.value == "cifar100"
        assert DatasetType.IMAGENET.value == "imagenet"
        assert DatasetType.MNIST.value == "mnist"
        assert DatasetType.CUSTOM.value == "custom"

    def test_device_type_values(self):
        """Test DeviceType enum values."""
        assert DeviceType.AUTO.value == "auto"
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"


# =============================================================================
# ModelConfig Tests
# =============================================================================


class TestModelConfig:
    """Test ModelConfig Pydantic model."""

    def test_valid_config(self):
        """Test valid model configuration."""
        config = ModelConfig(
            architecture="resnet18",
            pretrained=True,
            num_classes=10,
            input_shape=[3, 224, 224],
            device=DeviceType.AUTO,
        )
        assert config.architecture == "resnet18"
        assert config.pretrained is True
        assert config.num_classes == 10

    def test_default_values(self):
        """Test default values."""
        config = ModelConfig(architecture="resnet50")
        assert config.pretrained is True
        assert config.num_classes == 10  # Default is 10
        assert config.device == DeviceType.AUTO

    def test_string_device(self):
        """Test device as string (coerced to enum)."""
        config = ModelConfig(architecture="vgg16", device="cuda")
        assert config.device == DeviceType.CUDA

    def test_input_shape_validation(self):
        """Test input shape validation."""
        config = ModelConfig(
            architecture="resnet18",
            input_shape=[3, 256, 256],
        )
        assert config.input_shape == [3, 256, 256]


# =============================================================================
# DatasetConfig Tests
# =============================================================================


class TestDatasetConfig:
    """Test DatasetConfig Pydantic model."""

    def test_valid_config(self):
        """Test valid dataset configuration."""
        config = DatasetConfig(
            type=DatasetType.CIFAR10,
            root="./data",
            download=True,
            train=False,
            batch_size=32,
        )
        assert config.type == DatasetType.CIFAR10
        assert config.root == "./data"
        assert config.batch_size == 32

    def test_default_values(self):
        """Test default values."""
        config = DatasetConfig(type="mnist")
        assert config.root == "./data"
        assert config.download is True
        assert config.train is False
        assert config.batch_size == 32

    def test_string_type(self):
        """Test type as string (coerced to enum)."""
        config = DatasetConfig(type="imagenet")
        assert config.type == DatasetType.IMAGENET

    def test_subset_size(self):
        """Test subset size configuration."""
        config = DatasetConfig(
            type="cifar10",
            subset_size=1000,
        )
        assert config.subset_size == 1000


# =============================================================================
# AttackConfig Tests
# =============================================================================


class TestAttackConfig:
    """Test AttackConfig Pydantic model."""

    def test_valid_fgsm_config(self):
        """Test valid FGSM attack configuration."""
        config = AttackConfig(
            name="fgsm_attack",
            attack_type=AttackType.FGSM,
            epsilon=0.3,
            norm=NormType.LINF,
        )
        assert config.attack_type == AttackType.FGSM
        assert config.epsilon == 0.3

    def test_valid_pgd_config(self):
        """Test valid PGD attack configuration."""
        config = AttackConfig(
            name="pgd_attack",
            attack_type="pgd",
            epsilon=0.3,
            norm="Linf",
            iterations=40,
            step_size=0.01,
            random_start=True,
        )
        assert config.attack_type == AttackType.PGD
        assert config.iterations == 40
        assert config.step_size == 0.01

    def test_valid_cw_config(self):
        """Test valid C&W attack configuration."""
        config = AttackConfig(
            name="cw_attack",
            attack_type="cw",
            epsilon=0.5,
            norm="L2",
            iterations=100,
            learning_rate=0.01,
            confidence=0.0,
            binary_search_steps=9,
        )
        assert config.attack_type == AttackType.CW
        assert config.binary_search_steps == 9

    def test_epsilon_validation(self):
        """Test epsilon validation (0-1 range)."""
        # Valid epsilon
        config = AttackConfig(name="test", attack_type="fgsm", epsilon=0.5)
        assert config.epsilon == 0.5

        # Invalid epsilon (too high)
        with pytest.raises(ValueError):
            AttackConfig(name="test", attack_type="fgsm", epsilon=1.5)

    def test_targeted_attack(self):
        """Test targeted attack configuration."""
        config = AttackConfig(
            name="targeted_attack",
            attack_type="fgsm",
            epsilon=0.3,
            targeted=True,
            target_class=5,
        )
        assert config.targeted is True
        assert config.target_class == 5

    def test_targeted_requires_target_class(self):
        """Test that targeted attack requires target_class."""
        # This should raise validation error
        with pytest.raises(ValueError):
            AttackConfig(
                name="test",
                attack_type="fgsm",
                epsilon=0.3,
                targeted=True,
                target_class=None,
            )

    def test_auto_step_size(self):
        """Test automatic step size computation."""
        config = AttackConfig(
            name="pgd",
            attack_type="pgd",
            epsilon=0.3,
            iterations=30,
            step_size=None,
        )
        # step_size should be auto-computed
        assert config.step_size is None or config.step_size > 0


# =============================================================================
# DefenseConfig Tests
# =============================================================================


class TestDefenseConfig:
    """Test DefenseConfig Pydantic model."""

    def test_valid_smoothing_config(self):
        """Test valid randomized smoothing configuration."""
        config = DefenseConfig(
            name="smoothing",
            defense_type=DefenseType.RANDOMIZED_SMOOTHING,
            enabled=True,
            noise_std=0.25,
            num_samples=100,
        )
        assert config.defense_type == DefenseType.RANDOMIZED_SMOOTHING
        assert config.noise_std == 0.25

    def test_valid_adversarial_training_config(self):
        """Test valid adversarial training configuration."""
        config = DefenseConfig(
            name="adv_train",
            defense_type="adversarial_training",
            enabled=True,
            train_epsilon=0.3,
            train_iterations=7,
            train_ratio=0.5,
        )
        assert config.defense_type == DefenseType.ADVERSARIAL_TRAINING
        assert config.train_epsilon == 0.3

    def test_valid_input_transformation_config(self):
        """Test valid input transformation configuration."""
        config = DefenseConfig(
            name="jpeg",
            defense_type="input_transformation",
            enabled=True,
            jpeg_quality=75,
            blur_sigma=1.0,
            bit_depth=4,
        )
        assert config.defense_type == DefenseType.INPUT_TRANSFORMATION
        assert config.jpeg_quality == 75

    def test_disabled_defense(self):
        """Test disabled defense configuration."""
        config = DefenseConfig(
            name="disabled",
            defense_type="ensemble",
            enabled=False,
        )
        assert config.enabled is False


# =============================================================================
# OutputConfig Tests
# =============================================================================


class TestOutputConfig:
    """Test OutputConfig Pydantic model."""

    def test_valid_config(self):
        """Test valid output configuration."""
        config = OutputConfig(
            output_dir="./results",
            save_adversarial=True,
            save_perturbations=True,
            save_metrics=True,
            log_level="DEBUG",
        )
        assert config.output_dir == "./results"
        assert config.save_adversarial is True
        assert config.log_level == "DEBUG"

    def test_default_values(self):
        """Test default output values."""
        config = OutputConfig(output_dir="./out")
        assert config.save_adversarial is False
        assert config.save_perturbations is False
        assert config.save_metrics is True
        assert config.log_level == "INFO"

    def test_wandb_config(self):
        """Test Weights & Biases configuration."""
        config = OutputConfig(
            output_dir="./results",
            wandb=True,
            wandb_project="medtech-security",
            wandb_entity="team",
        )
        assert config.wandb is True
        assert config.wandb_project == "medtech-security"


# =============================================================================
# ExperimentConfig Tests
# =============================================================================


class TestExperimentConfig:
    """Test ExperimentConfig Pydantic model."""

    def test_valid_config(self):
        """Test valid experiment configuration."""
        config = ExperimentConfig(
            name="test_experiment",
            seed=42,
            model=ModelConfig(architecture="resnet18"),
            dataset=DatasetConfig(type="cifar10"),
            attacks=[
                AttackConfig(name="fgsm", attack_type="fgsm", epsilon=0.3),
            ],
            defenses=[
                DefenseConfig(name="smooth", defense_type="randomized_smoothing"),
            ],
            output=OutputConfig(output_dir="./results"),
        )
        assert config.name == "test_experiment"
        assert config.seed == 42
        assert len(config.attacks) == 1
        assert len(config.defenses) == 1

    def test_minimal_config(self):
        """Test minimal experiment configuration (requires at least one attack or defense)."""
        config = ExperimentConfig(
            name="minimal",
            model=ModelConfig(architecture="vgg16"),
            dataset=DatasetConfig(type="mnist"),
            attacks=[AttackConfig(name="fgsm", attack_type="fgsm", epsilon=0.3)],
            defenses=[],
            output=OutputConfig(output_dir="./out"),
        )
        assert config.name == "minimal"
        assert config.seed == 42  # default

    def test_deterministic(self):
        """Test deterministic flag."""
        config = ExperimentConfig(
            name="deterministic_test",
            deterministic=True,
            model=ModelConfig(architecture="resnet18"),
            dataset=DatasetConfig(type="cifar10"),
            attacks=[AttackConfig(name="fgsm", attack_type="fgsm", epsilon=0.3)],
            defenses=[],
            output=OutputConfig(output_dir="./results"),
        )
        assert config.deterministic is True


# =============================================================================
# ConfigLoader Tests
# =============================================================================


class TestConfigLoader:
    """Test ConfigLoader class."""

    def test_initialization(self):
        """Test loader initialization."""
        loader = ConfigLoader()
        assert loader.config_dir == Path.cwd()

    def test_initialization_with_dir(self, temp_dir):
        """Test loader initialization with custom directory."""
        loader = ConfigLoader(config_dir=temp_dir)
        assert loader.config_dir == temp_dir

    def test_load_yaml(self, temp_dir, valid_attack_yaml):
        """Test loading raw YAML file."""
        yaml_path = temp_dir / "test.yaml"
        yaml_path.write_text(valid_attack_yaml)

        loader = ConfigLoader(config_dir=temp_dir)
        data = loader.load_yaml(yaml_path)

        assert isinstance(data, dict)
        assert data["name"] == "test_fgsm_attack"

    def test_load_yaml_not_found(self):
        """Test loading non-existent file."""
        loader = ConfigLoader()

        with pytest.raises(ConfigError) as exc_info:
            loader.load_yaml("nonexistent.yaml")

        assert "not found" in str(exc_info.value).lower()

    def test_load_yaml_invalid_syntax(self, temp_dir):
        """Test loading invalid YAML."""
        invalid_yaml = temp_dir / "invalid.yaml"
        invalid_yaml.write_text("name: [unclosed bracket")

        loader = ConfigLoader(config_dir=temp_dir)

        with pytest.raises(ConfigError) as exc_info:
            loader.load_yaml(invalid_yaml)

        assert "parse" in str(exc_info.value).lower() or "yaml" in str(exc_info.value).lower()

    def test_load_attack(self, temp_dir, valid_attack_yaml):
        """Test loading attack configuration."""
        yaml_path = temp_dir / "attack.yaml"
        yaml_path.write_text(valid_attack_yaml)

        loader = ConfigLoader(config_dir=temp_dir)
        config = loader.load_attack(yaml_path)

        assert isinstance(config, AttackConfig)
        assert config.name == "test_fgsm_attack"
        assert config.attack_type == AttackType.FGSM

    def test_load_defense(self, temp_dir, valid_defense_yaml):
        """Test loading defense configuration."""
        yaml_path = temp_dir / "defense.yaml"
        yaml_path.write_text(valid_defense_yaml)

        loader = ConfigLoader(config_dir=temp_dir)
        config = loader.load_defense(yaml_path)

        assert isinstance(config, DefenseConfig)
        assert config.name == "test_defense"
        assert config.defense_type == DefenseType.RANDOMIZED_SMOOTHING

    def test_load_experiment(self, temp_dir, valid_experiment_yaml):
        """Test loading experiment configuration."""
        yaml_path = temp_dir / "experiment.yaml"
        yaml_path.write_text(valid_experiment_yaml)

        loader = ConfigLoader(config_dir=temp_dir)
        config = loader.load_experiment(yaml_path)

        assert isinstance(config, ExperimentConfig)
        assert config.name == "test_experiment"
        assert len(config.attacks) == 1
        assert len(config.defenses) == 1

    def test_save_yaml(self, temp_dir):
        """Test saving configuration to YAML."""
        config = AttackConfig(
            name="saved_attack",
            attack_type="fgsm",
            epsilon=0.3,
        )

        output_path = temp_dir / "saved.yaml"
        ConfigLoader.save_yaml(config, output_path)

        assert output_path.exists()

        # Reload and verify
        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "saved_attack"

    def test_generate_attack_template(self):
        """Test attack template generation."""
        template = ConfigLoader.generate_attack_template()

        assert isinstance(template, str)
        assert "attack_type" in template
        assert "epsilon" in template

    def test_generate_defense_template(self):
        """Test defense template generation."""
        template = ConfigLoader.generate_defense_template()

        assert isinstance(template, str)
        assert "defense_type" in template
        assert "enabled" in template

    def test_generate_experiment_template(self):
        """Test experiment template generation."""
        template = ConfigLoader.generate_experiment_template()

        assert isinstance(template, str)
        assert "model:" in template
        assert "dataset:" in template
        assert "attacks:" in template
        assert "defenses:" in template


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_attack_config(self, temp_dir, valid_attack_yaml):
        """Test load_attack_config function."""
        yaml_path = temp_dir / "attack.yaml"
        yaml_path.write_text(valid_attack_yaml)

        config = load_attack_config(yaml_path)

        assert isinstance(config, AttackConfig)

    def test_load_defense_config(self, temp_dir, valid_defense_yaml):
        """Test load_defense_config function."""
        yaml_path = temp_dir / "defense.yaml"
        yaml_path.write_text(valid_defense_yaml)

        config = load_defense_config(yaml_path)

        assert isinstance(config, DefenseConfig)

    def test_load_experiment_config(self, temp_dir, valid_experiment_yaml):
        """Test load_experiment_config function."""
        yaml_path = temp_dir / "experiment.yaml"
        yaml_path.write_text(valid_experiment_yaml)

        config = load_experiment_config(yaml_path)

        assert isinstance(config, ExperimentConfig)

    def test_validate_config_dict(self):
        """Test validate_config function with dictionary."""
        data = {
            "name": "test",
            "attack_type": "fgsm",
            "epsilon": 0.3,
        }

        config = validate_config(data, config_type="attack")

        assert isinstance(config, AttackConfig)

    def test_validate_config_invalid_type(self):
        """Test validate_config with invalid type."""
        with pytest.raises(ValueError):
            validate_config({}, config_type="invalid")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_missing_required_field(self, temp_dir):
        """Test validation error for missing required field."""
        yaml_content = """
name: "incomplete"
# Missing attack_type
epsilon: 0.3
"""
        yaml_path = temp_dir / "incomplete.yaml"
        yaml_path.write_text(yaml_content)

        loader = ConfigLoader(config_dir=temp_dir)

        with pytest.raises(ConfigError):
            loader.load_attack(yaml_path)

    def test_invalid_field_value(self, temp_dir):
        """Test validation error for invalid field value."""
        yaml_content = """
name: "invalid"
attack_type: "fgsm"
epsilon: 5.0  # Invalid: > 1.0
"""
        yaml_path = temp_dir / "invalid.yaml"
        yaml_path.write_text(yaml_content)

        loader = ConfigLoader(config_dir=temp_dir)

        with pytest.raises(ConfigError):
            loader.load_attack(yaml_path)

    def test_invalid_enum_value(self, temp_dir):
        """Test validation error for invalid enum value."""
        yaml_content = """
name: "invalid_enum"
attack_type: "invalid_attack_type"
epsilon: 0.3
"""
        yaml_path = temp_dir / "invalid_enum.yaml"
        yaml_path.write_text(yaml_content)

        loader = ConfigLoader(config_dir=temp_dir)

        with pytest.raises(ConfigError):
            loader.load_attack(yaml_path)

    def test_empty_yaml(self, temp_dir):
        """Test loading empty YAML file."""
        yaml_path = temp_dir / "empty.yaml"
        yaml_path.write_text("")

        loader = ConfigLoader(config_dir=temp_dir)
        data = loader.load_yaml(yaml_path)

        assert data == {}

    def test_non_dict_yaml(self, temp_dir):
        """Test loading YAML that is not a dictionary."""
        yaml_path = temp_dir / "list.yaml"
        yaml_path.write_text("- item1\n- item2")

        loader = ConfigLoader(config_dir=temp_dir)

        with pytest.raises(ConfigError) as exc_info:
            loader.load_yaml(yaml_path)

        assert "dictionary" in str(exc_info.value).lower()


# =============================================================================
# Config Error Tests
# =============================================================================


class TestConfigError:
    """Test ConfigError exception."""

    def test_basic_error(self):
        """Test basic ConfigError."""
        error = ConfigError("Test error message")
        assert str(error) == "Test error message"
        assert error.details == {}

    def test_error_with_details(self):
        """Test ConfigError with details."""
        error = ConfigError(
            "Validation failed",
            details={"path": "config.yaml", "field": "epsilon"},
        )
        assert error.details["path"] == "config.yaml"
        assert error.details["field"] == "epsilon"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_round_trip(self, temp_dir):
        """Test saving and loading configuration."""
        original = ExperimentConfig(
            name="round_trip_test",
            seed=123,
            model=ModelConfig(architecture="resnet50", num_classes=100),
            dataset=DatasetConfig(type="imagenet", batch_size=64),
            attacks=[
                AttackConfig(name="pgd", attack_type="pgd", epsilon=0.1, iterations=20),
            ],
            defenses=[
                DefenseConfig(name="adv", defense_type="adversarial_training"),
            ],
            output=OutputConfig(output_dir="./test_results"),
        )

        # Save
        output_path = temp_dir / "roundtrip.yaml"
        ConfigLoader.save_yaml(original, output_path)

        # Load
        loader = ConfigLoader()
        loaded = loader.load_experiment(output_path)

        # Compare
        assert loaded.name == original.name
        assert loaded.seed == original.seed
        assert loaded.model.architecture == original.model.architecture
        assert len(loaded.attacks) == len(original.attacks)

    def test_template_is_valid(self, temp_dir):
        """Test that generated templates are valid."""
        loader = ConfigLoader(config_dir=temp_dir)

        # Test attack template
        attack_template = loader.generate_attack_template()
        attack_path = temp_dir / "attack_template.yaml"
        attack_path.write_text(attack_template)
        attack_config = loader.load_attack(attack_path)
        assert isinstance(attack_config, AttackConfig)

        # Test defense template
        defense_template = loader.generate_defense_template()
        defense_path = temp_dir / "defense_template.yaml"
        defense_path.write_text(defense_template)
        defense_config = loader.load_defense(defense_path)
        assert isinstance(defense_config, DefenseConfig)

        # Test experiment template
        experiment_template = loader.generate_experiment_template()
        experiment_path = temp_dir / "experiment_template.yaml"
        experiment_path.write_text(experiment_template)
        experiment_config = loader.load_experiment(experiment_path)
        assert isinstance(experiment_config, ExperimentConfig)
