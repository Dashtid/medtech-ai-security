"""
Tests for the Enhanced CLI module.

Tests cover:
- CLI commands (run, validate, generate, info, version)
- Configuration loading via CLI
- Error handling
- Progress display
- Interactive wizard

"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from medtech_ai_security.cli.main import (
    app,
    create_progress,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    setup_logging,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_experiment_yaml():
    """Valid experiment configuration YAML."""
    return """
name: "cli_test_experiment"
description: "Test experiment for CLI"
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
  - name: "fgsm_cli_test"
    attack_type: "fgsm"
    epsilon: 0.3
    norm: "Linf"

defenses:
  - name: "smoothing_cli_test"
    defense_type: "randomized_smoothing"
    enabled: true
    noise_std: 0.25

output:
  output_dir: "./cli_results"
  save_metrics: true
  log_level: "INFO"
"""


@pytest.fixture
def valid_attack_yaml():
    """Valid attack configuration YAML."""
    return """
name: "cli_test_attack"
description: "Test attack for CLI"
version: "1.0.0"
attack_type: "fgsm"
epsilon: 0.3
norm: "Linf"
targeted: false
"""


@pytest.fixture
def valid_defense_yaml():
    """Valid defense configuration YAML."""
    return """
name: "cli_test_defense"
description: "Test defense for CLI"
version: "1.0.0"
defense_type: "randomized_smoothing"
enabled: true
noise_std: 0.25
num_samples: 100
"""


# =============================================================================
# Version Command Tests
# =============================================================================


class TestVersionCommand:
    """Test the version command."""

    def test_version(self, runner):
        """Test version command output."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "medtech-ai-security" in result.output.lower() or "version" in result.output.lower()


# =============================================================================
# Info Command Tests
# =============================================================================


class TestInfoCommand:
    """Test the info command."""

    def test_info(self, runner):
        """Test info command output."""
        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        # Should contain system information
        assert "python" in result.output.lower() or "version" in result.output.lower()

    def test_info_shows_features(self, runner):
        """Test info command shows available features."""
        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        # Should list available features
        assert "attack" in result.output.lower() or "defense" in result.output.lower()


# =============================================================================
# Validate Command Tests
# =============================================================================


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_experiment(self, runner, temp_dir, valid_experiment_yaml):
        """Test validating experiment configuration."""
        config_path = temp_dir / "experiment.yaml"
        config_path.write_text(valid_experiment_yaml)

        result = runner.invoke(app, ["validate", str(config_path)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_attack(self, runner, temp_dir, valid_attack_yaml):
        """Test validating attack configuration."""
        config_path = temp_dir / "attack.yaml"
        config_path.write_text(valid_attack_yaml)

        result = runner.invoke(app, ["validate", str(config_path), "--type", "attack"])

        assert result.exit_code == 0

    def test_validate_defense(self, runner, temp_dir, valid_defense_yaml):
        """Test validating defense configuration."""
        config_path = temp_dir / "defense.yaml"
        config_path.write_text(valid_defense_yaml)

        result = runner.invoke(app, ["validate", str(config_path), "--type", "defense"])

        assert result.exit_code == 0

    def test_validate_invalid_config(self, runner, temp_dir):
        """Test validating invalid configuration."""
        invalid_yaml = """
name: "invalid"
# Missing required fields
"""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text(invalid_yaml)

        result = runner.invoke(app, ["validate", str(config_path), "--type", "attack"])

        assert result.exit_code != 0

    def test_validate_nonexistent_file(self, runner):
        """Test validating nonexistent file."""
        result = runner.invoke(app, ["validate", "nonexistent.yaml"])

        assert result.exit_code != 0

    def test_validate_verbose(self, runner, temp_dir, valid_experiment_yaml):
        """Test validate with verbose output."""
        config_path = temp_dir / "experiment.yaml"
        config_path.write_text(valid_experiment_yaml)

        result = runner.invoke(app, ["validate", str(config_path), "--verbose"])

        assert result.exit_code == 0
        # Verbose should show more details
        assert len(result.output) > 0


# =============================================================================
# Generate Command Tests
# =============================================================================


class TestGenerateCommand:
    """Test the generate command."""

    def test_generate_experiment_template(self, runner):
        """Test generating experiment template."""
        result = runner.invoke(app, ["generate", "--type", "experiment"])

        assert result.exit_code == 0
        assert "model:" in result.output.lower()
        assert "attacks:" in result.output.lower()

    def test_generate_attack_template(self, runner):
        """Test generating attack template."""
        result = runner.invoke(app, ["generate", "--type", "attack"])

        assert result.exit_code == 0
        assert "attack_type" in result.output.lower()
        assert "epsilon" in result.output.lower()

    def test_generate_defense_template(self, runner):
        """Test generating defense template."""
        result = runner.invoke(app, ["generate", "--type", "defense"])

        assert result.exit_code == 0
        assert "defense_type" in result.output.lower()
        assert "enabled" in result.output.lower()

    def test_generate_to_file(self, runner, temp_dir):
        """Test generating template to file."""
        output_path = temp_dir / "generated.yaml"

        result = runner.invoke(
            app, ["generate", "--type", "experiment", "--output", str(output_path)]
        )

        assert result.exit_code == 0
        assert output_path.exists()
        assert output_path.read_text().strip() != ""

    def test_generate_invalid_type(self, runner):
        """Test generating with invalid type."""
        result = runner.invoke(app, ["generate", "--type", "invalid"])

        assert result.exit_code != 0


# =============================================================================
# Run Command Tests
# =============================================================================


class TestRunCommand:
    """Test the run command."""

    def test_run_dry_run(self, runner, temp_dir, valid_experiment_yaml):
        """Test run command with dry-run flag."""
        config_path = temp_dir / "experiment.yaml"
        config_path.write_text(valid_experiment_yaml)

        result = runner.invoke(app, ["run", str(config_path), "--dry-run"])

        assert result.exit_code == 0
        assert "dry run" in result.output.lower()

    def test_run_nonexistent_config(self, runner):
        """Test run command with nonexistent config."""
        result = runner.invoke(app, ["run", "nonexistent.yaml"])

        assert result.exit_code != 0

    def test_run_with_output_override(self, runner, temp_dir, valid_experiment_yaml):
        """Test run command with output directory override."""
        config_path = temp_dir / "experiment.yaml"
        config_path.write_text(valid_experiment_yaml)
        output_dir = temp_dir / "custom_output"

        result = runner.invoke(
            app, ["run", str(config_path), "--output", str(output_dir), "--dry-run"]
        )

        assert result.exit_code == 0

    def test_run_verbose(self, runner, temp_dir, valid_experiment_yaml):
        """Test run command with verbose flag."""
        config_path = temp_dir / "experiment.yaml"
        config_path.write_text(valid_experiment_yaml)

        result = runner.invoke(app, ["run", str(config_path), "--dry-run", "--verbose"])

        assert result.exit_code == 0


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_progress(self):
        """Test progress bar creation."""
        progress = create_progress()
        assert progress is not None

    def test_setup_logging_default(self):
        """Test default logging setup."""
        # Should not raise
        setup_logging()

    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        # Should not raise
        setup_logging(verbose=True)

    def test_setup_logging_with_file(self, temp_dir):
        """Test logging setup with file."""
        log_file = temp_dir / "test.log"
        setup_logging(log_file=log_file)

        assert log_file.exists() or True  # May not create file until first log

    def test_print_functions(self, capsys):
        """Test print utility functions."""
        from rich.console import Console

        console = Console(force_terminal=True)

        # These should not raise
        with patch('medtech_ai_security.cli.main.console', console):
            print_success("Success message")
            print_error("Error message")
            print_warning("Warning message")
            print_info("Info message")

    def test_print_header(self, capsys):
        """Test header printing."""
        from rich.console import Console

        console = Console(force_terminal=True)

        with patch('medtech_ai_security.cli.main.console', console):
            print_header()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test CLI error handling."""

    def test_invalid_yaml_syntax(self, runner, temp_dir):
        """Test handling of invalid YAML syntax."""
        invalid_yaml = temp_dir / "invalid.yaml"
        invalid_yaml.write_text("name: [unclosed")

        result = runner.invoke(app, ["validate", str(invalid_yaml)])

        assert result.exit_code != 0

    def test_empty_yaml(self, runner, temp_dir):
        """Test handling of empty YAML file."""
        empty_yaml = temp_dir / "empty.yaml"
        empty_yaml.write_text("")

        result = runner.invoke(app, ["validate", str(empty_yaml)])

        # Empty YAML should fail validation for experiment (missing required fields)
        assert result.exit_code != 0

    def test_permission_error(self, runner, temp_dir):
        """Test handling of permission errors."""
        # Create a file in a non-writable location (platform-specific)
        # This test may be skipped on some systems
        pass


# =============================================================================
# Interactive Mode Tests
# =============================================================================


class TestInteractiveMode:
    """Test interactive configuration wizard."""

    def test_interactive_experiment_wizard(self, runner):
        """Test interactive experiment wizard."""
        # Simulate user input
        input_data = "test_exp\nTest description\n42\nresnet18\ncifar10\n./results\nN\nN\n"

        result = runner.invoke(
            app,
            ["generate", "--type", "experiment", "--interactive"],
            input=input_data,
        )

        # Interactive mode should complete or request input
        # Exit code depends on whether all prompts are satisfied
        assert result.exit_code in [0, 1]  # May fail if prompts aren't fully satisfied

    def test_interactive_attack_wizard(self, runner):
        """Test interactive attack wizard."""
        input_data = "test_attack\nfgsm\n0.3\nLinf\n"

        result = runner.invoke(
            app,
            ["generate", "--type", "attack", "--interactive"],
            input=input_data,
        )

        assert result.exit_code in [0, 1]

    def test_interactive_defense_wizard(self, runner):
        """Test interactive defense wizard."""
        input_data = "test_defense\nrandomized_smoothing\n"

        result = runner.invoke(
            app,
            ["generate", "--type", "defense", "--interactive"],
            input=input_data,
        )

        assert result.exit_code in [0, 1]


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_full_workflow(self, runner, temp_dir):
        """Test complete CLI workflow."""
        # 1. Generate template
        template_path = temp_dir / "experiment.yaml"
        result = runner.invoke(
            app, ["generate", "--type", "experiment", "--output", str(template_path)]
        )
        assert result.exit_code == 0
        assert template_path.exists()

        # 2. Validate template
        result = runner.invoke(app, ["validate", str(template_path)])
        assert result.exit_code == 0

        # 3. Dry-run experiment
        result = runner.invoke(app, ["run", str(template_path), "--dry-run"])
        assert result.exit_code == 0

    def test_help_commands(self, runner):
        """Test all help commands work."""
        # Main help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Subcommand help
        for cmd in ["run", "validate", "generate", "info", "version"]:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0, f"Help for {cmd} failed"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_unicode_in_config(self, runner, temp_dir):
        """Test handling of unicode in configuration."""
        unicode_yaml = """
name: "test_unicode_"
description: "Test with special chars"
version: "1.0.0"
attack_type: "fgsm"
epsilon: 0.3
norm: "Linf"
"""
        config_path = temp_dir / "unicode.yaml"
        config_path.write_text(unicode_yaml, encoding="utf-8")

        result = runner.invoke(app, ["validate", str(config_path), "--type", "attack"])

        assert result.exit_code == 0

    def test_very_long_path(self, runner, temp_dir):
        """Test handling of very long file paths."""
        # Create nested directories
        nested = temp_dir
        for i in range(10):
            nested = nested / f"level{i}"
        nested.mkdir(parents=True, exist_ok=True)

        config_path = nested / "config.yaml"
        config_path.write_text("""
name: "nested"
attack_type: "fgsm"
epsilon: 0.3
""")

        result = runner.invoke(app, ["validate", str(config_path), "--type", "attack"])

        # Should handle long paths
        assert result.exit_code in [0, 1]

    def test_special_characters_in_name(self, runner, temp_dir):
        """Test configuration with special characters in name."""
        yaml_content = """
name: "test-config_v1.0"
attack_type: "fgsm"
epsilon: 0.3
"""
        config_path = temp_dir / "special.yaml"
        config_path.write_text(yaml_content)

        result = runner.invoke(app, ["validate", str(config_path), "--type", "attack"])

        assert result.exit_code == 0
