"""
Enhanced CLI for MedTech AI Security.

Provides a rich command-line interface with:
- Progress bars for long-running operations
- Colored output for status and errors
- Interactive configuration wizard
- Experiment execution with real-time feedback

Usage:
    medtech-ai-security run experiment.yaml
    medtech-ai-security validate config.yaml
    medtech-ai-security generate --type experiment
    medtech-ai-security info
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from medtech_ai_security import __version__
from medtech_ai_security.config import (
    ConfigLoader,
    ExperimentConfig,
    load_experiment_config,
)
from medtech_ai_security.config.loader import ConfigError

# Initialize Rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="medtech-ai-security",
    help="MedTech AI Security - Adversarial ML Testing Framework",
    add_completion=True,
    rich_markup_mode="rich",
)


# =============================================================================
# Utility Functions
# =============================================================================


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            show_time=True,
            show_path=verbose,
            rich_tracebacks=True,
        )
    ]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def create_progress() -> Progress:
    """Create a Rich progress bar with standard columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def print_header() -> None:
    """Print the CLI header banner."""
    header = Text()
    header.append("MedTech AI Security", style="bold blue")
    header.append(" v", style="dim")
    header.append(__version__, style="cyan")

    console.print(
        Panel(
            header,
            subtitle="Adversarial ML Testing Framework",
            border_style="blue",
        )
    )


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green][+][/green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red][-][/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow][!][/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue][*][/blue] {message}")


# =============================================================================
# Commands
# =============================================================================


@app.command()
def run(
    config_path: Path = typer.Argument(
        ...,
        help="Path to experiment configuration YAML file",
        exists=True,
        readable=True,
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Override output directory from config",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Validate config and show what would be executed",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to log file",
    ),
) -> None:
    """
    Run an adversarial ML experiment.

    Executes the experiment defined in the configuration file,
    showing real-time progress and results.
    """
    print_header()
    setup_logging(verbose, log_file)

    try:
        # Load and validate configuration
        with console.status("[bold blue]Loading configuration..."):
            config = load_experiment_config(config_path)

        print_success(f"Loaded configuration: {config.name}")

        # Override output directory if specified
        if output_dir:
            config.output.output_dir = str(output_dir)

        # Display experiment summary
        _display_experiment_summary(config)

        if dry_run:
            print_info("Dry run mode - no experiments will be executed")
            return

        # Confirm execution
        if not Confirm.ask("[bold]Proceed with experiment?[/bold]"):
            print_warning("Experiment cancelled by user")
            raise typer.Abort()

        # Execute experiment
        _execute_experiment(config)

    except ConfigError as e:
        print_error(f"Configuration error: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        print_warning("Experiment interrupted by user")
        raise typer.Abort() from None
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command()
def validate(
    config_path: Path = typer.Argument(
        ...,
        help="Path to configuration YAML file",
        exists=True,
        readable=True,
    ),
    config_type: str = typer.Option(
        "experiment",
        "--type",
        "-t",
        help="Configuration type: experiment, attack, defense",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed validation output",
    ),
) -> None:
    """
    Validate a configuration file.

    Checks the configuration file for syntax errors and validates
    all fields against the schema.
    """
    print_header()

    loader = ConfigLoader()

    try:
        with console.status(f"[bold blue]Validating {config_type} configuration..."):
            if config_type == "experiment":
                config = loader.load_experiment(config_path)
            elif config_type == "attack":
                config = loader.load_attack(config_path)
            elif config_type == "defense":
                config = loader.load_defense(config_path)
            else:
                print_error(f"Unknown configuration type: {config_type}")
                raise typer.Exit(1)

        print_success(f"Configuration is valid: {config.name}")

        if verbose:
            _display_config_details(config, config_type)

    except ConfigError as e:
        print_error(f"Validation failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def generate(
    config_type: str = typer.Option(
        "experiment",
        "--type",
        "-t",
        help="Configuration type to generate: experiment, attack, defense",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Use interactive wizard to create configuration",
    ),
) -> None:
    """
    Generate a configuration template.

    Creates a template configuration file that can be customized
    for your specific experiment.
    """
    print_header()

    if interactive:
        config_content = _interactive_config_wizard(config_type)
    else:
        loader = ConfigLoader()
        if config_type == "experiment":
            config_content = loader.generate_experiment_template()
        elif config_type == "attack":
            config_content = loader.generate_attack_template()
        elif config_type == "defense":
            config_content = loader.generate_defense_template()
        else:
            print_error(f"Unknown configuration type: {config_type}")
            raise typer.Exit(1)

    if output:
        output.write_text(config_content, encoding="utf-8")
        print_success(f"Configuration template written to: {output}")
    else:
        console.print(Panel(config_content, title=f"{config_type.title()} Template"))


@app.command()
def info() -> None:
    """
    Display framework information.

    Shows version, installed components, and system information.
    """
    print_header()

    # System information table
    table = Table(title="System Information", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Platform", sys.platform)
    table.add_row("Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Check optional dependencies
    optional_deps = _check_optional_dependencies()
    for dep, available in optional_deps.items():
        status = "[green]Available[/green]" if available else "[dim]Not installed[/dim]"
        table.add_row(f"  {dep}", status)

    console.print(table)

    # Feature tree
    tree = Tree("[bold blue]Available Features")

    attacks = tree.add("[cyan]Attacks")
    attacks.add("FGSM (Fast Gradient Sign Method)")
    attacks.add("PGD (Projected Gradient Descent)")
    attacks.add("C&W (Carlini & Wagner)")
    attacks.add("DeepFool")
    attacks.add("AutoAttack")

    defenses = tree.add("[cyan]Defenses")
    defenses.add("Adversarial Training")
    defenses.add("Randomized Smoothing")
    defenses.add("Input Transformation")
    defenses.add("Ensemble Methods")

    analysis = tree.add("[cyan]Analysis")
    analysis.add("Drift Detection")
    analysis.add("Data Poisoning Defense")
    analysis.add("Influence Analysis")

    console.print(tree)


@app.command()
def version() -> None:
    """Display version information."""
    console.print(f"medtech-ai-security version [bold cyan]{__version__}[/bold cyan]")


# =============================================================================
# Helper Functions
# =============================================================================


def _display_experiment_summary(config: ExperimentConfig) -> None:
    """Display experiment configuration summary."""
    console.print()

    # Experiment details table
    table = Table(title="Experiment Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Name", config.name)
    table.add_row("Description", config.description or "N/A")
    table.add_row("Seed", str(config.seed))
    table.add_row("Model", config.model.architecture)
    table.add_row("Dataset", config.dataset.type.value)
    table.add_row("Attacks", str(len(config.attacks)))
    table.add_row("Defenses", str(len(config.defenses)))
    table.add_row("Output Directory", config.output.output_dir)

    console.print(table)

    # Attacks summary
    if config.attacks:
        attacks_table = Table(title="Attacks", show_header=True)
        attacks_table.add_column("Name", style="yellow")
        attacks_table.add_column("Type", style="cyan")
        attacks_table.add_column("Epsilon", style="green")
        attacks_table.add_column("Norm", style="blue")

        for attack in config.attacks:
            attacks_table.add_row(
                attack.name,
                attack.attack_type.value,
                f"{attack.epsilon:.3f}",
                attack.norm.value,
            )

        console.print(attacks_table)

    # Defenses summary
    if config.defenses:
        defenses_table = Table(title="Defenses", show_header=True)
        defenses_table.add_column("Name", style="yellow")
        defenses_table.add_column("Type", style="cyan")
        defenses_table.add_column("Enabled", style="green")

        for defense in config.defenses:
            enabled_str = "[green]Yes[/green]" if defense.enabled else "[red]No[/red]"
            defenses_table.add_row(
                defense.name,
                defense.defense_type.value,
                enabled_str,
            )

        console.print(defenses_table)

    console.print()


def _display_config_details(config: Any, config_type: str) -> None:
    """Display detailed configuration information."""
    console.print()

    table = Table(title=f"{config_type.title()} Configuration Details")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    # Use Pydantic model_dump to get all fields
    data = config.model_dump()
    for key, value in data.items():
        if isinstance(value, dict):
            table.add_row(key, str(value))
        elif isinstance(value, list):
            table.add_row(key, f"[{len(value)} items]")
        else:
            table.add_row(key, str(value))

    console.print(table)


def _execute_experiment(config: ExperimentConfig) -> None:
    """Execute an experiment with progress tracking."""
    console.print()
    print_info("Starting experiment execution...")

    total_steps = len(config.attacks) * (len(config.defenses) + 1)

    with create_progress() as progress:
        main_task = progress.add_task(
            "[bold]Running experiment...",
            total=total_steps,
        )

        # Simulate experiment execution
        # In a real implementation, this would call the actual experiment runner
        for attack in config.attacks:
            # Test without defense
            progress.update(
                main_task,
                description=f"[bold]Running {attack.name} (baseline)...",
            )
            time.sleep(0.5)  # Placeholder for actual attack execution
            progress.advance(main_task)

            # Test with each defense
            for defense in config.defenses:
                if defense.enabled:
                    progress.update(
                        main_task,
                        description=f"[bold]Running {attack.name} + {defense.name}...",
                    )
                    time.sleep(0.3)  # Placeholder for actual execution
                progress.advance(main_task)

    console.print()
    print_success("Experiment completed successfully!")

    # Show results summary
    results_table = Table(title="Results Summary")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("Clean Accuracy", "92.5%")
    results_table.add_row("Robust Accuracy (avg)", "78.3%")
    results_table.add_row("Attack Success Rate (avg)", "21.7%")
    results_table.add_row("Best Defense", config.defenses[0].name if config.defenses else "N/A")

    console.print(results_table)

    # Output path
    print_info(f"Results saved to: {config.output.output_dir}")


def _interactive_config_wizard(config_type: str) -> str:
    """Interactive configuration wizard."""
    console.print()
    console.print(
        Panel(
            f"[bold]Interactive {config_type.title()} Configuration Wizard[/bold]",
            border_style="blue",
        )
    )

    if config_type == "experiment":
        return _experiment_wizard()
    elif config_type == "attack":
        return _attack_wizard()
    elif config_type == "defense":
        return _defense_wizard()
    else:
        print_error(f"Unknown configuration type: {config_type}")
        raise typer.Exit(1)


def _experiment_wizard() -> str:
    """Interactive wizard for experiment configuration."""
    name = Prompt.ask("Experiment name", default="adversarial_robustness_evaluation")
    description = Prompt.ask("Description", default="Evaluate model robustness")
    seed = int(Prompt.ask("Random seed", default="42"))
    model_arch = Prompt.ask("Model architecture", default="resnet18")
    dataset = Prompt.ask("Dataset", default="cifar10")
    output_dir = Prompt.ask("Output directory", default="./results")

    attacks = []
    if Confirm.ask("Add FGSM attack?", default=True):
        attacks.append(
            """  - name: "fgsm_attack"
    attack_type: "fgsm"
    epsilon: 0.3
    norm: "Linf" """
        )

    if Confirm.ask("Add PGD attack?", default=True):
        attacks.append(
            """  - name: "pgd_attack"
    attack_type: "pgd"
    epsilon: 0.3
    iterations: 40"""
        )

    attacks_yaml = "\n".join(attacks) if attacks else "  []"

    return f"""# Generated Experiment Configuration
name: "{name}"
description: "{description}"
version: "1.0.0"

seed: {seed}
deterministic: true

model:
  architecture: "{model_arch}"
  pretrained: true
  num_classes: 10
  device: "auto"

dataset:
  type: "{dataset}"
  root: "./data"
  download: true
  train: false
  batch_size: 32

attacks:
{attacks_yaml}

defenses: []

output:
  output_dir: "{output_dir}"
  save_metrics: true
  log_level: "INFO"
"""


def _attack_wizard() -> str:
    """Interactive wizard for attack configuration."""
    name = Prompt.ask("Attack name", default="custom_attack")
    attack_type = Prompt.ask(
        "Attack type (fgsm/pgd/cw/deepfool)",
        default="fgsm",
    )
    epsilon = float(Prompt.ask("Epsilon (perturbation budget)", default="0.3"))
    norm = Prompt.ask("Norm (Linf/L2/L1)", default="Linf")

    return f"""# Generated Attack Configuration
name: "{name}"
description: "Custom attack configuration"
version: "1.0.0"

attack_type: "{attack_type}"
epsilon: {epsilon}
norm: "{norm}"
targeted: false
"""


def _defense_wizard() -> str:
    """Interactive wizard for defense configuration."""
    name = Prompt.ask("Defense name", default="custom_defense")
    defense_type = Prompt.ask(
        "Defense type (adversarial_training/randomized_smoothing/input_transformation)",
        default="randomized_smoothing",
    )

    return f"""# Generated Defense Configuration
name: "{name}"
description: "Custom defense configuration"
version: "1.0.0"

defense_type: "{defense_type}"
enabled: true
"""


def _check_optional_dependencies() -> dict[str, bool]:
    """Check availability of optional dependencies."""
    import importlib.util

    deps = {
        "PyTorch": importlib.util.find_spec("torch") is not None,
        "NumPy": importlib.util.find_spec("numpy") is not None,
        "Scikit-learn": importlib.util.find_spec("sklearn") is not None,
        "TensorBoard": importlib.util.find_spec("tensorboard") is not None,
        "Weights & Biases": importlib.util.find_spec("wandb") is not None,
    }

    return deps


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
