"""
MedTech AI Security - Command Line Interface.

Provides a rich, interactive CLI for running adversarial ML experiments,
evaluating model robustness, and managing security configurations.

Usage:
    medtech-ai-security run experiment.yaml
    medtech-ai-security validate config.yaml
    medtech-ai-security generate --type experiment --output config.yaml
"""

from medtech_ai_security.cli.main import app, main

__all__ = ["app", "main"]
