"""
Integration Module

Integrations with external systems:
- DefectDojo API for vulnerability management
- SBOM tools (Grype, Syft)
- Alerting systems
"""

from medtech_ai_security.integration.defectdojo import (
    DefectDojoClient,
    DefectDojoConfig,
    Finding,
)

__all__ = [
    "DefectDojoClient",
    "DefectDojoConfig",
    "Finding",
]
