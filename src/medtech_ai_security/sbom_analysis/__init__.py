"""
SBOM Analysis Module for Medical Device Security

This module provides Graph Neural Network-based analysis of Software Bill of Materials
(SBOM) for medical device security assessment.

Key Features:
- SBOM parsing for CycloneDX and SPDX formats
- Graph representation of dependency relationships
- GNN-based vulnerability propagation prediction
- Supply chain risk scoring
- Dependency visualization

Based on 2025 research on GNN-based vulnerability detection:
- Node classification for vulnerability prediction
- Graph-level risk assessment
- Transitive dependency analysis
"""

from medtech_ai_security.sbom_analysis.analyzer import (
    AnalysisReport,
    SBOMAnalyzer,
)
from medtech_ai_security.sbom_analysis.gnn_model import (
    GNNConfig,
    VulnerabilityGNN,
)
from medtech_ai_security.sbom_analysis.graph_builder import (
    NodeFeatures,
    SBOMGraphBuilder,
)
from medtech_ai_security.sbom_analysis.parser import (
    Dependency,
    DependencyGraph,
    Package,
    SBOMParser,
)
from medtech_ai_security.sbom_analysis.risk_scorer import (
    RiskReport,
    SupplyChainRiskScorer,
)

__all__ = [
    # Parser
    "SBOMParser",
    "Package",
    "Dependency",
    "DependencyGraph",
    # Graph Builder
    "SBOMGraphBuilder",
    "NodeFeatures",
    # GNN Model
    "VulnerabilityGNN",
    "GNNConfig",
    # Risk Scorer
    "SupplyChainRiskScorer",
    "RiskReport",
    # Analyzer
    "SBOMAnalyzer",
    "AnalysisReport",
]
