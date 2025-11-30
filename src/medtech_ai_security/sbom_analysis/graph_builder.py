"""
Graph Builder for SBOM Dependency Analysis.

Converts parsed SBOM data into graph representations suitable for
Graph Neural Network processing.

Features:
- Node feature extraction (package metadata, vulnerability info)
- Edge construction (dependency relationships)
- Graph-level feature aggregation
- Support for heterogeneous graphs (multiple node/edge types)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from medtech_ai_security.sbom_analysis.parser import (
    DependencyGraph,
    Package,
    PackageType,
)

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the SBOM graph."""

    PACKAGE = "package"
    VULNERABILITY = "vulnerability"
    LICENSE = "license"


class EdgeType(Enum):
    """Types of edges in the SBOM graph."""

    DEPENDS_ON = "depends_on"
    HAS_VULNERABILITY = "has_vulnerability"
    HAS_LICENSE = "has_license"
    TRANSITIVE = "transitive"


@dataclass
class NodeFeatures:
    """Feature vector for a node in the graph."""

    # Package metadata features
    name_embedding: np.ndarray = field(default_factory=lambda: np.zeros(64))
    version_numeric: float = 0.0
    package_type_onehot: np.ndarray = field(default_factory=lambda: np.zeros(10))

    # Vulnerability features
    has_vulnerability: int = 0
    max_cvss_score: float = 0.0
    vulnerability_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Dependency features
    direct_dependency_count: int = 0
    transitive_dependency_count: int = 0
    dependent_count: int = 0  # How many packages depend on this

    # Supply chain features
    is_root: int = 0
    depth_from_root: int = 0
    is_leaf: int = 0

    def to_vector(self) -> np.ndarray:
        """Convert features to a single feature vector."""
        scalar_features = np.array(
            [
                self.version_numeric,
                self.has_vulnerability,
                self.max_cvss_score,
                self.vulnerability_count,
                self.critical_count,
                self.high_count,
                self.medium_count,
                self.low_count,
                self.direct_dependency_count,
                self.transitive_dependency_count,
                self.dependent_count,
                self.is_root,
                self.depth_from_root,
                self.is_leaf,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [self.name_embedding, self.package_type_onehot, scalar_features]
        )

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector."""
        return 64 + 10 + 14  # name + type_onehot + scalar


@dataclass
class GraphData:
    """Graph data structure for GNN processing."""

    # Node features: (num_nodes, feature_dim)
    node_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 88)))

    # Edge index: (2, num_edges) - source, target node indices
    edge_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), dtype=np.int64))

    # Edge features: (num_edges, edge_feature_dim)
    edge_features: np.ndarray | None = None

    # Node labels for training (vulnerability propagation)
    node_labels: np.ndarray | None = None

    # Graph-level labels
    graph_labels: np.ndarray | None = None

    # Metadata
    node_ids: list[str] = field(default_factory=list)
    num_nodes: int = 0
    num_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "node_features": self.node_features.tolist(),
            "edge_index": self.edge_index.tolist(),
            "node_ids": self.node_ids,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }
        if self.edge_features is not None:
            result["edge_features"] = self.edge_features.tolist()
        if self.node_labels is not None:
            result["node_labels"] = self.node_labels.tolist()
        if self.graph_labels is not None:
            result["graph_labels"] = self.graph_labels.tolist()
        return result


class SBOMGraphBuilder:
    """Builds graph representations from SBOM dependency graphs."""

    # Package type to one-hot index mapping
    PACKAGE_TYPE_INDEX = {
        PackageType.NPM: 0,
        PackageType.PYPI: 1,
        PackageType.MAVEN: 2,
        PackageType.NUGET: 3,
        PackageType.CARGO: 4,
        PackageType.GO: 5,
        PackageType.APK: 6,
        PackageType.DEB: 7,
        PackageType.RPM: 8,
        PackageType.CONTAINER: 9,
        PackageType.UNKNOWN: 9,  # Map unknown to last index
    }

    def __init__(
        self,
        use_name_embeddings: bool = True,
        embedding_dim: int = 64,
        include_vulnerability_nodes: bool = False,
    ):
        """Initialize graph builder.

        Args:
            use_name_embeddings: Whether to compute package name embeddings
            embedding_dim: Dimension of name embeddings
            include_vulnerability_nodes: Whether to create separate nodes for vulns
        """
        self.use_name_embeddings = use_name_embeddings
        self.embedding_dim = embedding_dim
        self.include_vulnerability_nodes = include_vulnerability_nodes

    def build(self, dep_graph: DependencyGraph) -> GraphData:
        """Build a graph representation from a dependency graph.

        Args:
            dep_graph: Parsed SBOM dependency graph

        Returns:
            GraphData suitable for GNN processing
        """
        # Create node index mapping
        node_to_idx: dict[str, int] = {}
        idx = 0
        for pkg_id in dep_graph.packages:
            node_to_idx[pkg_id] = idx
            idx += 1

        num_nodes = len(node_to_idx)
        logger.info(f"Building graph with {num_nodes} nodes")

        # Compute depth from root for each node
        depths = self._compute_depths(dep_graph, node_to_idx)

        # Build node features
        feature_dim = NodeFeatures().feature_dim
        node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        node_ids = []

        for pkg_id, pkg in dep_graph.packages.items():
            idx = node_to_idx[pkg_id]
            features = self._extract_node_features(pkg, dep_graph, depths.get(pkg_id, 0))
            node_features[idx] = features.to_vector()
            node_ids.append(pkg_id)

        # Build edge index
        edges_source = []
        edges_target = []
        edge_features_list = []

        for dep in dep_graph.dependencies:
            if dep.source in node_to_idx and dep.target in node_to_idx:
                src_idx = node_to_idx[dep.source]
                tgt_idx = node_to_idx[dep.target]
                edges_source.append(src_idx)
                edges_target.append(tgt_idx)

                # Edge features
                edge_feat = self._extract_edge_features(dep)
                edge_features_list.append(edge_feat)

        num_edges = len(edges_source)

        if num_edges > 0:
            edge_index = np.array([edges_source, edges_target], dtype=np.int64)
            edge_features = np.array(edge_features_list, dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 4), dtype=np.float32)

        # Create labels: 1 if package has vulnerability or is affected by transitive vuln
        node_labels = self._compute_vulnerability_labels(dep_graph, node_to_idx)

        logger.info(f"Built graph: {num_nodes} nodes, {num_edges} edges")

        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_labels=node_labels,
            node_ids=node_ids,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

    def _extract_node_features(
        self, package: Package, dep_graph: DependencyGraph, depth: int
    ) -> NodeFeatures:
        """Extract features for a package node."""
        features = NodeFeatures()

        # Name embedding (simple hash-based for now)
        if self.use_name_embeddings:
            features.name_embedding = self._compute_name_embedding(package.name)

        # Version as numeric
        features.version_numeric = self._version_to_numeric(package.version)

        # Package type one-hot
        type_idx = self.PACKAGE_TYPE_INDEX.get(package.package_type, 9)
        features.package_type_onehot = np.zeros(10, dtype=np.float32)
        features.package_type_onehot[type_idx] = 1.0

        # Vulnerability features
        if package.vulnerabilities:
            features.has_vulnerability = 1
            features.vulnerability_count = len(package.vulnerabilities)
            cvss_scores = [v.cvss_score for v in package.vulnerabilities]
            features.max_cvss_score = max(cvss_scores) if cvss_scores else 0.0

            for vuln in package.vulnerabilities:
                severity = vuln.severity.upper() if vuln.severity else ""
                if severity == "CRITICAL" or vuln.cvss_score >= 9.0:
                    features.critical_count += 1
                elif severity == "HIGH" or vuln.cvss_score >= 7.0:
                    features.high_count += 1
                elif severity == "MEDIUM" or vuln.cvss_score >= 4.0:
                    features.medium_count += 1
                else:
                    features.low_count += 1

        # Dependency features
        direct_deps = dep_graph.get_direct_dependencies(package.id)
        features.direct_dependency_count = len(direct_deps)

        trans_deps = dep_graph.get_transitive_dependencies(package.id)
        features.transitive_dependency_count = len(trans_deps)

        dependents = dep_graph.get_dependents(package.id)
        features.dependent_count = len(dependents)

        # Supply chain position
        features.is_root = 1 if dep_graph.root_package == package.id else 0
        features.depth_from_root = depth
        features.is_leaf = 1 if features.direct_dependency_count == 0 else 0

        return features

    def _extract_edge_features(self, dep: Any) -> np.ndarray:
        """Extract features for a dependency edge."""
        # Edge feature vector:
        # [is_direct, is_transitive, is_dev, is_optional]
        features = np.zeros(4, dtype=np.float32)

        dep_type = dep.dependency_type.lower() if dep.dependency_type else ""
        features[0] = 1.0 if dep_type == "direct" else 0.0
        features[1] = 1.0 if dep_type == "transitive" else 0.0
        features[2] = 1.0 if dep_type == "dev" else 0.0
        features[3] = 1.0 if dep_type == "optional" else 0.0

        return features

    def _compute_depths(
        self, dep_graph: DependencyGraph, node_to_idx: dict[str, int]
    ) -> dict[str, int]:
        """Compute depth from root for each package using BFS."""
        depths: dict[str, int] = {}

        # Find roots (packages with no dependents)
        roots = set(dep_graph.packages.keys())
        for dep in dep_graph.dependencies:
            if dep.target in roots:
                roots.discard(dep.target)

        # If explicit root package, use that
        if dep_graph.root_package:
            roots = {dep_graph.root_package}

        # BFS from roots
        queue = list(roots)
        for root in roots:
            depths[root] = 0

        while queue:
            current = queue.pop(0)
            current_depth = depths.get(current, 0)

            for dep in dep_graph.dependencies:
                if dep.source == current:
                    if dep.target not in depths:
                        depths[dep.target] = current_depth + 1
                        queue.append(dep.target)

        # Handle disconnected components
        for pkg_id in dep_graph.packages:
            if pkg_id not in depths:
                depths[pkg_id] = 0

        return depths

    def _compute_vulnerability_labels(
        self, dep_graph: DependencyGraph, node_to_idx: dict[str, int]
    ) -> np.ndarray:
        """Compute vulnerability labels for nodes.

        Labels:
        - 0: No vulnerability
        - 1: Direct vulnerability
        - 2: Transitive vulnerability exposure
        """
        num_nodes = len(node_to_idx)
        labels = np.zeros(num_nodes, dtype=np.int64)

        # Mark directly vulnerable packages
        vulnerable_pkgs = set()
        for pkg in dep_graph.get_vulnerable_packages():
            idx = node_to_idx.get(pkg.id)
            if idx is not None:
                labels[idx] = 1
                vulnerable_pkgs.add(pkg.id)

        # Mark packages with transitive exposure
        for pkg_id in dep_graph.packages:
            if pkg_id in vulnerable_pkgs:
                continue

            # Check if any dependency is vulnerable
            trans_deps = dep_graph.get_transitive_dependencies(pkg_id)
            for dep_pkg in trans_deps:
                if dep_pkg.id in vulnerable_pkgs:
                    idx = node_to_idx.get(pkg_id)
                    if idx is not None and labels[idx] == 0:
                        labels[idx] = 2
                    break

        return labels

    def _compute_name_embedding(self, name: str) -> np.ndarray:
        """Compute a simple embedding for package name.

        Uses character-level hashing for efficiency.
        In production, this could use a pre-trained embedding model.
        """
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        # Simple hash-based embedding
        for i, char in enumerate(name.lower()):
            idx = hash(char + str(i)) % self.embedding_dim
            embedding[idx] += 1.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def _version_to_numeric(self, version: str) -> float:
        """Convert version string to a numeric value."""
        if not version:
            return 0.0

        # Parse semantic version (major.minor.patch)
        import re

        # Extract numeric parts
        parts = re.findall(r"\d+", version)
        if not parts:
            return 0.0

        # Weight by position: major * 1000000 + minor * 1000 + patch
        value = 0.0
        weights = [1000000, 1000, 1, 0.001]
        for i, part in enumerate(parts[:4]):
            try:
                value += int(part) * weights[i]
            except (ValueError, IndexError):
                pass

        # Normalize to [0, 1] range (assume max version ~1000.0.0)
        return min(value / 1000000000, 1.0)


def build_batch(graphs: list[GraphData]) -> GraphData:
    """Combine multiple graphs into a single batched graph.

    Args:
        graphs: List of graph data objects

    Returns:
        Single GraphData with batched nodes and edges
    """
    if not graphs:
        return GraphData()

    # Compute offsets for node indices
    node_offsets = [0]
    for g in graphs[:-1]:
        node_offsets.append(node_offsets[-1] + g.num_nodes)

    # Concatenate node features
    all_features = np.concatenate([g.node_features for g in graphs], axis=0)

    # Concatenate and adjust edge indices
    all_edges_src = []
    all_edges_tgt = []
    for g, offset in zip(graphs, node_offsets):
        if g.num_edges > 0:
            all_edges_src.extend((g.edge_index[0] + offset).tolist())
            all_edges_tgt.extend((g.edge_index[1] + offset).tolist())

    if all_edges_src:
        all_edge_index = np.array([all_edges_src, all_edges_tgt], dtype=np.int64)
    else:
        all_edge_index = np.zeros((2, 0), dtype=np.int64)

    # Concatenate edge features if present
    edge_features = None
    if all(g.edge_features is not None for g in graphs):
        valid_edge_features = [g.edge_features for g in graphs if g.num_edges > 0]
        if valid_edge_features:
            edge_features = np.concatenate(valid_edge_features, axis=0)

    # Concatenate labels
    node_labels = None
    if all(g.node_labels is not None for g in graphs):
        node_labels = np.concatenate([g.node_labels for g in graphs], axis=0)

    # Collect node IDs
    all_node_ids = []
    for g in graphs:
        all_node_ids.extend(g.node_ids)

    return GraphData(
        node_features=all_features,
        edge_index=all_edge_index,
        edge_features=edge_features,
        node_labels=node_labels,
        node_ids=all_node_ids,
        num_nodes=sum(g.num_nodes for g in graphs),
        num_edges=sum(g.num_edges for g in graphs),
    )


if __name__ == "__main__":
    # Test graph building
    from medtech_ai_security.sbom_analysis.parser import SBOMParser, create_sample_sbom

    sample_sbom = create_sample_sbom()
    parser = SBOMParser()
    dep_graph = parser.parse_json(sample_sbom)

    builder = SBOMGraphBuilder()
    graph_data = builder.build(dep_graph)

    print("[+] Built graph:")
    print(f"    Nodes: {graph_data.num_nodes}")
    print(f"    Edges: {graph_data.num_edges}")
    print(f"    Feature dim: {graph_data.node_features.shape[1]}")
    print(f"    Node IDs: {graph_data.node_ids}")

    print("\n[+] Node labels (vulnerability):")
    for i, (node_id, label) in enumerate(zip(graph_data.node_ids, graph_data.node_labels)):
        label_str = {0: "clean", 1: "vulnerable", 2: "transitive"}.get(label, "?")
        print(f"    {node_id}: {label_str}")

    print(f"\n[+] Edge index shape: {graph_data.edge_index.shape}")
