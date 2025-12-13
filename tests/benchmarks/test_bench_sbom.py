"""Benchmark tests for SBOM analysis module.

Run with:
    pytest tests/benchmarks/test_bench_sbom.py --benchmark-only
    pytest tests/benchmarks/test_bench_sbom.py --benchmark-compare
"""

import json
import pytest


@pytest.fixture
def small_sbom():
    """Generate a small SBOM with 50 components."""
    return generate_sbom(num_components=50)


@pytest.fixture
def medium_sbom():
    """Generate a medium SBOM with 200 components."""
    return generate_sbom(num_components=200)


@pytest.fixture
def large_sbom():
    """Generate a large SBOM with 500 components."""
    return generate_sbom(num_components=500)


def generate_sbom(num_components: int) -> dict:
    """Generate a synthetic CycloneDX SBOM for benchmarking."""
    import random

    random.seed(42)

    components = []
    for i in range(num_components):
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/package-{i}@{random.randint(1, 10)}.{random.randint(0, 20)}.{random.randint(0, 10)}",
            "name": f"package-{i}",
            "version": f"{random.randint(1, 10)}.{random.randint(0, 20)}.{random.randint(0, 10)}",
            "purl": f"pkg:pypi/package-{i}@{random.randint(1, 10)}.{random.randint(0, 20)}.{random.randint(0, 10)}",
            "licenses": [{"license": {"id": random.choice(["MIT", "Apache-2.0", "BSD-3-Clause", "GPL-3.0"])}}],
        }
        components.append(component)

    # Generate dependencies (tree structure)
    dependencies = []
    for i in range(num_components):
        num_deps = random.randint(0, min(5, num_components - i - 1))
        deps = [f"pkg:pypi/package-{j}@" for j in random.sample(range(i + 1, num_components), min(num_deps, num_components - i - 1))] if i < num_components - 1 else []
        dependencies.append({
            "ref": components[i]["bom-ref"],
            "dependsOn": deps[:num_deps]
        })

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "components": components,
        "dependencies": dependencies
    }


class TestSBOMParsingBenchmarks:
    """Benchmark suite for SBOM parsing operations."""

    def test_bench_parse_small_sbom(self, benchmark, small_sbom, tmp_path):
        """Benchmark parsing small SBOM file."""
        from medtech_ai_security.sbom_analysis.analyzer import SBOMParser

        sbom_file = tmp_path / "small_sbom.json"
        sbom_file.write_text(json.dumps(small_sbom))

        parser = SBOMParser()
        result = benchmark(parser.parse, str(sbom_file))
        assert result is not None
        assert len(result.components) == 50

    def test_bench_parse_medium_sbom(self, benchmark, medium_sbom, tmp_path):
        """Benchmark parsing medium SBOM file."""
        from medtech_ai_security.sbom_analysis.analyzer import SBOMParser

        sbom_file = tmp_path / "medium_sbom.json"
        sbom_file.write_text(json.dumps(medium_sbom))

        parser = SBOMParser()
        result = benchmark(parser.parse, str(sbom_file))
        assert len(result.components) == 200

    def test_bench_parse_large_sbom(self, benchmark, large_sbom, tmp_path):
        """Benchmark parsing large SBOM file."""
        from medtech_ai_security.sbom_analysis.analyzer import SBOMParser

        sbom_file = tmp_path / "large_sbom.json"
        sbom_file.write_text(json.dumps(large_sbom))

        parser = SBOMParser()
        result = benchmark(parser.parse, str(sbom_file))
        assert len(result.components) == 500


class TestGraphBuildingBenchmarks:
    """Benchmark suite for dependency graph operations."""

    def test_bench_build_graph_small(self, benchmark, small_sbom):
        """Benchmark graph building for small SBOM."""
        from medtech_ai_security.sbom_analysis.analyzer import DependencyGraphBuilder

        builder = DependencyGraphBuilder()
        result = benchmark(builder.build_from_cyclonedx, small_sbom)
        assert result is not None

    def test_bench_build_graph_large(self, benchmark, large_sbom):
        """Benchmark graph building for large SBOM."""
        from medtech_ai_security.sbom_analysis.analyzer import DependencyGraphBuilder

        builder = DependencyGraphBuilder()
        result = benchmark(builder.build_from_cyclonedx, large_sbom)
        assert result is not None

    def test_bench_graph_centrality(self, benchmark, medium_sbom):
        """Benchmark centrality calculation."""
        from medtech_ai_security.sbom_analysis.analyzer import DependencyGraphBuilder

        builder = DependencyGraphBuilder()
        graph = builder.build_from_cyclonedx(medium_sbom)

        result = benchmark(builder.calculate_centrality, graph)
        assert len(result) > 0

    def test_bench_find_critical_paths(self, benchmark, medium_sbom):
        """Benchmark critical path finding."""
        from medtech_ai_security.sbom_analysis.analyzer import DependencyGraphBuilder

        builder = DependencyGraphBuilder()
        graph = builder.build_from_cyclonedx(medium_sbom)

        result = benchmark(builder.find_critical_paths, graph)
        assert result is not None


class TestGNNBenchmarks:
    """Benchmark suite for GNN-based analysis."""

    @pytest.fixture
    def gnn_model(self):
        """Load or create a GNN model for benchmarking."""
        from medtech_ai_security.sbom_analysis.analyzer import VulnerabilityGNN

        model = VulnerabilityGNN(
            node_features=88,
            hidden_dim=64,
            num_classes=3
        )
        return model

    def test_bench_gnn_inference_small(self, benchmark, gnn_model, small_sbom):
        """Benchmark GNN inference on small graph."""
        from medtech_ai_security.sbom_analysis.analyzer import DependencyGraphBuilder

        builder = DependencyGraphBuilder()
        graph = builder.build_from_cyclonedx(small_sbom)
        graph_data = builder.to_torch_geometric(graph)

        result = benchmark(gnn_model.predict, graph_data)
        assert result is not None

    def test_bench_gnn_inference_large(self, benchmark, gnn_model, large_sbom):
        """Benchmark GNN inference on large graph."""
        from medtech_ai_security.sbom_analysis.analyzer import DependencyGraphBuilder

        builder = DependencyGraphBuilder()
        graph = builder.build_from_cyclonedx(large_sbom)
        graph_data = builder.to_torch_geometric(graph)

        result = benchmark(gnn_model.predict, graph_data)
        assert result is not None

    def test_bench_feature_extraction(self, benchmark, medium_sbom):
        """Benchmark node feature extraction."""
        from medtech_ai_security.sbom_analysis.analyzer import FeatureExtractor

        extractor = FeatureExtractor()

        result = benchmark(extractor.extract_node_features, medium_sbom["components"])
        assert result.shape[0] == 200


class TestRiskScoringBenchmarks:
    """Benchmark suite for risk scoring operations."""

    def test_bench_score_components(self, benchmark, medium_sbom):
        """Benchmark component risk scoring."""
        from medtech_ai_security.sbom_analysis.analyzer import RiskScorer

        scorer = RiskScorer()

        result = benchmark(scorer.score_components, medium_sbom["components"])
        assert len(result) == 200

    def test_bench_aggregate_risk(self, benchmark, large_sbom):
        """Benchmark aggregate risk calculation."""
        from medtech_ai_security.sbom_analysis.analyzer import RiskScorer

        scorer = RiskScorer()
        component_scores = scorer.score_components(large_sbom["components"])

        result = benchmark(scorer.aggregate_risk, component_scores)
        assert 0 <= result <= 10

    def test_bench_license_analysis(self, benchmark, medium_sbom):
        """Benchmark license compliance analysis."""
        from medtech_ai_security.sbom_analysis.analyzer import LicenseAnalyzer

        analyzer = LicenseAnalyzer()

        result = benchmark(analyzer.analyze, medium_sbom["components"])
        assert result is not None
