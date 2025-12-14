"""Benchmark tests for SBOM analysis module.

Run with:
    pytest tests/benchmarks/test_bench_sbom.py --benchmark-only
    pytest tests/benchmarks/test_bench_sbom.py --benchmark-compare
"""

import json
import random
import pytest

from medtech_ai_security.sbom_analysis import (
    SBOMParser,
    SBOMAnalyzer,
    SupplyChainRiskScorer,
)


def generate_sbom(num_components: int) -> dict:
    """Generate a synthetic CycloneDX SBOM for benchmarking."""
    random.seed(42)

    components = []
    for i in range(num_components):
        version = f"{random.randint(1, 10)}.{random.randint(0, 20)}.{random.randint(0, 10)}"
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/package-{i}@{version}",
            "name": f"package-{i}",
            "version": version,
            "purl": f"pkg:pypi/package-{i}@{version}",
            "licenses": [{"license": {"id": random.choice(["MIT", "Apache-2.0", "BSD-3-Clause", "GPL-3.0"])}}],
        }
        components.append(component)

    # Generate dependencies (tree structure)
    dependencies = []
    for i in range(num_components):
        num_deps = random.randint(0, min(5, num_components - i - 1))
        if i < num_components - 1:
            dep_indices = random.sample(range(i + 1, num_components), min(num_deps, num_components - i - 1))
            deps = [components[j]["bom-ref"] for j in dep_indices]
        else:
            deps = []
        dependencies.append({
            "ref": components[i]["bom-ref"],
            "dependsOn": deps
        })

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "components": components,
        "dependencies": dependencies
    }


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


@pytest.fixture
def parser():
    """Create an SBOM parser instance."""
    return SBOMParser()


@pytest.fixture
def analyzer():
    """Create an SBOM analyzer instance."""
    return SBOMAnalyzer(use_gnn=False)


@pytest.fixture
def risk_scorer():
    """Create a risk scorer instance."""
    return SupplyChainRiskScorer(medical_context=True)


class TestSBOMParsingBenchmarks:
    """Benchmark suite for SBOM parsing operations."""

    def test_bench_parse_small_sbom(self, benchmark, parser, small_sbom, tmp_path):
        """Benchmark parsing small SBOM file."""
        sbom_path = tmp_path / "small_sbom.json"
        sbom_path.write_text(json.dumps(small_sbom))

        def run_parse():
            return parser.parse(str(sbom_path))

        result = benchmark(run_parse)
        assert result is not None
        assert result.package_count > 0

    def test_bench_parse_medium_sbom(self, benchmark, parser, medium_sbom, tmp_path):
        """Benchmark parsing medium SBOM file."""
        sbom_path = tmp_path / "medium_sbom.json"
        sbom_path.write_text(json.dumps(medium_sbom))

        def run_parse():
            return parser.parse(str(sbom_path))

        result = benchmark(run_parse)
        assert result is not None
        assert result.package_count > 0

    def test_bench_parse_large_sbom(self, benchmark, parser, large_sbom, tmp_path):
        """Benchmark parsing large SBOM file."""
        sbom_path = tmp_path / "large_sbom.json"
        sbom_path.write_text(json.dumps(large_sbom))

        def run_parse():
            return parser.parse(str(sbom_path))

        result = benchmark(run_parse)
        assert result is not None
        assert result.package_count > 0

    def test_bench_parse_json_direct(self, benchmark, parser, medium_sbom):
        """Benchmark parsing SBOM from JSON string directly."""
        sbom_json = json.dumps(medium_sbom)

        def run_parse():
            return parser.parse_json(sbom_json)

        result = benchmark(run_parse)
        assert result is not None
        assert result.package_count > 0


class TestGraphBuildingBenchmarks:
    """Benchmark suite for dependency graph operations."""

    def test_bench_build_graph_small(self, benchmark, parser, small_sbom, tmp_path):
        """Benchmark graph building for small SBOM."""
        sbom_path = tmp_path / "small_sbom.json"
        sbom_path.write_text(json.dumps(small_sbom))

        def run_build():
            return parser.parse(str(sbom_path))

        result = benchmark(run_build)
        assert result is not None
        assert result.package_count > 0

    def test_bench_build_graph_large(self, benchmark, parser, large_sbom, tmp_path):
        """Benchmark graph building for large SBOM."""
        sbom_path = tmp_path / "large_sbom.json"
        sbom_path.write_text(json.dumps(large_sbom))

        def run_build():
            return parser.parse(str(sbom_path))

        result = benchmark(run_build)
        assert result is not None
        assert result.package_count > 0

    def test_bench_graph_package_count(self, benchmark, parser, medium_sbom, tmp_path):
        """Benchmark getting graph package count."""
        sbom_path = tmp_path / "medium_sbom.json"
        sbom_path.write_text(json.dumps(medium_sbom))
        graph = parser.parse(str(sbom_path))

        def run_count():
            return graph.package_count

        result = benchmark(run_count)
        assert result > 0

    def test_bench_graph_dependency_count(self, benchmark, parser, medium_sbom, tmp_path):
        """Benchmark getting graph dependency count."""
        sbom_path = tmp_path / "medium_sbom.json"
        sbom_path.write_text(json.dumps(medium_sbom))
        graph = parser.parse(str(sbom_path))

        def run_count():
            return graph.dependency_count

        result = benchmark(run_count)
        assert result >= 0


class TestRiskScoringBenchmarks:
    """Benchmark suite for risk scoring operations."""

    def test_bench_score_small_graph(self, benchmark, parser, risk_scorer, small_sbom, tmp_path):
        """Benchmark risk scoring for small SBOM."""
        sbom_path = tmp_path / "small_sbom.json"
        sbom_path.write_text(json.dumps(small_sbom))
        graph = parser.parse(str(sbom_path))

        def run_score():
            return risk_scorer.score(graph)

        result = benchmark(run_score)
        assert result is not None
        assert result.total_packages > 0

    def test_bench_score_medium_graph(self, benchmark, parser, risk_scorer, medium_sbom, tmp_path):
        """Benchmark risk scoring for medium SBOM."""
        sbom_path = tmp_path / "medium_sbom.json"
        sbom_path.write_text(json.dumps(medium_sbom))
        graph = parser.parse(str(sbom_path))

        def run_score():
            return risk_scorer.score(graph)

        result = benchmark(run_score)
        assert result is not None
        assert result.total_packages > 0

    def test_bench_score_large_graph(self, benchmark, parser, risk_scorer, large_sbom, tmp_path):
        """Benchmark risk scoring for large SBOM."""
        sbom_path = tmp_path / "large_sbom.json"
        sbom_path.write_text(json.dumps(large_sbom))
        graph = parser.parse(str(sbom_path))

        def run_score():
            return risk_scorer.score(graph)

        result = benchmark(run_score)
        assert result is not None
        assert result.total_packages > 0


class TestAnalyzerBenchmarks:
    """Benchmark suite for full analyzer operations."""

    def test_bench_analyze_small(self, benchmark, analyzer, small_sbom, tmp_path):
        """Benchmark full analysis for small SBOM."""
        sbom_path = tmp_path / "small_sbom.json"
        sbom_path.write_text(json.dumps(small_sbom))

        def run_analyze():
            return analyzer.analyze(str(sbom_path))

        result = benchmark(run_analyze)
        assert result is not None
        assert result.risk_report is not None

    def test_bench_analyze_medium(self, benchmark, analyzer, medium_sbom, tmp_path):
        """Benchmark full analysis for medium SBOM."""
        sbom_path = tmp_path / "medium_sbom.json"
        sbom_path.write_text(json.dumps(medium_sbom))

        def run_analyze():
            return analyzer.analyze(str(sbom_path))

        result = benchmark(run_analyze)
        assert result is not None
        assert result.risk_report is not None

    def test_bench_analyze_json_direct(self, benchmark, analyzer, medium_sbom):
        """Benchmark analysis from JSON string directly."""
        sbom_json = json.dumps(medium_sbom)

        def run_analyze():
            return analyzer.analyze_json(sbom_json)

        result = benchmark(run_analyze)
        assert result is not None
        assert result.risk_report is not None

    def test_bench_generate_html_report(self, benchmark, analyzer, small_sbom, tmp_path):
        """Benchmark HTML report generation."""
        sbom_path = tmp_path / "small_sbom.json"
        sbom_path.write_text(json.dumps(small_sbom))
        report = analyzer.analyze(str(sbom_path))

        def run_html():
            return analyzer.generate_html_report(report)

        result = benchmark(run_html)
        assert result is not None
        assert "<html" in result.lower()
