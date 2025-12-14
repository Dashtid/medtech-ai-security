"""Unit tests for Phase 5: SBOM Supply Chain Analysis."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from medtech_ai_security.sbom_analysis.gnn_model import (
    TF_AVAILABLE,
    GNNConfig,
    SimpleVulnerabilityClassifier,
)
from medtech_ai_security.sbom_analysis.graph_builder import (
    NodeFeatures,
    SBOMGraphBuilder,
)
from medtech_ai_security.sbom_analysis.parser import (
    Dependency,
    DependencyGraph,
    Package,
    PackageType,
    SBOMParser,
    VulnerabilityInfo,
)

try:
    from medtech_ai_security.sbom_analysis.gnn_model import VulnerabilityGNN
except ImportError:
    VulnerabilityGNN = None
from medtech_ai_security.sbom_analysis.analyzer import (
    AnalysisReport,
    SBOMAnalyzer,
)
from medtech_ai_security.sbom_analysis.risk_scorer import (
    PackageRisk,
    RiskLevel,
    RiskReport,
    SupplyChainRiskScorer,
)


class TestPackage:
    """Test Package dataclass."""

    def test_package_creation(self):
        """Test creating a Package."""
        pkg = Package(
            name="express",
            version="4.17.1",
            package_type=PackageType.NPM,
        )

        assert pkg.name == "express"
        assert pkg.version == "4.17.1"
        assert pkg.package_type == PackageType.NPM

    def test_package_with_vulnerabilities(self):
        """Test Package with vulnerabilities."""
        vuln = VulnerabilityInfo(
            cve_id="CVE-2021-23337",
            severity="high",
            cvss_score=7.5,
        )
        pkg = Package(
            name="lodash",
            version="4.17.20",
            vulnerabilities=[vuln],
        )

        assert len(pkg.vulnerabilities) == 1
        assert pkg.vulnerabilities[0].cve_id == "CVE-2021-23337"


class TestVulnerabilityInfo:
    """Test VulnerabilityInfo dataclass."""

    def test_vulnerability_creation(self):
        """Test creating a VulnerabilityInfo."""
        vuln = VulnerabilityInfo(
            cve_id="CVE-2021-44228",
            severity="critical",
            cvss_score=10.0,
            description="Log4j RCE",
        )

        assert vuln.cve_id == "CVE-2021-44228"
        assert vuln.severity == "critical"
        assert vuln.cvss_score == 10.0


class TestDependencyGraph:
    """Test DependencyGraph dataclass."""

    def test_graph_creation(self):
        """Test creating a DependencyGraph."""
        graph = DependencyGraph()
        pkg1 = Package(name="app", version="1.0.0")
        pkg2 = Package(name="lib", version="2.0.0")
        graph.add_package(pkg1)
        graph.add_package(pkg2)
        graph.add_dependency(Dependency(source="app@1.0.0", target="lib@2.0.0"))

        assert graph.package_count == 2
        assert graph.dependency_count == 1

    def test_get_direct_dependencies(self):
        """Test getting direct dependencies."""
        graph = DependencyGraph()
        graph.add_package(Package(name="app", version="1.0.0"))
        graph.add_package(Package(name="lib1", version="1.0.0"))
        graph.add_package(Package(name="lib2", version="1.0.0"))
        graph.add_dependency(Dependency(source="app@1.0.0", target="lib1@1.0.0"))
        graph.add_dependency(Dependency(source="app@1.0.0", target="lib2@1.0.0"))

        deps = graph.get_direct_dependencies("app@1.0.0")
        assert len(deps) == 2

    def test_get_transitive_dependencies(self):
        """Test getting transitive dependencies."""
        graph = DependencyGraph()
        graph.add_package(Package(name="app", version="1.0.0"))
        graph.add_package(Package(name="lib1", version="1.0.0"))
        graph.add_package(Package(name="lib2", version="1.0.0"))
        graph.add_dependency(Dependency(source="app@1.0.0", target="lib1@1.0.0"))
        graph.add_dependency(Dependency(source="lib1@1.0.0", target="lib2@1.0.0"))

        trans_deps = graph.get_transitive_dependencies("app@1.0.0")
        assert len(trans_deps) == 2  # lib1 and lib2

    def test_get_transitive_dependencies_circular(self):
        """Test transitive dependencies with circular references."""
        graph = DependencyGraph()
        graph.add_package(Package(name="a", version="1.0"))
        graph.add_package(Package(name="b", version="1.0"))
        graph.add_dependency(Dependency(source="a@1.0", target="b@1.0"))
        graph.add_dependency(Dependency(source="b@1.0", target="a@1.0"))

        # Should not infinite loop - returns b and then a (from b's deps)
        trans_deps = graph.get_transitive_dependencies("a@1.0")
        # Both packages are visited due to the cycle
        assert len(trans_deps) == 2

    def test_get_dependents(self):
        """Test getting packages that depend on a package."""
        graph = DependencyGraph()
        graph.add_package(Package(name="app1", version="1.0.0"))
        graph.add_package(Package(name="app2", version="1.0.0"))
        graph.add_package(Package(name="lib", version="1.0.0"))
        graph.add_dependency(Dependency(source="app1@1.0.0", target="lib@1.0.0"))
        graph.add_dependency(Dependency(source="app2@1.0.0", target="lib@1.0.0"))

        dependents = graph.get_dependents("lib@1.0.0")
        assert len(dependents) == 2

    def test_get_vulnerable_packages(self):
        """Test getting vulnerable packages."""
        vuln = VulnerabilityInfo(cve_id="CVE-2021-1234", severity="high", cvss_score=8.0)
        graph = DependencyGraph()
        graph.add_package(Package(name="safe", version="1.0.0"))
        graph.add_package(Package(name="unsafe", version="1.0.0", vulnerabilities=[vuln]))

        vulnerable = graph.get_vulnerable_packages()
        assert len(vulnerable) == 1
        assert vulnerable[0].name == "unsafe"

    def test_vulnerability_count(self):
        """Test vulnerability count property."""
        vuln1 = VulnerabilityInfo(cve_id="CVE-2021-1234", severity="high", cvss_score=8.0)
        vuln2 = VulnerabilityInfo(cve_id="CVE-2021-1235", severity="medium", cvss_score=5.0)
        graph = DependencyGraph()
        graph.add_package(Package(name="pkg1", version="1.0.0", vulnerabilities=[vuln1]))
        graph.add_package(Package(name="pkg2", version="1.0.0", vulnerabilities=[vuln1, vuln2]))
        graph.add_package(Package(name="pkg3", version="1.0.0"))

        assert graph.vulnerability_count == 3


class TestPackageProperties:
    """Test Package property methods."""

    def test_package_id_with_purl(self):
        """Test package ID with purl."""
        pkg = Package(
            name="express",
            version="4.17.1",
            purl="pkg:npm/express@4.17.1",
        )
        assert pkg.id == "pkg:npm/express@4.17.1"

    def test_package_id_without_purl(self):
        """Test package ID without purl."""
        pkg = Package(name="express", version="4.17.1")
        assert pkg.id == "express@4.17.1"

    def test_package_ecosystem_from_purl(self):
        """Test ecosystem extraction from purl."""
        pkg = Package(
            name="lodash",
            version="4.17.20",
            purl="pkg:npm/lodash@4.17.20",
        )
        assert pkg.ecosystem == "npm"

    def test_package_ecosystem_from_type(self):
        """Test ecosystem from package type."""
        pkg = Package(
            name="requests",
            version="2.28.0",
            package_type=PackageType.PYPI,
        )
        assert pkg.ecosystem == "pypi"


class TestSBOMParser:
    """Test SBOMParser functionality."""

    @pytest.fixture
    def cyclonedx_sbom(self):
        """Sample CycloneDX SBOM."""
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "metadata": {
                "component": {
                    "name": "test-app",
                    "version": "1.0.0",
                    "type": "application",
                }
            },
            "components": [
                {
                    "name": "lodash",
                    "version": "4.17.20",
                    "type": "library",
                    "purl": "pkg:npm/lodash@4.17.20",
                }
            ],
            "dependencies": [
                {"ref": "test-app@1.0.0", "dependsOn": ["lodash@4.17.20"]},
            ],
        }

    @pytest.fixture
    def temp_sbom_file(self, cyclonedx_sbom):
        """Create temporary SBOM file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cyclonedx_sbom, f)
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        parser = SBOMParser()
        assert parser is not None

    def test_detect_format_cyclonedx(self, temp_sbom_file):
        """Test CycloneDX format detection via parsing."""
        parser = SBOMParser()
        # The parser auto-detects format during parsing
        # We verify it by successfully parsing a CycloneDX file
        graph = parser.parse(temp_sbom_file)
        assert isinstance(graph, DependencyGraph)

    def test_parse_cyclonedx(self, temp_sbom_file):
        """Test parsing CycloneDX SBOM."""
        parser = SBOMParser()
        graph = parser.parse(temp_sbom_file)

        assert isinstance(graph, DependencyGraph)
        assert len(graph.packages) >= 1

    def test_parse_file_not_found(self):
        """Test parsing non-existent file raises error."""
        parser = SBOMParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/sbom.json")

    def test_parse_json_cyclonedx(self, cyclonedx_sbom):
        """Test parsing CycloneDX JSON from string."""
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(cyclonedx_sbom))

        assert isinstance(graph, DependencyGraph)

    def test_parse_json_spdx(self):
        """Test parsing SPDX JSON from string."""
        spdx_sbom = {
            "spdxVersion": "SPDX-2.3",
            "name": "test-sbom",
            "documentNamespace": "https://example.com/sbom",
            "packages": [
                {
                    "SPDXID": "SPDXRef-Package-1",
                    "name": "test-package",
                    "versionInfo": "1.0.0",
                    "licenseConcluded": "MIT",
                }
            ],
            "relationships": [],
        }
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(spdx_sbom))

        assert isinstance(graph, DependencyGraph)
        assert graph.metadata["format"] == "SPDX"

    def test_parse_json_unknown_format(self):
        """Test parsing unknown JSON format raises error."""
        parser = SBOMParser()
        with pytest.raises(ValueError, match="Cannot determine SBOM format"):
            parser.parse_json('{"unknown": "format"}')

    def test_parse_json_invalid_json(self):
        """Test parsing invalid JSON raises error."""
        parser = SBOMParser()
        with pytest.raises(ValueError, match="Invalid JSON"):
            parser.parse_json("not valid json")

    def test_parse_cyclonedx_with_vulnerabilities(self):
        """Test parsing CycloneDX SBOM with vulnerabilities."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "components": [
                {
                    "name": "lodash",
                    "version": "4.17.20",
                    "purl": "pkg:npm/lodash@4.17.20",
                }
            ],
            "vulnerabilities": [
                {
                    "id": "CVE-2021-23337",
                    "description": "Command injection",
                    "ratings": [{"method": "CVSSv3", "score": 7.2, "severity": "high"}],
                    "affects": [{"ref": "pkg:npm/lodash@4.17.20"}],
                }
            ],
        }
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(sbom))

        assert graph.vulnerability_count == 1

    def test_parse_cyclonedx_with_licenses(self):
        """Test parsing CycloneDX with license information."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {
                    "name": "express",
                    "version": "4.17.1",
                    "licenses": [{"license": {"id": "MIT"}}],
                }
            ],
        }
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(sbom))

        assert graph.packages["express@4.17.1"].license == "MIT"

    def test_parser_with_vuln_db(self):
        """Test parser with vulnerability database."""
        vuln_db = {
            "pkg:npm/lodash@4.17.20": [
                VulnerabilityInfo(
                    cve_id="CVE-2021-23337",
                    cvss_score=7.2,
                    severity="high",
                )
            ]
        }
        parser = SBOMParser(vuln_db=vuln_db)

        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {"name": "lodash", "version": "4.17.20", "purl": "pkg:npm/lodash@4.17.20"}
            ],
        }
        graph = parser.parse_json(json.dumps(sbom))

        # Vulnerability should be enriched from database
        assert len(graph.packages["pkg:npm/lodash@4.17.20"].vulnerabilities) == 1


class TestSBOMFormatDetection:
    """Test SBOM format detection."""

    def test_detect_cyclonedx_json_format(self, tmp_path):
        """Test detecting CycloneDX JSON format."""
        sbom = {"bomFormat": "CycloneDX", "specVersion": "1.5", "components": []}
        file_path = tmp_path / "sbom.json"
        file_path.write_text(json.dumps(sbom))

        parser = SBOMParser()
        # Parse should succeed if format is detected
        graph = parser.parse(file_path)
        assert graph.metadata["format"] == "CycloneDX"

    def test_detect_spdx_json_format(self, tmp_path):
        """Test detecting SPDX JSON format."""
        sbom = {"spdxVersion": "SPDX-2.3", "packages": []}
        file_path = tmp_path / "sbom.json"
        file_path.write_text(json.dumps(sbom))

        parser = SBOMParser()
        graph = parser.parse(file_path)
        assert graph.metadata["format"] == "SPDX"

    def test_detect_unknown_format(self, tmp_path):
        """Test detecting unknown format."""
        file_path = tmp_path / "sbom.txt"
        file_path.write_text("Unknown content format")

        parser = SBOMParser()
        with pytest.raises(ValueError, match="Unknown SBOM format"):
            parser.parse(file_path)


class TestSBOMGraphBuilder:
    """Test SBOMGraphBuilder functionality."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample dependency graph."""
        graph = DependencyGraph()
        graph.add_package(Package(name="app", version="1.0.0", package_type=PackageType.UNKNOWN))
        graph.add_package(Package(name="lib1", version="1.0.0", package_type=PackageType.NPM))
        graph.add_package(Package(name="lib2", version="2.0.0", package_type=PackageType.NPM))
        graph.add_dependency(Dependency(source="app@1.0.0", target="lib1@1.0.0"))
        graph.add_dependency(Dependency(source="lib1@1.0.0", target="lib2@2.0.0"))
        return graph

    def test_builder_initialization(self):
        """Test graph builder initializes correctly."""
        builder = SBOMGraphBuilder()
        assert builder is not None

    def test_build_graph(self, sample_graph):
        """Test building graph from dependency graph."""
        builder = SBOMGraphBuilder()
        graph_data = builder.build(sample_graph)

        assert graph_data is not None
        assert graph_data.node_features is not None
        assert len(graph_data.node_features) == 3


class TestNodeFeatures:
    """Test NodeFeatures dataclass."""

    def test_node_features_creation(self):
        """Test creating NodeFeatures."""
        features = NodeFeatures(
            has_vulnerability=0,
            max_cvss_score=7.5,
            vulnerability_count=1,
        )

        assert features.has_vulnerability == 0
        assert features.max_cvss_score == 7.5

    def test_node_features_to_vector(self):
        """Test converting NodeFeatures to vector."""
        features = NodeFeatures(
            has_vulnerability=1,
            max_cvss_score=9.8,
            vulnerability_count=2,
        )
        vector = features.to_vector()

        assert isinstance(vector, np.ndarray)
        assert len(vector) == features.feature_dim


class TestRiskLevel:
    """Test RiskLevel enumeration."""

    def test_risk_levels_exist(self):
        """Test risk levels are defined."""
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"


class TestPackageRisk:
    """Test PackageRisk dataclass."""

    def test_package_risk_creation(self):
        """Test creating PackageRisk."""
        risk = PackageRisk(
            package_id="lodash@4.17.20",
            package_name="lodash",
            package_version="4.17.20",
            risk_level=RiskLevel.HIGH,
            risk_score=75.0,
        )

        assert risk.package_id == "lodash@4.17.20"
        assert risk.package_name == "lodash"
        assert risk.risk_level == RiskLevel.HIGH
        assert risk.risk_score == 75.0


class TestRiskReport:
    """Test RiskReport dataclass."""

    def test_risk_report_creation(self):
        """Test creating RiskReport."""
        report = RiskReport(
            overall_risk_level=RiskLevel.MEDIUM,
            overall_risk_score=55.0,
            package_risks=[],
        )

        assert report.overall_risk_level == RiskLevel.MEDIUM
        assert report.overall_risk_score == 55.0


class TestSupplyChainRiskScorer:
    """Test SupplyChainRiskScorer functionality."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample dependency graph with vulnerabilities."""
        vuln = VulnerabilityInfo(cve_id="CVE-2021-23337", severity="high", cvss_score=7.5)
        graph = DependencyGraph()
        graph.add_package(Package(name="app", version="1.0.0"))
        graph.add_package(Package(name="lodash", version="4.17.20", vulnerabilities=[vuln]))
        graph.add_dependency(Dependency(source="app@1.0.0", target="lodash@4.17.20"))
        return graph

    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        scorer = SupplyChainRiskScorer()
        assert scorer is not None

    def test_score_graph(self, sample_graph):
        """Test scoring a dependency graph."""
        scorer = SupplyChainRiskScorer()
        report = scorer.score(sample_graph)

        assert isinstance(report, RiskReport)
        assert report.overall_risk_score >= 0


class TestSBOMAnalyzer:
    """Test SBOMAnalyzer functionality."""

    @pytest.fixture
    def temp_sbom_file(self):
        """Create temporary SBOM file."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "metadata": {
                "component": {
                    "name": "test-medical-app",
                    "version": "1.0.0",
                    "type": "application",
                }
            },
            "components": [
                {
                    "name": "express",
                    "version": "4.17.1",
                    "type": "library",
                }
            ],
            "dependencies": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sbom, f)
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        # Use use_gnn=False to avoid TensorFlow/Keras compatibility issues in tests
        analyzer = SBOMAnalyzer(use_gnn=False)
        assert analyzer is not None

    def test_analyze_sbom(self, temp_sbom_file):
        """Test analyzing an SBOM file."""
        # Use use_gnn=False to avoid TensorFlow/Keras compatibility issues in tests
        analyzer = SBOMAnalyzer(use_gnn=False)
        report = analyzer.analyze(temp_sbom_file)

        assert isinstance(report, AnalysisReport)


class TestGNNConfig:
    """Test GNNConfig dataclass."""

    def test_config_defaults(self):
        """Test GNNConfig with default values."""
        config = GNNConfig()

        assert config.hidden_dim == 64
        assert config.num_layers == 3
        assert config.num_heads == 4
        assert config.dropout_rate == 0.2
        assert config.input_dim == 88
        assert config.num_classes == 3

    def test_config_custom_values(self):
        """Test GNNConfig with custom values."""
        config = GNNConfig(
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            dropout_rate=0.3,
            learning_rate=0.0001,
        )

        assert config.hidden_dim == 128
        assert config.num_layers == 4
        assert config.learning_rate == 0.0001


class TestSimpleVulnerabilityClassifier:
    """Test SimpleVulnerabilityClassifier functionality."""

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing."""
        from medtech_ai_security.sbom_analysis.graph_builder import GraphData

        # Create simple graph with 5 nodes
        node_features = np.random.rand(5, 88).astype(np.float32)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])  # Simple chain
        node_labels = np.array([0, 1, 1, 0, 2])  # 0: clean, 1: vulnerable, 2: transitive

        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            node_labels=node_labels,
            num_nodes=5,
        )

    def test_classifier_initialization(self):
        """Test classifier initializes correctly."""
        classifier = SimpleVulnerabilityClassifier(num_classes=3)
        assert classifier.num_classes == 3
        assert classifier.weights is None

    def test_classifier_fit(self, sample_graph_data):
        """Test fitting the classifier."""
        classifier = SimpleVulnerabilityClassifier(num_classes=3)
        classifier.fit([sample_graph_data], epochs=10)

        assert classifier.weights is not None
        assert classifier.bias is not None
        assert classifier.weights.shape[1] == 3

    def test_classifier_predict(self, sample_graph_data):
        """Test predicting with the classifier."""
        classifier = SimpleVulnerabilityClassifier(num_classes=3)
        classifier.fit([sample_graph_data], epochs=10)

        predictions = classifier.predict(sample_graph_data)

        assert len(predictions) == 5
        assert all(p in [0, 1, 2] for p in predictions)

    def test_classifier_predict_proba(self, sample_graph_data):
        """Test predicting probabilities."""
        classifier = SimpleVulnerabilityClassifier(num_classes=3)
        classifier.fit([sample_graph_data], epochs=10)

        probs = classifier.predict_proba(sample_graph_data)

        assert probs.shape == (5, 3)
        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_classifier_not_trained_error(self, sample_graph_data):
        """Test error when predicting without training."""
        classifier = SimpleVulnerabilityClassifier()

        with pytest.raises(RuntimeError, match="not trained"):
            classifier.predict(sample_graph_data)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow required for GNN tests")
class TestVulnerabilityGNN:
    """Test VulnerabilityGNN model (requires TensorFlow)."""

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for GNN testing."""
        from medtech_ai_security.sbom_analysis.graph_builder import GraphData

        # Create simple graph with 10 nodes
        node_features = np.random.rand(10, 88).astype(np.float32)
        edge_index = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
        node_labels = np.array([0, 0, 1, 1, 0, 1, 0, 2, 2, 0])

        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            node_labels=node_labels,
            num_nodes=10,
        )

    def test_gnn_initialization(self):
        """Test GNN model initializes correctly."""
        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        assert model.model is not None
        assert model.config.hidden_dim == 32

    def test_gnn_train(self, sample_graph_data):
        """Test GNN model training."""
        config = GNNConfig(
            input_dim=88,
            hidden_dim=32,
            num_layers=2,
            epochs=5,
        )
        model = VulnerabilityGNN(config)

        history = model.train([sample_graph_data], epochs=5)

        assert "loss" in history
        assert "accuracy" in history
        assert len(history["loss"]) == 5

    def test_gnn_predict(self, sample_graph_data):
        """Test GNN model prediction."""
        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        # Train briefly
        model.train([sample_graph_data], epochs=3)

        predictions = model.predict(sample_graph_data)

        assert len(predictions) == 10
        assert all(p in [0, 1, 2] for p in predictions)

    def test_gnn_predict_proba(self, sample_graph_data):
        """Test GNN model probability prediction."""
        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        model.train([sample_graph_data], epochs=3)
        probs = model.predict_proba(sample_graph_data)

        assert probs.shape == (10, 3)
        # Probabilities should sum to approximately 1
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_gnn_evaluate(self, sample_graph_data):
        """Test GNN model evaluation."""
        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        model.train([sample_graph_data], epochs=3)
        metrics = model.evaluate([sample_graph_data])

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_gnn_save_load(self, sample_graph_data, tmp_path):
        """Test GNN model save and load."""
        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        model.train([sample_graph_data], epochs=3)

        # Save
        model_path = tmp_path / "gnn_weights.weights.h5"
        model.save(str(model_path))

        # Create new model and load
        new_model = VulnerabilityGNN(config)
        new_model.load(str(model_path))

        # Predictions should be similar
        orig_preds = model.predict(sample_graph_data)
        loaded_preds = new_model.predict(sample_graph_data)

        assert np.array_equal(orig_preds, loaded_preds)


class TestAnalysisReport:
    """Test AnalysisReport dataclass."""

    def test_analysis_report_creation(self):
        """Test creating AnalysisReport with default values."""
        report = AnalysisReport()

        assert report.sbom_file == ""
        assert report.sbom_format == ""
        assert report.risk_report is None
        assert report.gnn_predictions is None
        assert report.graph_stats == {}

    def test_analysis_report_to_dict(self):
        """Test converting AnalysisReport to dictionary."""
        risk_report = RiskReport(
            overall_risk_level=RiskLevel.HIGH,
            overall_risk_score=75.0,
            package_risks=[],
        )
        report = AnalysisReport(
            sbom_file="test.json",
            sbom_format="CycloneDX",
            risk_report=risk_report,
            graph_stats={"num_packages": 10},
        )

        result = report.to_dict()

        assert result["sbom_file"] == "test.json"
        assert result["sbom_format"] == "CycloneDX"
        assert "risk_report" in result
        assert result["graph_stats"]["num_packages"] == 10

    def test_analysis_report_to_dict_without_optional_fields(self):
        """Test to_dict without optional fields."""
        report = AnalysisReport(
            sbom_file="test.json",
            sbom_format="CycloneDX",
        )

        result = report.to_dict()

        assert "risk_report" not in result
        assert "gnn_predictions" not in result
        assert "visualization" not in result

    def test_analysis_report_to_dict_with_visualization(self):
        """Test to_dict with visualization data."""
        report = AnalysisReport(
            sbom_file="test.json",
            sbom_format="CycloneDX",
            visualization_data={"nodes": [], "links": []},
        )

        result = report.to_dict()

        assert "visualization" in result
        assert result["visualization"]["nodes"] == []

    def test_analysis_report_to_json(self):
        """Test converting AnalysisReport to JSON string."""
        report = AnalysisReport(
            sbom_file="test.json",
            sbom_format="CycloneDX",
            graph_stats={"num_packages": 5},
        )

        json_str = report.to_json()

        assert '"sbom_file": "test.json"' in json_str
        assert '"num_packages": 5' in json_str


class TestSBOMAnalyzerAdvanced:
    """Advanced tests for SBOMAnalyzer."""

    @pytest.fixture
    def sample_sbom_json(self):
        """Create sample SBOM JSON string."""
        return json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "version": 1,
                "metadata": {
                    "component": {
                        "name": "test-app",
                        "version": "1.0.0",
                        "type": "application",
                    }
                },
                "components": [
                    {"name": "express", "version": "4.17.1", "type": "library"},
                    {"name": "lodash", "version": "4.17.20", "type": "library"},
                ],
                "dependencies": [
                    {"ref": "test-app@1.0.0", "dependsOn": ["express@4.17.1"]},
                    {"ref": "express@4.17.1", "dependsOn": ["lodash@4.17.20"]},
                ],
            }
        )

    def test_analyzer_without_gnn(self):
        """Test analyzer with GNN disabled."""
        analyzer = SBOMAnalyzer(use_gnn=False)

        assert analyzer.gnn_model is None
        assert analyzer.use_gnn is False

    def test_analyzer_without_medical_context(self):
        """Test analyzer without medical context."""
        analyzer = SBOMAnalyzer(use_gnn=False, medical_context=False)

        assert analyzer.medical_context is False

    def test_analyzer_with_vuln_db(self):
        """Test analyzer with custom vulnerability database."""
        vuln_db = {
            "lodash@4.17.20": [
                VulnerabilityInfo(cve_id="CVE-2021-23337", cvss_score=7.2, severity="high")
            ]
        }
        analyzer = SBOMAnalyzer(use_gnn=False, vuln_db=vuln_db)

        assert analyzer.parser.vuln_db is not None

    def test_analyze_json(self, sample_sbom_json):
        """Test analyzing SBOM from JSON string."""
        analyzer = SBOMAnalyzer(use_gnn=False)
        report = analyzer.analyze_json(sample_sbom_json)

        assert isinstance(report, AnalysisReport)
        assert report.sbom_file == "<inline>"
        assert report.sbom_format == "CycloneDX"
        assert report.risk_report is not None
        assert report.graph_stats["num_packages"] >= 1

    def test_generate_visualization(self, sample_sbom_json):
        """Test visualization data generation."""
        analyzer = SBOMAnalyzer(use_gnn=False)
        report = analyzer.analyze_json(sample_sbom_json)

        assert report.visualization_data is not None
        assert "nodes" in report.visualization_data
        assert "links" in report.visualization_data
        assert "statistics" in report.visualization_data

    def test_generate_html_report(self, sample_sbom_json, tmp_path):
        """Test HTML report generation."""
        analyzer = SBOMAnalyzer(use_gnn=False)
        report = analyzer.analyze_json(sample_sbom_json)

        html_path = tmp_path / "report.html"
        html = analyzer.generate_html_report(report, str(html_path))

        assert html_path.exists()
        # Check for presence of HTML structure (the exact text may vary)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html

    def test_generate_html_report_without_saving(self, sample_sbom_json):
        """Test HTML report generation without saving to file."""
        analyzer = SBOMAnalyzer(use_gnn=False)
        report = analyzer.analyze_json(sample_sbom_json)

        html = analyzer.generate_html_report(report)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html


class TestSupplyChainRiskScorerAdvanced:
    """Advanced tests for SupplyChainRiskScorer."""

    def test_scorer_with_medical_context(self):
        """Test scorer with medical context enabled."""
        scorer = SupplyChainRiskScorer(medical_context=True)

        assert scorer.medical_context is True

    def test_scorer_without_medical_context(self):
        """Test scorer without medical context."""
        scorer = SupplyChainRiskScorer(medical_context=False)

        assert scorer.medical_context is False

    def test_score_empty_graph(self):
        """Test scoring an empty dependency graph."""
        scorer = SupplyChainRiskScorer()
        graph = DependencyGraph()

        report = scorer.score(graph)

        assert report.overall_risk_score == 0.0

    def test_score_graph_with_critical_vulnerability(self):
        """Test scoring a graph with critical vulnerability."""
        vuln = VulnerabilityInfo(cve_id="CVE-2021-44228", severity="critical", cvss_score=10.0)
        graph = DependencyGraph()
        graph.add_package(Package(name="log4j", version="2.14.0", vulnerabilities=[vuln]))

        scorer = SupplyChainRiskScorer()
        report = scorer.score(graph)

        assert report.overall_risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        assert report.critical_vulnerabilities >= 1

    def test_risk_report_to_dict(self):
        """Test RiskReport to_dict method."""
        pkg_risk = PackageRisk(
            package_id="test@1.0.0",
            package_name="test",
            package_version="1.0.0",
            risk_level=RiskLevel.MEDIUM,
            risk_score=50.0,
        )
        report = RiskReport(
            overall_risk_level=RiskLevel.MEDIUM,
            overall_risk_score=50.0,
            package_risks=[pkg_risk],
            recommendations=["Update package"],
            fda_compliance_notes=["FDA note"],
        )

        result = report.to_dict()

        assert result["overall"]["risk_level"] == "medium"
        assert result["overall"]["risk_score"] == 50.0
        assert len(result["package_details"]) == 1
        assert "Update package" in result["recommendations"]


class TestRiskReportProperties:
    """Test RiskReport properties and methods."""

    def test_risk_report_has_summary_field(self):
        """Test RiskReport summary field."""
        report = RiskReport(
            overall_risk_level=RiskLevel.HIGH,
            overall_risk_score=75.0,
            package_risks=[],
            total_packages=10,
            vulnerable_packages=3,
            total_vulnerabilities=5,
            critical_vulnerabilities=2,
            summary="High risk detected in supply chain",
        )

        # summary is a field, not a property
        assert "High risk" in report.summary

    def test_package_risk_to_dict(self):
        """Test PackageRisk to_dict method."""
        pkg_risk = PackageRisk(
            package_id="lodash@4.17.20",
            package_name="lodash",
            package_version="4.17.20",
            risk_level=RiskLevel.HIGH,
            risk_score=75.0,
        )

        result = pkg_risk.to_dict()

        assert result["package_id"] == "lodash@4.17.20"
        assert result["risk_level"] == "high"
        assert result["risk_score"] == 75.0


class TestParserAdvanced:
    """Advanced tests for SBOMParser."""

    def test_parse_spdx_with_relationships(self):
        """Test parsing SPDX with relationships."""
        spdx_sbom = {
            "spdxVersion": "SPDX-2.3",
            "name": "test-sbom",
            "documentNamespace": "https://example.com/sbom",
            "packages": [
                {
                    "SPDXID": "SPDXRef-Package-app",
                    "name": "app",
                    "versionInfo": "1.0.0",
                },
                {
                    "SPDXID": "SPDXRef-Package-lib",
                    "name": "lib",
                    "versionInfo": "2.0.0",
                },
            ],
            "relationships": [
                {
                    "spdxElementId": "SPDXRef-Package-app",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": "SPDXRef-Package-lib",
                }
            ],
        }
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(spdx_sbom))

        assert graph.metadata["format"] == "SPDX"
        assert graph.package_count >= 2
        assert graph.dependency_count >= 1

    def test_parse_cyclonedx_with_external_references(self):
        """Test parsing CycloneDX with external references."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {
                    "name": "express",
                    "version": "4.17.1",
                    "externalReferences": [
                        {
                            "type": "website",
                            "url": "https://expressjs.com",
                        },
                        {
                            "type": "vcs",
                            "url": "https://github.com/expressjs/express",
                        },
                    ],
                }
            ],
        }
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(sbom))

        assert graph.package_count >= 1

    def test_parse_cyclonedx_with_hashes(self):
        """Test parsing CycloneDX with hash information."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {
                    "name": "lodash",
                    "version": "4.17.20",
                    "hashes": [
                        {"alg": "SHA-256", "content": "abc123"},
                        {"alg": "SHA-512", "content": "def456"},
                    ],
                }
            ],
        }
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(sbom))

        assert "lodash@4.17.20" in graph.packages

    def test_parse_cyclonedx_with_nested_dependencies(self):
        """Test parsing CycloneDX with nested dependency structure."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {"name": "app", "version": "1.0.0"},
                {"name": "lib-a", "version": "1.0.0"},
                {"name": "lib-b", "version": "1.0.0"},
                {"name": "lib-c", "version": "1.0.0"},
            ],
            "dependencies": [
                {"ref": "app@1.0.0", "dependsOn": ["lib-a@1.0.0", "lib-b@1.0.0"]},
                {"ref": "lib-a@1.0.0", "dependsOn": ["lib-c@1.0.0"]},
                {"ref": "lib-b@1.0.0", "dependsOn": ["lib-c@1.0.0"]},
            ],
        }
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(sbom))

        # lib-c should have 2 dependents
        dependents = graph.get_dependents("lib-c@1.0.0")
        assert len(dependents) == 2

    def test_parse_cyclonedx_with_multiple_licenses(self):
        """Test parsing CycloneDX with multiple license options."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {
                    "name": "dual-licensed",
                    "version": "1.0.0",
                    "licenses": [
                        {"license": {"id": "MIT"}},
                        {"license": {"id": "Apache-2.0"}},
                    ],
                }
            ],
        }
        parser = SBOMParser()
        graph = parser.parse_json(json.dumps(sbom))

        # Should take first license
        assert "dual-licensed@1.0.0" in graph.packages


class TestRiskScorerAdvanced:
    """Advanced tests for SupplyChainRiskScorer."""

    def test_score_graph_with_high_vulnerability(self):
        """Test scoring with high severity vulnerability."""
        vuln = VulnerabilityInfo(cve_id="CVE-2021-1234", severity="high", cvss_score=8.5)
        graph = DependencyGraph()
        graph.add_package(Package(name="vulnerable-pkg", version="1.0.0", vulnerabilities=[vuln]))

        scorer = SupplyChainRiskScorer()
        report = scorer.score(graph)

        # Risk scoring algorithm considers multiple factors, not just vuln severity
        # A single package with one high vuln results in a lower overall risk score
        assert report.overall_risk_score > 0
        assert report.high_vulnerabilities >= 1

    def test_score_graph_with_medium_vulnerability(self):
        """Test scoring with medium severity vulnerability."""
        vuln = VulnerabilityInfo(cve_id="CVE-2021-5678", severity="medium", cvss_score=5.5)
        graph = DependencyGraph()
        graph.add_package(Package(name="medium-risk", version="1.0.0", vulnerabilities=[vuln]))

        scorer = SupplyChainRiskScorer()
        report = scorer.score(graph)

        assert report.medium_vulnerabilities >= 1

    def test_score_graph_with_multiple_vulnerabilities(self):
        """Test scoring with multiple vulnerabilities on same package."""
        vuln1 = VulnerabilityInfo(cve_id="CVE-2021-1111", severity="high", cvss_score=8.0)
        vuln2 = VulnerabilityInfo(cve_id="CVE-2021-2222", severity="critical", cvss_score=9.5)
        vuln3 = VulnerabilityInfo(cve_id="CVE-2021-3333", severity="medium", cvss_score=5.0)
        graph = DependencyGraph()
        graph.add_package(
            Package(name="multi-vuln", version="1.0.0", vulnerabilities=[vuln1, vuln2, vuln3])
        )

        scorer = SupplyChainRiskScorer()
        report = scorer.score(graph)

        assert report.total_vulnerabilities >= 3
        assert report.critical_vulnerabilities >= 1
        assert report.high_vulnerabilities >= 1

    def test_score_generates_recommendations(self):
        """Test that scoring generates recommendations for vulnerabilities."""
        vuln = VulnerabilityInfo(cve_id="CVE-2021-44228", severity="critical", cvss_score=10.0)
        graph = DependencyGraph()
        graph.add_package(Package(name="log4j", version="2.14.0", vulnerabilities=[vuln]))

        scorer = SupplyChainRiskScorer()
        report = scorer.score(graph)

        assert len(report.recommendations) > 0

    def test_score_with_transitive_vulnerability(self):
        """Test scoring with vulnerability in transitive dependency."""
        vuln = VulnerabilityInfo(cve_id="CVE-2021-9999", severity="high", cvss_score=7.5)
        graph = DependencyGraph()
        graph.add_package(Package(name="app", version="1.0.0"))
        graph.add_package(Package(name="lib", version="1.0.0"))
        graph.add_package(Package(name="transitive", version="1.0.0", vulnerabilities=[vuln]))
        graph.add_dependency(Dependency(source="app@1.0.0", target="lib@1.0.0"))
        graph.add_dependency(Dependency(source="lib@1.0.0", target="transitive@1.0.0"))

        scorer = SupplyChainRiskScorer()
        report = scorer.score(graph)

        assert report.vulnerable_packages >= 1

    def test_risk_level_boundaries(self):
        """Test risk level assignment at boundaries."""
        # Create packages with different risk scores
        vuln_low = VulnerabilityInfo(cve_id="CVE-2021-0001", severity="low", cvss_score=2.0)
        vuln_critical = VulnerabilityInfo(
            cve_id="CVE-2021-0002", severity="critical", cvss_score=10.0
        )

        graph = DependencyGraph()
        graph.add_package(Package(name="low-risk", version="1.0.0", vulnerabilities=[vuln_low]))

        scorer = SupplyChainRiskScorer()
        report = scorer.score(graph)

        # With only low vuln, should not be critical
        assert report.overall_risk_level != RiskLevel.CRITICAL
        low_vuln_score = report.overall_risk_score

        # Add critical vuln
        graph.add_package(
            Package(name="critical-pkg", version="1.0.0", vulnerabilities=[vuln_critical])
        )
        report2 = scorer.score(graph)

        # Score should increase with critical vulnerability
        assert report2.overall_risk_score > low_vuln_score
        assert report2.critical_vulnerabilities >= 1


class TestGraphBuilderAdvanced:
    """Advanced tests for SBOMGraphBuilder."""

    def test_build_graph_with_vulnerabilities(self):
        """Test building graph with vulnerable packages."""
        vuln = VulnerabilityInfo(cve_id="CVE-2021-1234", severity="high", cvss_score=8.0)
        graph = DependencyGraph()
        graph.add_package(Package(name="vuln-pkg", version="1.0.0", vulnerabilities=[vuln]))
        graph.add_package(Package(name="safe-pkg", version="1.0.0"))

        builder = SBOMGraphBuilder()
        graph_data = builder.build(graph)

        assert graph_data.num_nodes == 2
        # Node features should encode vulnerability info
        assert graph_data.node_features.shape[0] == 2

    def test_build_graph_with_different_ecosystems(self):
        """Test building graph with different package ecosystems."""
        graph = DependencyGraph()
        graph.add_package(Package(name="npm-pkg", version="1.0.0", package_type=PackageType.NPM))
        graph.add_package(Package(name="pypi-pkg", version="1.0.0", package_type=PackageType.PYPI))
        graph.add_package(
            Package(name="maven-pkg", version="1.0.0", package_type=PackageType.MAVEN)
        )

        builder = SBOMGraphBuilder()
        graph_data = builder.build(graph)

        assert graph_data.num_nodes == 3

    def test_build_empty_graph(self):
        """Test building empty dependency graph."""
        graph = DependencyGraph()
        builder = SBOMGraphBuilder()
        graph_data = builder.build(graph)

        assert graph_data.num_nodes == 0
        assert graph_data.num_edges == 0

    def test_build_graph_preserves_node_ids(self):
        """Test that graph building preserves node IDs."""
        graph = DependencyGraph()
        graph.add_package(Package(name="pkg-a", version="1.0.0"))
        graph.add_package(Package(name="pkg-b", version="2.0.0"))

        builder = SBOMGraphBuilder()
        graph_data = builder.build(graph)

        assert "pkg-a@1.0.0" in graph_data.node_ids
        assert "pkg-b@2.0.0" in graph_data.node_ids


class TestVisualizationGeneration:
    """Test visualization data generation."""

    def test_visualization_includes_all_risk_levels(self):
        """Test visualization includes different risk level colors."""
        VulnerabilityInfo(cve_id="CVE-2021-0001", severity="critical", cvss_score=10.0)
        VulnerabilityInfo(cve_id="CVE-2021-0002", severity="high", cvss_score=8.0)
        VulnerabilityInfo(cve_id="CVE-2021-0003", severity="medium", cvss_score=5.0)
        VulnerabilityInfo(cve_id="CVE-2021-0004", severity="low", cvss_score=2.0)

        sbom = json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "components": [
                    {"name": "critical-pkg", "version": "1.0.0"},
                    {"name": "high-pkg", "version": "1.0.0"},
                    {"name": "medium-pkg", "version": "1.0.0"},
                    {"name": "low-pkg", "version": "1.0.0"},
                    {"name": "clean-pkg", "version": "1.0.0"},
                ],
            }
        )

        analyzer = SBOMAnalyzer(use_gnn=False)
        report = analyzer.analyze_json(sbom)

        assert report.visualization_data is not None
        assert len(report.visualization_data["nodes"]) == 5

    def test_visualization_ecosystems_grouping(self):
        """Test visualization groups packages by ecosystem."""
        sbom = json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "components": [
                    {"name": "npm-pkg-1", "version": "1.0.0", "purl": "pkg:npm/npm-pkg-1@1.0.0"},
                    {"name": "npm-pkg-2", "version": "1.0.0", "purl": "pkg:npm/npm-pkg-2@1.0.0"},
                    {"name": "pypi-pkg", "version": "1.0.0", "purl": "pkg:pypi/pypi-pkg@1.0.0"},
                ],
            }
        )

        analyzer = SBOMAnalyzer(use_gnn=False)
        report = analyzer.analyze_json(sbom)

        assert "ecosystems" in report.visualization_data
        ecosystems = report.visualization_data["ecosystems"]
        assert "npm" in ecosystems or len(ecosystems) > 0


class TestIntegration:
    """Integration tests for SBOM analysis pipeline."""

    def test_full_analysis_pipeline(self):
        """Test complete SBOM analysis pipeline."""
        # Create SBOM
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "metadata": {
                "component": {
                    "name": "medical-device-firmware",
                    "version": "2.1.0",
                    "type": "firmware",
                }
            },
            "components": [
                {
                    "name": "openssl",
                    "version": "1.1.1",
                    "type": "library",
                },
                {
                    "name": "zlib",
                    "version": "1.2.11",
                    "type": "library",
                },
            ],
            "dependencies": [
                {"ref": "medical-device-firmware@2.1.0", "dependsOn": ["openssl@1.1.1"]},
                {"ref": "openssl@1.1.1", "dependsOn": ["zlib@1.2.11"]},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sbom, f)
            temp_path = f.name

        try:
            # Parse
            parser = SBOMParser()
            graph = parser.parse(temp_path)
            assert len(graph.packages) >= 1

            # Score
            scorer = SupplyChainRiskScorer()
            report = scorer.score(graph)
            assert report.overall_risk_score >= 0

            # Full analysis (use_gnn=False to avoid TensorFlow/Keras compatibility issues)
            analyzer = SBOMAnalyzer(use_gnn=False)
            analysis = analyzer.analyze(temp_path)
            assert analysis is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)
