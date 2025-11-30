"""Unit tests for Phase 5: SBOM Supply Chain Analysis."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from medtech_ai_security.sbom_analysis.parser import (
    Package,
    VulnerabilityInfo,
    Dependency,
    DependencyGraph,
    SBOMParser,
    SBOMFormat,
    PackageType,
)
from medtech_ai_security.sbom_analysis.graph_builder import (
    SBOMGraphBuilder,
    NodeFeatures,
    NodeType,
    EdgeType,
)
from medtech_ai_security.sbom_analysis.gnn_model import (
    GNNConfig,
    GraphConvLayer,
    GraphAttentionLayer,
    SimpleVulnerabilityClassifier,
    TF_AVAILABLE,
)
try:
    from medtech_ai_security.sbom_analysis.gnn_model import VulnerabilityGNN
except ImportError:
    VulnerabilityGNN = None
from medtech_ai_security.sbom_analysis.risk_scorer import (
    RiskLevel,
    PackageRisk,
    RiskReport,
    SupplyChainRiskScorer,
)
from medtech_ai_security.sbom_analysis.analyzer import (
    SBOMAnalyzer,
    AnalysisReport,
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(sbom, f)
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = SBOMAnalyzer()
        assert analyzer is not None

    def test_analyze_sbom(self, temp_sbom_file):
        """Test analyzing an SBOM file."""
        analyzer = SBOMAnalyzer()
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


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestVulnerabilityGNN:
    """Test VulnerabilityGNN model (requires TensorFlow)."""

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for GNN testing."""
        from medtech_ai_security.sbom_analysis.graph_builder import GraphData

        # Create simple graph with 10 nodes
        node_features = np.random.rand(10, 88).astype(np.float32)
        edge_index = np.array([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6]
        ])
        node_labels = np.array([0, 0, 1, 1, 0, 1, 0, 2, 2, 0])

        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            node_labels=node_labels,
            num_nodes=10,
        )

    def test_gnn_initialization(self):
        """Test GNN model initializes correctly."""
        if VulnerabilityGNN is None:
            pytest.skip("VulnerabilityGNN not available")

        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        assert model.model is not None
        assert model.config.hidden_dim == 32

    def test_gnn_train(self, sample_graph_data):
        """Test GNN model training."""
        if VulnerabilityGNN is None:
            pytest.skip("VulnerabilityGNN not available")

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
        if VulnerabilityGNN is None:
            pytest.skip("VulnerabilityGNN not available")

        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        # Train briefly
        model.train([sample_graph_data], epochs=3)

        predictions = model.predict(sample_graph_data)

        assert len(predictions) == 10
        assert all(p in [0, 1, 2] for p in predictions)

    def test_gnn_predict_proba(self, sample_graph_data):
        """Test GNN model probability prediction."""
        if VulnerabilityGNN is None:
            pytest.skip("VulnerabilityGNN not available")

        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        model.train([sample_graph_data], epochs=3)
        probs = model.predict_proba(sample_graph_data)

        assert probs.shape == (10, 3)
        # Probabilities should sum to approximately 1
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_gnn_evaluate(self, sample_graph_data):
        """Test GNN model evaluation."""
        if VulnerabilityGNN is None:
            pytest.skip("VulnerabilityGNN not available")

        config = GNNConfig(input_dim=88, hidden_dim=32, num_layers=2)
        model = VulnerabilityGNN(config)

        model.train([sample_graph_data], epochs=3)
        metrics = model.evaluate([sample_graph_data])

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_gnn_save_load(self, sample_graph_data, tmp_path):
        """Test GNN model save and load."""
        if VulnerabilityGNN is None:
            pytest.skip("VulnerabilityGNN not available")

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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
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

            # Full analysis
            analyzer = SBOMAnalyzer()
            analysis = analyzer.analyze(temp_path)
            assert analysis is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)
