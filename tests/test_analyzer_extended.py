"""Extended tests for SBOM Analyzer - covering GNN, CLI, and visualization."""

import json
from pathlib import Path

import numpy as np
import pytest

from medtech_ai_security.sbom_analysis.analyzer import (
    AnalysisReport,
    SBOMAnalyzer,
)
from medtech_ai_security.sbom_analysis.parser import (
    DependencyGraph,
    Package,
)
from medtech_ai_security.sbom_analysis.risk_scorer import (
    RiskLevel,
    RiskReport,
)


class TestAnalysisReportWithGNN:
    """Test AnalysisReport with GNN predictions."""

    def test_analysis_report_to_dict_with_gnn_predictions(self):
        """Test to_dict includes GNN predictions when available."""
        report = AnalysisReport(
            sbom_file="test.json",
            sbom_format="CycloneDX",
            gnn_predictions={
                "predictions": {"pkg@1.0.0": {"label": 0, "label_name": "clean"}},
                "summary": {"predicted_vulnerable": 0, "predicted_clean": 1},
            },
        )

        result = report.to_dict()

        assert "gnn_predictions" in result
        assert result["gnn_predictions"]["summary"]["predicted_clean"] == 1


class TestGNNPredictions:
    """Test GNN prediction functionality in analyzer."""

    @pytest.fixture
    def mock_gnn_model(self):
        """Create a mock GNN model."""
        from unittest.mock import MagicMock

        mock = MagicMock()
        # Return probabilities for 3 nodes, 3 classes each
        mock.predict_proba.return_value = np.array(
            [
                [0.8, 0.1, 0.1],  # clean
                [0.1, 0.8, 0.1],  # vulnerable
                [0.1, 0.1, 0.8],  # transitive
            ]
        )
        return mock

    def test_run_gnn_predictions_success(self, mock_gnn_model):
        """Test successful GNN predictions."""
        from medtech_ai_security.sbom_analysis.graph_builder import GraphData

        analyzer = SBOMAnalyzer(use_gnn=False)
        analyzer.gnn_model = mock_gnn_model

        # Create mock graph data
        graph_data = GraphData(
            node_features=np.random.rand(3, 88).astype(np.float32),
            edge_index=np.array([[0, 1], [1, 2]]),
            node_labels=np.array([0, 1, 2]),
            num_nodes=3,
            node_ids=["pkg-a@1.0.0", "pkg-b@1.0.0", "pkg-c@1.0.0"],
        )

        dep_graph = DependencyGraph()
        dep_graph.add_package(Package(name="pkg-a", version="1.0.0"))
        dep_graph.add_package(Package(name="pkg-b", version="1.0.0"))
        dep_graph.add_package(Package(name="pkg-c", version="1.0.0"))

        result = analyzer._run_gnn_predictions(graph_data, dep_graph)

        assert "predictions" in result
        assert "summary" in result
        assert result["summary"]["predicted_vulnerable"] == 1
        assert result["summary"]["predicted_transitive"] == 1
        assert result["summary"]["predicted_clean"] == 1
        assert "pkg-a@1.0.0" in result["predictions"]
        assert result["predictions"]["pkg-a@1.0.0"]["label_name"] == "clean"

    def test_run_gnn_predictions_error(self, mock_gnn_model):
        """Test GNN predictions handles errors gracefully."""
        from medtech_ai_security.sbom_analysis.graph_builder import GraphData

        analyzer = SBOMAnalyzer(use_gnn=False)
        mock_gnn_model.predict_proba.side_effect = ValueError("Test error")
        analyzer.gnn_model = mock_gnn_model

        graph_data = GraphData(
            node_features=np.random.rand(3, 88).astype(np.float32),
            edge_index=np.array([[0, 1], [1, 2]]),
            node_labels=np.array([0, 1, 2]),
            num_nodes=3,
        )

        dep_graph = DependencyGraph()

        result = analyzer._run_gnn_predictions(graph_data, dep_graph)

        assert "error" in result
        assert "Test error" in result["error"]

    def test_analyze_with_gnn_model(self, mock_gnn_model, tmp_path):
        """Test analyze includes GNN predictions when model available."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {"name": "pkg-a", "version": "1.0.0"},
                {"name": "pkg-b", "version": "1.0.0"},
                {"name": "pkg-c", "version": "1.0.0"},
            ],
        }
        sbom_file = tmp_path / "test.json"
        sbom_file.write_text(json.dumps(sbom))

        analyzer = SBOMAnalyzer(use_gnn=False)
        analyzer.gnn_model = mock_gnn_model

        report = analyzer.analyze(str(sbom_file))

        assert report.gnn_predictions is not None
        assert "predictions" in report.gnn_predictions

    def test_analyze_json_with_gnn_model(self, mock_gnn_model):
        """Test analyze_json includes GNN predictions."""
        sbom = json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "components": [
                    {"name": "pkg-a", "version": "1.0.0"},
                    {"name": "pkg-b", "version": "1.0.0"},
                    {"name": "pkg-c", "version": "1.0.0"},
                ],
            }
        )

        analyzer = SBOMAnalyzer(use_gnn=False)
        analyzer.gnn_model = mock_gnn_model

        report = analyzer.analyze_json(sbom)

        assert report.gnn_predictions is not None


class TestVisualizationColorCodes:
    """Test visualization color code assignments."""

    @pytest.fixture
    def create_analyzer_with_risk_report(self):
        """Create analyzer and risk report for testing visualization."""

        def _create(risk_level):
            from medtech_ai_security.sbom_analysis.risk_scorer import PackageRisk

            analyzer = SBOMAnalyzer(use_gnn=False)
            dep_graph = DependencyGraph()
            dep_graph.add_package(Package(name="test-pkg", version="1.0.0"))

            pkg_risk = PackageRisk(
                package_id="test-pkg@1.0.0",
                package_name="test-pkg",
                package_version="1.0.0",
                risk_level=risk_level,
                risk_score=50.0,
            )

            risk_report = RiskReport(
                overall_risk_level=risk_level,
                overall_risk_score=50.0,
                package_risks=[pkg_risk],
            )

            return analyzer, dep_graph, risk_report

        return _create

    def test_visualization_critical_color(self, create_analyzer_with_risk_report):
        """Test critical risk level gets red color."""
        analyzer, dep_graph, risk_report = create_analyzer_with_risk_report(RiskLevel.CRITICAL)
        viz_data = analyzer._generate_visualization(dep_graph, risk_report)

        node = viz_data["nodes"][0]
        assert node["color"] == "#dc3545"  # Red

    def test_visualization_high_color(self, create_analyzer_with_risk_report):
        """Test high risk level gets orange color."""
        analyzer, dep_graph, risk_report = create_analyzer_with_risk_report(RiskLevel.HIGH)
        viz_data = analyzer._generate_visualization(dep_graph, risk_report)

        node = viz_data["nodes"][0]
        assert node["color"] == "#fd7e14"  # Orange

    def test_visualization_medium_color(self, create_analyzer_with_risk_report):
        """Test medium risk level gets yellow color."""
        analyzer, dep_graph, risk_report = create_analyzer_with_risk_report(RiskLevel.MEDIUM)
        viz_data = analyzer._generate_visualization(dep_graph, risk_report)

        node = viz_data["nodes"][0]
        assert node["color"] == "#ffc107"  # Yellow

    def test_visualization_low_color(self, create_analyzer_with_risk_report):
        """Test low risk level gets green color."""
        analyzer, dep_graph, risk_report = create_analyzer_with_risk_report(RiskLevel.LOW)
        viz_data = analyzer._generate_visualization(dep_graph, risk_report)

        node = viz_data["nodes"][0]
        assert node["color"] == "#28a745"  # Green

    def test_visualization_info_color(self, create_analyzer_with_risk_report):
        """Test info risk level gets gray color."""
        analyzer, dep_graph, risk_report = create_analyzer_with_risk_report(RiskLevel.INFO)
        viz_data = analyzer._generate_visualization(dep_graph, risk_report)

        node = viz_data["nodes"][0]
        assert node["color"] == "#6c757d"  # Gray

    def test_visualization_package_without_risk(self):
        """Test package without risk entry gets gray color."""
        analyzer = SBOMAnalyzer(use_gnn=False)
        dep_graph = DependencyGraph()
        dep_graph.add_package(Package(name="test-pkg", version="1.0.0"))

        # Empty risk report with no package risks
        risk_report = RiskReport(
            overall_risk_level=RiskLevel.INFO,
            overall_risk_score=0.0,
            package_risks=[],
        )

        viz_data = analyzer._generate_visualization(dep_graph, risk_report)

        node = viz_data["nodes"][0]
        assert node["color"] == "#6c757d"  # Gray (no risk data)


class TestSBOMAnalyzerCLI:
    """Test SBOM Analyzer CLI functionality."""

    @pytest.fixture
    def sample_sbom_file(self, tmp_path):
        """Create a sample SBOM file for CLI testing."""
        sbom = {
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
            ],
            "dependencies": [],
        }
        sbom_file = tmp_path / "test-sbom.json"
        sbom_file.write_text(json.dumps(sbom))
        return str(sbom_file)

    def test_cli_analyze_command(self, sample_sbom_file, tmp_path, monkeypatch):
        """Test CLI analyze command."""
        from medtech_ai_security.sbom_analysis import analyzer

        output_file = tmp_path / "output.json"
        monkeypatch.setattr(
            "sys.argv",
            ["medsec-sbom", "analyze", sample_sbom_file, "-o", str(output_file), "--no-gnn"],
        )

        # Run main (should not raise)
        try:
            analyzer.main()
        except SystemExit as e:
            # Exit code 0 is success
            assert e.code is None or e.code == 0

        # Check output was created
        assert output_file.exists()

    def test_cli_analyze_with_html(self, sample_sbom_file, tmp_path, monkeypatch):
        """Test CLI analyze command with HTML output."""
        from medtech_ai_security.sbom_analysis import analyzer

        html_file = tmp_path / "report.html"
        monkeypatch.setattr(
            "sys.argv",
            ["medsec-sbom", "analyze", sample_sbom_file, "--html", str(html_file), "--no-gnn"],
        )

        try:
            analyzer.main()
        except SystemExit as e:
            assert e.code is None or e.code == 0

        assert html_file.exists()
        content = html_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_cli_analyze_file_not_found(self, tmp_path, monkeypatch):
        """Test CLI analyze command with non-existent file."""
        from medtech_ai_security.sbom_analysis import analyzer

        monkeypatch.setattr(
            "sys.argv",
            ["medsec-sbom", "analyze", "/nonexistent/path.json", "--no-gnn"],
        )

        with pytest.raises(SystemExit) as exc_info:
            analyzer.main()

        assert exc_info.value.code == 1

    def test_cli_demo_command(self, monkeypatch):
        """Test CLI demo command."""
        from medtech_ai_security.sbom_analysis import analyzer

        monkeypatch.setattr("sys.argv", ["medsec-sbom", "demo"])

        try:
            analyzer.main()
        except SystemExit as e:
            assert e.code is None or e.code == 0

    def test_cli_demo_with_html(self, tmp_path, monkeypatch):
        """Test CLI demo command with HTML output."""
        from medtech_ai_security.sbom_analysis import analyzer

        html_file = tmp_path / "demo-report.html"
        monkeypatch.setattr("sys.argv", ["medsec-sbom", "demo", "--html", str(html_file)])

        try:
            analyzer.main()
        except SystemExit as e:
            assert e.code is None or e.code == 0

        assert html_file.exists()

    def test_cli_parse_command(self, sample_sbom_file, monkeypatch):
        """Test CLI parse command."""
        from medtech_ai_security.sbom_analysis import analyzer

        monkeypatch.setattr("sys.argv", ["medsec-sbom", "parse", sample_sbom_file])

        try:
            analyzer.main()
        except SystemExit as e:
            assert e.code is None or e.code == 0

    def test_cli_parse_file_not_found(self, monkeypatch):
        """Test CLI parse command with non-existent file."""
        from medtech_ai_security.sbom_analysis import analyzer

        monkeypatch.setattr(
            "sys.argv",
            ["medsec-sbom", "parse", "/nonexistent/path.json"],
        )

        with pytest.raises(SystemExit) as exc_info:
            analyzer.main()

        assert exc_info.value.code == 1

    def test_cli_no_command(self, monkeypatch, capsys):
        """Test CLI with no command shows help."""
        from medtech_ai_security.sbom_analysis import analyzer

        monkeypatch.setattr("sys.argv", ["medsec-sbom"])

        try:
            analyzer.main()
        except SystemExit:
            pass

    def test_cli_verbose_flag(self, sample_sbom_file, monkeypatch):
        """Test CLI with verbose flag."""
        from medtech_ai_security.sbom_analysis import analyzer

        monkeypatch.setattr(
            "sys.argv",
            ["medsec-sbom", "analyze", sample_sbom_file, "--no-gnn", "-v"],
        )

        try:
            analyzer.main()
        except SystemExit as e:
            assert e.code is None or e.code == 0

    def test_cli_no_medical_context_flag(self, sample_sbom_file, monkeypatch):
        """Test CLI with --no-medical flag."""
        from medtech_ai_security.sbom_analysis import analyzer

        monkeypatch.setattr(
            "sys.argv",
            ["medsec-sbom", "analyze", sample_sbom_file, "--no-gnn", "--no-medical"],
        )

        try:
            analyzer.main()
        except SystemExit as e:
            assert e.code is None or e.code == 0


class TestSBOMAnalyzerGNNLoading:
    """Test GNN model loading behavior."""

    def test_analyzer_gnn_loading_disabled(self):
        """Test analyzer with GNN disabled does not load model."""
        analyzer = SBOMAnalyzer(use_gnn=False)
        assert analyzer.gnn_model is None

    def test_analyzer_gnn_loading_import_error(self, monkeypatch):
        """Test analyzer handles GNN import errors gracefully."""
        # Create analyzer - it may or may not have TF available
        analyzer = SBOMAnalyzer(use_gnn=True)
        # Either it loaded or didn't - both are valid
        # The key is it doesn't crash
        assert analyzer is not None


class TestCLIParseWithVulnerabilities:
    """Test CLI parse command with vulnerabilities."""

    def test_parse_shows_vulnerabilities(self, tmp_path, monkeypatch, capsys):
        """Test parse command shows vulnerability information."""
        from medtech_ai_security.sbom_analysis import analyzer

        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "components": [
                {"name": "lodash", "version": "4.17.20", "purl": "pkg:npm/lodash@4.17.20"},
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
        sbom_file = tmp_path / "vuln-sbom.json"
        sbom_file.write_text(json.dumps(sbom))

        monkeypatch.setattr("sys.argv", ["medsec-sbom", "parse", str(sbom_file)])

        try:
            analyzer.main()
        except SystemExit as e:
            assert e.code is None or e.code == 0

        captured = capsys.readouterr()
        assert "lodash" in captured.out
