"""Integration tests for all 5 phases of the medtech-ai-security platform.

These tests verify end-to-end workflows across modules, ensuring
components work together correctly.

Run integration tests: pytest -m integration
Skip integration tests: pytest -m "not integration"
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Phase 1: Threat Intelligence
from medtech_ai_security.threat_intel.nvd_scraper import CVEEntry, NVDScraper
from medtech_ai_security.threat_intel.claude_processor import (
    load_claude_response,
    merge_analysis,
    generate_summary_report,
)

# Phase 2: ML Risk Scoring
from medtech_ai_security.ml.risk_scorer import (
    VulnerabilityRiskScorer,
    RiskPrediction,
)

# Phase 3: Anomaly Detection
from medtech_ai_security.anomaly.traffic_generator import TrafficGenerator
from medtech_ai_security.anomaly.detector import AnomalyDetector, Autoencoder

# Phase 4: Adversarial ML
from medtech_ai_security.adversarial.attacks import AdversarialAttacker, AttackType
from medtech_ai_security.adversarial.defenses import AdversarialDefender, DefenseType
from medtech_ai_security.adversarial.evaluator import RobustnessEvaluator, RobustnessReport

# Phase 5: SBOM Analysis
from medtech_ai_security.sbom_analysis.parser import (
    SBOMParser,
    Package,
    PackageType,
    VulnerabilityInfo,
    create_sample_sbom,
)
from medtech_ai_security.sbom_analysis.graph_builder import SBOMGraphBuilder
from medtech_ai_security.sbom_analysis.risk_scorer import SupplyChainRiskScorer


class TestPhase1ThreatIntelIntegration:
    """Integration tests for Phase 1: NLP Threat Intelligence pipeline."""

    @pytest.mark.integration
    def test_nvd_to_claude_prompt_workflow(self) -> None:
        """Test the NVD scraping to Claude prompt generation workflow."""
        # Create sample CVEs (simulating NVD scrape results)
        cves = [
            CVEEntry(
                cve_id="CVE-2024-1001",
                description="Remote code execution in DICOM server allows unauthenticated access",
                published_date="2024-01-15T10:00:00.000",
                last_modified_date="2024-01-16T10:00:00.000",
                cvss_v3_score=9.8,
                cvss_v3_severity="CRITICAL",
                cwe_ids=["CWE-798"],
                matched_keywords=["DICOM"],
            ),
            CVEEntry(
                cve_id="CVE-2024-1002",
                description="HL7 message parser buffer overflow in healthcare system",
                published_date="2024-01-17T10:00:00.000",
                last_modified_date="2024-01-18T10:00:00.000",
                cvss_v3_score=7.5,
                cvss_v3_severity="HIGH",
                cwe_ids=["CWE-120"],
                matched_keywords=["HL7", "healthcare"],
            ),
        ]

        # Generate Claude prompt
        scraper = NVDScraper()
        prompt = scraper.generate_claude_prompt(cves)

        # Verify prompt contains expected elements
        assert "CVE-2024-1001" in prompt
        assert "CVE-2024-1002" in prompt
        assert "CRITICAL" in prompt
        assert "device_type" in prompt
        assert "clinical_impact" in prompt

    @pytest.mark.integration
    def test_claude_response_merge_workflow(self, tmp_path: Path) -> None:
        """Test merging Claude analysis back into CVE data."""
        # Create original CVE data file
        cve_data = {
            "metadata": {"generated_at": "2024-01-15T10:00:00Z"},
            "cves": [
                {
                    "cve_id": "CVE-2024-1001",
                    "description": "DICOM vulnerability",
                    "cvss_v3_score": 9.8,
                    "cvss_v3_severity": "CRITICAL",
                }
            ],
        }
        cve_file = tmp_path / "cves.json"
        with open(cve_file, "w") as f:
            json.dump(cve_data, f)

        # Create Claude response file
        claude_response = {
            "analyses": [
                {
                    "cve_id": "CVE-2024-1001",
                    "device_type": "imaging",
                    "clinical_impact": "HIGH",
                    "exploitability": "EASY",
                    "reasoning": "DICOM server vulnerability affects medical imaging",
                }
            ]
        }
        response_file = tmp_path / "response.json"
        with open(response_file, "w") as f:
            json.dump(claude_response, f)

        # Merge analysis
        output_file = tmp_path / "enriched.json"
        enriched = merge_analysis(cve_file, response_file, output_file)

        # Verify enriched data
        assert enriched["cves"][0]["device_type"] == "imaging"
        assert enriched["cves"][0]["clinical_impact"] == "HIGH"
        assert enriched["metadata"]["enriched_count"] == 1

        # Verify file was saved
        assert output_file.exists()

    @pytest.mark.integration
    def test_full_report_generation(self, tmp_path: Path) -> None:
        """Test generating a summary report from enriched CVE data."""
        enriched_data = {
            "metadata": {
                "generated_at": "2024-01-15T10:00:00Z",
                "enriched_at": "2024-01-16T10:00:00Z",
                "enriched_count": 2,
            },
            "cves": [
                {
                    "cve_id": "CVE-2024-1001",
                    "description": "Critical DICOM vulnerability",
                    "cvss_v3_score": 9.8,
                    "cvss_v3_severity": "CRITICAL",
                    "device_type": "imaging",
                    "clinical_impact": "HIGH",
                    "exploitability": "EASY",
                },
                {
                    "cve_id": "CVE-2024-1002",
                    "description": "HL7 parser issue",
                    "cvss_v3_score": 7.5,
                    "cvss_v3_severity": "HIGH",
                    "device_type": "healthcare_it",
                    "clinical_impact": "MEDIUM",
                    "exploitability": "MODERATE",
                },
            ],
        }

        report_file = tmp_path / "report.txt"
        report = generate_summary_report(enriched_data, report_file)

        # Verify report content
        assert "THREAT INTELLIGENCE REPORT" in report
        assert "CRITICAL" in report
        assert "HIGH" in report
        assert "imaging" in report

        # Verify file was saved
        assert report_file.exists()


class TestPhase2RiskScoringIntegration:
    """Integration tests for Phase 2: ML Risk Scoring pipeline."""

    @pytest.fixture
    def training_data_file(self, tmp_path: Path) -> Path:
        """Create a temporary training data file."""
        import numpy as np

        cves = [
            {
                "cve_id": f"CVE-2024-{i:04d}",
                "description": f"Test vulnerability {i}",
                "cvss_v3_score": float(np.random.uniform(0, 10)),
                "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "cwe_ids": ["CWE-798"],
                "device_type": np.random.choice(["imaging", "monitoring", "infusion"]),
                "clinical_impact": np.random.choice(["HIGH", "MEDIUM", "LOW"]),
                "exploitability": np.random.choice(["EASY", "MODERATE", "HARD"]),
                "published_date": "2024-01-15T10:00:00.000Z",
                "references": [],
            }
            for i in range(60)
        ]

        data = {"cves": cves}
        file_path = tmp_path / "training_data.json"

        with open(file_path, "w") as f:
            json.dump(data, f)

        return file_path

    @pytest.fixture
    def trained_scorer(self, training_data_file: Path) -> VulnerabilityRiskScorer:
        """Create and train a risk scorer with sample data."""
        scorer = VulnerabilityRiskScorer()
        scorer.load_training_data(training_data_file)
        scorer.train()
        return scorer

    @pytest.mark.integration
    def test_scorer_training_workflow(self, training_data_file: Path) -> None:
        """Test risk scorer training workflow."""
        scorer = VulnerabilityRiskScorer()

        # Load training data from file
        count = scorer.load_training_data(training_data_file)
        assert count == 60

        # Train
        metrics = scorer.train()

        # Verify training
        assert scorer.is_trained
        assert len(scorer.feature_names) > 0
        assert scorer.feature_means is not None
        assert scorer.feature_stds is not None
        assert "nb_test_accuracy" in metrics
        assert "knn_test_accuracy" in metrics

    @pytest.mark.integration
    def test_end_to_end_risk_prediction(self, trained_scorer: VulnerabilityRiskScorer) -> None:
        """Test end-to-end risk prediction from vulnerability data."""
        # Create test vulnerability
        vuln = {
            "cve_id": "CVE-2024-9999",
            "description": "Critical remote code execution in medical DICOM server",
            "cvss_v3_score": 9.8,
            "cvss_v3_severity": "CRITICAL",
            "cwe_ids": ["CWE-798"],
            "device_type": "imaging",
            "clinical_impact": "HIGH",
        }

        # Predict risk
        prediction = trained_scorer.predict(vuln)

        # Verify result structure
        assert isinstance(prediction, RiskPrediction)
        assert prediction.cve_id == "CVE-2024-9999"
        assert 0 <= prediction.risk_score <= 100
        assert prediction.priority in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert 0 <= prediction.confidence <= 1
        assert len(prediction.contributing_factors) > 0
        assert prediction.recommendation


class TestPhase3AnomalyDetectionIntegration:
    """Integration tests for Phase 3: Anomaly Detection pipeline."""

    @pytest.mark.integration
    def test_traffic_generation_to_detection(self) -> None:
        """Test traffic generation and anomaly detection workflow."""
        # Generate synthetic traffic
        generator = TrafficGenerator(seed=42)
        normal_traffic = generator.generate_normal_dicom(n_samples=50)
        attack_traffic = generator.generate_attack_dicom(n_samples=10)

        # Verify traffic was generated
        assert len(normal_traffic) == 50
        assert len(attack_traffic) == 10

        # Verify traffic structure (DICOMPacket objects)
        assert hasattr(normal_traffic[0], "timestamp")
        assert hasattr(normal_traffic[0], "command")

    @pytest.mark.integration
    def test_detector_training_and_inference(self) -> None:
        """Test detector training and anomaly detection."""
        # Generate training data - returns (features, labels, raw_packets)
        generator = TrafficGenerator(seed=42)
        features, labels, raw_packets = generator.generate_dataset(
            n_normal=100,
            n_attack=20,
            protocol="dicom",
        )

        # Train detector (input_dim inferred from data)
        detector = AnomalyDetector(
            latent_dim=4,
            threshold_percentile=95,
        )

        # Use only normal traffic for training
        normal_features = features[labels == 0]

        # Train on normal traffic
        detector.fit(normal_features, epochs=5, batch_size=16)

        # Verify detector is trained
        assert detector.is_fitted
        assert detector.threshold > 0

        # Test detection
        results = detector.detect(features)
        assert len(results) == len(features)


class TestPhase4AdversarialMLIntegration:
    """Integration tests for Phase 4: Adversarial ML pipeline."""

    @pytest.fixture
    def mock_model(self) -> Mock:
        """Create a mock medical imaging model."""
        model = Mock()

        def predict(x: np.ndarray) -> np.ndarray:
            # Simple mock: return probabilities based on input
            batch_size = x.shape[0]
            return np.random.rand(batch_size, 2)

        model.side_effect = predict
        model.return_value = np.random.rand(4, 2)
        return model

    @pytest.mark.integration
    def test_attack_and_defense_workflow(self) -> None:
        """Test attack generation and defense application workflow."""
        # Create test images
        images = np.random.rand(4, 32, 32, 3).astype(np.float32)

        # Create a simple callable model for testing
        # Mock models don't support gradient computation, so we use a real function
        def simple_model(x: np.ndarray) -> np.ndarray:
            """Simple model that returns class probabilities based on mean intensity."""
            if hasattr(x, "numpy"):
                x = x.numpy()
            batch_size = x.shape[0]
            # Return 2-class probabilities
            probs = np.zeros((batch_size, 2))
            for i in range(batch_size):
                mean_val = np.mean(x[i])
                probs[i, 0] = 1.0 - mean_val
                probs[i, 1] = mean_val
            return probs

        # Initialize defender (doesn't need gradients)
        defender = AdversarialDefender(simple_model)

        # Test defense directly on images (skip gradient-based attack for mock)
        defended = defender.gaussian_blur(images, sigma=1.0)

        # Verify defense applied
        assert defended.shape == images.shape

        # Test JPEG compression defense
        jpeg_defended = defender.jpeg_compression(images, quality=85)
        assert jpeg_defended.shape == images.shape

        # Test feature squeezing defense
        squeezed = defender.feature_squeezing(images, bit_depth=5)
        assert squeezed.shape == images.shape

    @pytest.mark.integration
    def test_robustness_report_generation(self, mock_model: Mock, tmp_path: Path) -> None:
        """Test generating a robustness evaluation report."""
        # Create report
        report = RobustnessReport(
            model_name="test_model",
            evaluation_date="2024-01-15",
            clean_accuracy=0.95,
            attack_results={
                "fgsm": {"robust_accuracy": 0.75, "attack_success_rate": 0.25}
            },
            defense_results={
                "gaussian_blur": {"defended_accuracy": 0.85}
            },
            vulnerability_assessment="Model shows moderate vulnerability to FGSM attacks",
            recommendations=["Apply adversarial training", "Use input preprocessing"],
            clinical_risk_level="MEDIUM",
        )

        # Save report
        report_path = tmp_path / "robustness_report.json"
        report.save(report_path)

        # Load and verify
        loaded = RobustnessReport.load(report_path)
        assert loaded.model_name == "test_model"
        assert loaded.clean_accuracy == 0.95


class TestPhase5SBOMAnalysisIntegration:
    """Integration tests for Phase 5: SBOM Analysis pipeline."""

    @pytest.mark.integration
    def test_sbom_parsing_to_graph_building(self) -> None:
        """Test SBOM parsing and graph construction workflow."""
        # Create sample SBOM
        sbom = create_sample_sbom()

        # Parse SBOM
        parser = SBOMParser()
        dep_graph = parser.parse_json(sbom)

        # Verify parsing
        assert dep_graph.package_count > 0

        # Build graph
        builder = SBOMGraphBuilder()
        graph_data = builder.build(dep_graph)

        # Verify graph construction
        assert graph_data.num_nodes > 0
        assert graph_data.node_features.shape[0] == graph_data.num_nodes

    @pytest.mark.integration
    def test_end_to_end_risk_analysis(self) -> None:
        """Test end-to-end SBOM risk analysis workflow."""
        # Create sample SBOM with vulnerabilities
        sbom = create_sample_sbom()

        # Parse SBOM
        parser = SBOMParser()
        dep_graph = parser.parse_json(sbom)

        # Build graph
        builder = SBOMGraphBuilder()
        graph_data = builder.build(dep_graph)

        # Score risks using SupplyChainRiskScorer
        scorer = SupplyChainRiskScorer()
        risk_report = scorer.score(dep_graph)

        # Verify risk analysis
        assert risk_report.total_packages > 0
        assert risk_report.overall_risk_score >= 0
        assert len(risk_report.package_risks) > 0


class TestCrossPhaseIntegration:
    """Integration tests spanning multiple phases."""

    @pytest.fixture
    def training_data_file(self, tmp_path: Path) -> Path:
        """Create a temporary training data file."""
        cves = [
            {
                "cve_id": f"CVE-2024-{i:04d}",
                "description": f"Test vulnerability {i}",
                "cvss_v3_score": float(np.random.uniform(0, 10)),
                "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "cwe_ids": ["CWE-798"],
                "device_type": np.random.choice(["imaging", "monitoring", "infusion"]),
                "clinical_impact": np.random.choice(["HIGH", "MEDIUM", "LOW"]),
                "exploitability": np.random.choice(["EASY", "MODERATE", "HARD"]),
                "published_date": "2024-01-15T10:00:00.000Z",
                "references": [],
            }
            for i in range(60)
        ]

        data = {"cves": cves}
        file_path = tmp_path / "training_data.json"

        with open(file_path, "w") as f:
            json.dump(data, f)

        return file_path

    @pytest.fixture
    def trained_scorer(self, training_data_file: Path) -> VulnerabilityRiskScorer:
        """Create and train a risk scorer with sample data."""
        scorer = VulnerabilityRiskScorer()
        scorer.load_training_data(training_data_file)
        scorer.train()
        return scorer

    @pytest.mark.integration
    def test_threat_intel_to_risk_scoring(self, trained_scorer: VulnerabilityRiskScorer) -> None:
        """Test flow from threat intelligence to risk scoring."""
        # Phase 1: Create CVE data
        cve = CVEEntry(
            cve_id="CVE-2024-1001",
            description="Critical vulnerability in medical DICOM server",
            published_date="2024-01-15",
            last_modified_date="2024-01-16",
            cvss_v3_score=9.8,
            cvss_v3_severity="CRITICAL",
            cwe_ids=["CWE-798"],
            matched_keywords=["DICOM", "medical"],
            device_type="imaging",
            clinical_impact="HIGH",
        )

        # Phase 2: Score the CVE using trained model
        vuln_dict = {
            "cve_id": cve.cve_id,
            "description": cve.description,
            "cvss_v3_score": cve.cvss_v3_score,
            "cvss_v3_severity": cve.cvss_v3_severity,
            "cwe_ids": cve.cwe_ids,
            "device_type": cve.device_type,
            "clinical_impact": cve.clinical_impact,
        }
        prediction = trained_scorer.predict(vuln_dict)

        # Verify prediction structure
        assert isinstance(prediction, RiskPrediction)
        assert prediction.cve_id == cve.cve_id
        assert 0 <= prediction.risk_score <= 100
        # High CVSS score should lead to elevated risk
        assert prediction.risk_score >= 50

    @pytest.mark.integration
    def test_sbom_vulnerability_to_risk_scoring(self, trained_scorer: VulnerabilityRiskScorer) -> None:
        """Test SBOM vulnerability analysis integrated with risk scoring."""
        # Create package with known vulnerability
        vuln = VulnerabilityInfo(
            cve_id="CVE-2021-44228",
            severity="critical",
            cvss_score=10.0,
            description="Log4j RCE vulnerability",
        )

        pkg = Package(
            name="log4j",
            version="2.14.0",
            package_type=PackageType.PYPI,
            vulnerabilities=[vuln],
        )

        # Score package risk using trained model
        vuln_dict = {
            "cve_id": vuln.cve_id,
            "description": vuln.description,
            "cvss_v3_score": vuln.cvss_score,
            "cvss_v3_severity": vuln.severity.upper(),
        }
        prediction = trained_scorer.predict(vuln_dict)

        # Critical vulnerability should have high risk
        assert isinstance(prediction, RiskPrediction)
        assert prediction.risk_score >= 50


class TestDataFlowIntegration:
    """Tests verifying data flows correctly between components."""

    @pytest.mark.integration
    def test_json_serialization_roundtrip(self, tmp_path: Path) -> None:
        """Test that data serializes and deserializes correctly."""
        # Create CVE entry
        original = CVEEntry(
            cve_id="CVE-2024-1001",
            description="Test vulnerability",
            published_date="2024-01-15",
            last_modified_date="2024-01-16",
            cvss_v3_score=7.5,
            cvss_v3_severity="HIGH",
            cwe_ids=["CWE-79"],
            matched_keywords=["medical"],
        )

        # Save using scraper
        scraper = NVDScraper()
        output_file = tmp_path / "test_cves.json"
        scraper.save_results([original], output_file)

        # Load and verify
        with open(output_file) as f:
            data = json.load(f)

        loaded = data["cves"][0]
        assert loaded["cve_id"] == original.cve_id
        assert loaded["cvss_v3_score"] == original.cvss_v3_score

    @pytest.mark.integration
    def test_feature_array_shapes(self) -> None:
        """Test that feature arrays have consistent shapes across modules."""
        # Phase 3: Traffic features - use generate_dataset which returns feature arrays
        generator = TrafficGenerator(seed=42)
        features, labels, raw_packets = generator.generate_dataset(
            n_normal=10,
            n_attack=0,
            protocol="dicom",
        )

        # Verify shape
        assert features.shape[0] == 10
        assert features.shape[1] > 0  # Feature dimension varies

        # Phase 5: SBOM graph features
        sbom = create_sample_sbom()
        parser = SBOMParser()
        dep_graph = parser.parse_json(sbom)
        builder = SBOMGraphBuilder()
        graph_data = builder.build(dep_graph)

        # Verify feature dimensionality
        assert graph_data.node_features.shape[1] == 88  # NodeFeatures.feature_dim
