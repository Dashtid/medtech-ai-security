"""
Tests for the FastAPI REST API module.

Tests cover:
- Health endpoints (/health, /ready, /metrics)
- SBOM analysis endpoint
- Anomaly detection endpoint
- CVE lookup endpoint
- Adversarial ML testing endpoint
- Error handlers
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from medtech_ai_security.api.main import (
    AdversarialRequest,
    AdversarialResponse,
    AnomalyRequest,
    AnomalyResponse,
    CVERequest,
    CVEResponse,
    HealthResponse,
    SBOMRequest,
    SBOMResponse,
    app,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_creation(self):
        """Test model creation."""
        response = HealthResponse(
            status="healthy",
            version="1.1.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        assert response.status == "healthy"
        assert response.version == "1.1.0"

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            HealthResponse(status="healthy")


class TestSBOMRequest:
    """Test SBOMRequest model."""

    def test_creation_minimal(self):
        """Test minimal request creation."""
        request = SBOMRequest(sbom={"components": []})

        assert request.sbom == {"components": []}
        assert request.format == "cyclonedx"  # Default
        assert request.include_gnn is True  # Default

    def test_creation_with_options(self):
        """Test request with custom options."""
        request = SBOMRequest(
            sbom={"components": [{"name": "test"}]},
            format="spdx",
            include_gnn=False,
        )

        assert request.format == "spdx"
        assert request.include_gnn is False


class TestSBOMResponse:
    """Test SBOMResponse model."""

    def test_creation(self):
        """Test response creation."""
        response = SBOMResponse(
            risk_level="high",
            risk_score=75.0,
            total_packages=10,
            vulnerable_packages=3,
            critical_vulnerabilities=1,
            high_vulnerabilities=2,
            recommendations=["Update package A"],
            fda_compliance_notes=["SBOM is compliant"],
        )

        assert response.risk_level == "high"
        assert response.risk_score == 75.0
        assert response.total_packages == 10


class TestAnomalyRequest:
    """Test AnomalyRequest model."""

    def test_creation(self):
        """Test request creation."""
        request = AnomalyRequest(
            traffic_data=[{"timestamp": "2023-01-01", "src_ip": "192.168.1.1"}],
            protocol="dicom",
        )

        assert len(request.traffic_data) == 1
        assert request.protocol == "dicom"

    def test_default_protocol(self):
        """Test default protocol value."""
        request = AnomalyRequest(traffic_data=[])
        assert request.protocol == "dicom"


class TestAnomalyResponse:
    """Test AnomalyResponse model."""

    def test_creation(self):
        """Test response creation."""
        response = AnomalyResponse(
            total_records=100,
            anomalies_detected=5,
            anomaly_rate=5.0,
            alerts=[{"type": "test", "severity": "high"}],
        )

        assert response.total_records == 100
        assert response.anomalies_detected == 5
        assert response.anomaly_rate == 5.0


class TestCVERequest:
    """Test CVERequest model."""

    def test_creation(self):
        """Test request creation."""
        request = CVERequest(cve_id="CVE-2024-1234")
        assert request.cve_id == "CVE-2024-1234"


class TestCVEResponse:
    """Test CVEResponse model."""

    def test_creation(self):
        """Test response creation."""
        response = CVEResponse(
            cve_id="CVE-2024-1234",
            description="Test vulnerability",
            cvss_score=7.5,
            severity="high",
            affected_products=["Product A"],
            references=["https://example.com"],
            clinical_impact="May affect device operation",
        )

        assert response.cve_id == "CVE-2024-1234"
        assert response.cvss_score == 7.5
        assert response.clinical_impact is not None

    def test_optional_fields(self):
        """Test optional fields."""
        response = CVEResponse(
            cve_id="CVE-2024-5678",
            description="Test",
            severity="medium",
            affected_products=[],
            references=[],
        )

        assert response.cvss_score is None
        assert response.clinical_impact is None


class TestAdversarialRequest:
    """Test AdversarialRequest model."""

    def test_creation(self):
        """Test request creation."""
        request = AdversarialRequest(
            model_type="cnn",
            attack_method="fgsm",
            epsilon=0.03,
        )

        assert request.model_type == "cnn"
        assert request.attack_method == "fgsm"
        assert request.epsilon == 0.03

    def test_defaults(self):
        """Test default values."""
        request = AdversarialRequest(model_type="resnet")

        assert request.attack_method == "fgsm"
        assert request.epsilon == 0.03

    def test_epsilon_bounds(self):
        """Test epsilon validation."""
        # Valid epsilon
        request = AdversarialRequest(model_type="cnn", epsilon=0.5)
        assert request.epsilon == 0.5

        # Invalid epsilon (too high)
        with pytest.raises(Exception):
            AdversarialRequest(model_type="cnn", epsilon=1.5)


class TestAdversarialResponse:
    """Test AdversarialResponse model."""

    def test_creation(self):
        """Test response creation."""
        response = AdversarialResponse(
            attack_method="fgsm",
            clean_accuracy=0.95,
            adversarial_accuracy=0.70,
            robustness_score=73.7,
            recommendations=["Use adversarial training"],
        )

        assert response.attack_method == "fgsm"
        assert response.clean_accuracy == 0.95
        assert response.robustness_score == 73.7


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.1.0"
        assert "timestamp" in data

    def test_readiness_check(self, client):
        """Test /ready endpoint."""
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["version"] == "1.1.0"

    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        content = response.text
        assert "medtech_api_requests_total" in content
        assert "medtech_api_health" in content


# =============================================================================
# SBOM Analysis Endpoint Tests
# =============================================================================


class TestSBOMEndpoint:
    """Test SBOM analysis endpoint."""

    def test_analyze_sbom_basic(self, client):
        """Test basic SBOM analysis."""
        request_data = {
            "sbom": {
                "components": [
                    {"name": "package-a", "version": "1.0.0"},
                    {"name": "package-b", "version": "2.0.0"},
                ]
            },
            "format": "cyclonedx",
        }

        response = client.post("/api/v1/sbom/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "risk_level" in data
        assert "risk_score" in data
        assert data["total_packages"] == 2
        assert "recommendations" in data
        assert "fda_compliance_notes" in data

    def test_analyze_sbom_empty_components(self, client):
        """Test SBOM with empty components."""
        request_data = {"sbom": {"components": []}}

        response = client.post("/api/v1/sbom/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["total_packages"] == 0

    def test_analyze_sbom_spdx_format(self, client):
        """Test SBOM in SPDX format."""
        request_data = {
            "sbom": {"packages": [{"name": "test-package"}]},
            "format": "spdx",
        }

        response = client.post("/api/v1/sbom/analyze", json=request_data)

        assert response.status_code == 200

    def test_analyze_sbom_without_gnn(self, client):
        """Test SBOM analysis without GNN."""
        request_data = {
            "sbom": {"components": []},
            "include_gnn": False,
        }

        response = client.post("/api/v1/sbom/analyze", json=request_data)

        assert response.status_code == 200

    def test_analyze_sbom_missing_sbom(self, client):
        """Test request missing SBOM field."""
        response = client.post("/api/v1/sbom/analyze", json={})

        assert response.status_code == 422  # Validation error


# =============================================================================
# Anomaly Detection Endpoint Tests
# =============================================================================


class TestAnomalyEndpoint:
    """Test anomaly detection endpoint."""

    def test_detect_anomalies_basic(self, client):
        """Test basic anomaly detection."""
        request_data = {
            "traffic_data": [
                {"src_ip": "192.168.1.1", "dst_ip": "192.168.1.2"},
                {"src_ip": "192.168.1.3", "dst_ip": "192.168.1.4"},
            ],
            "protocol": "dicom",
        }

        response = client.post("/api/v1/anomaly/detect", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 2
        assert "anomalies_detected" in data
        assert "anomaly_rate" in data
        assert "alerts" in data

    def test_detect_anomalies_hl7(self, client):
        """Test HL7 anomaly detection."""
        request_data = {
            "traffic_data": [{"message_type": "ADT", "event": "A01"}],
            "protocol": "hl7",
        }

        response = client.post("/api/v1/anomaly/detect", json=request_data)

        assert response.status_code == 200

    def test_detect_anomalies_empty_data(self, client):
        """Test with empty traffic data."""
        request_data = {"traffic_data": []}

        response = client.post("/api/v1/anomaly/detect", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 0
        assert data["anomaly_rate"] == 0

    def test_detect_anomalies_large_dataset(self, client):
        """Test with larger dataset."""
        request_data = {
            "traffic_data": [{"id": i, "src_ip": f"192.168.1.{i % 255}"} for i in range(100)]
        }

        response = client.post("/api/v1/anomaly/detect", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 100

    def test_detect_anomalies_missing_data(self, client):
        """Test request missing traffic_data."""
        response = client.post("/api/v1/anomaly/detect", json={})

        assert response.status_code == 422


# =============================================================================
# CVE Lookup Endpoint Tests
# =============================================================================


class TestCVEEndpoint:
    """Test CVE lookup endpoint."""

    def test_get_cve_valid(self, client):
        """Test valid CVE lookup."""
        response = client.get("/api/v1/cve/CVE-2024-1234")

        assert response.status_code == 200
        data = response.json()
        assert data["cve_id"] == "CVE-2024-1234"
        assert "description" in data
        assert "cvss_score" in data
        assert "severity" in data
        assert "affected_products" in data
        assert "references" in data

    def test_get_cve_lowercase(self, client):
        """Test CVE lookup with lowercase input."""
        response = client.get("/api/v1/cve/cve-2024-5678")

        assert response.status_code == 200
        data = response.json()
        assert data["cve_id"] == "CVE-2024-5678"

    def test_get_cve_invalid_format(self, client):
        """Test CVE lookup with invalid format."""
        response = client.get("/api/v1/cve/INVALID-1234")

        assert response.status_code == 400
        data = response.json()
        assert "Invalid CVE ID format" in data["detail"]

    def test_get_cve_clinical_impact(self, client):
        """Test CVE includes clinical impact."""
        response = client.get("/api/v1/cve/CVE-2024-9999")

        assert response.status_code == 200
        data = response.json()
        assert "clinical_impact" in data


# =============================================================================
# Adversarial ML Endpoint Tests
# =============================================================================


class TestAdversarialEndpoint:
    """Test adversarial ML testing endpoint."""

    def test_test_adversarial_fgsm(self, client):
        """Test FGSM attack method."""
        request_data = {
            "model_type": "cnn",
            "attack_method": "fgsm",
            "epsilon": 0.03,
        }

        response = client.post("/api/v1/adversarial/test", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["attack_method"] == "fgsm"
        assert "clean_accuracy" in data
        assert "adversarial_accuracy" in data
        assert "robustness_score" in data
        assert "recommendations" in data

    def test_test_adversarial_pgd(self, client):
        """Test PGD attack method."""
        request_data = {
            "model_type": "resnet",
            "attack_method": "pgd",
            "epsilon": 0.1,
        }

        response = client.post("/api/v1/adversarial/test", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["attack_method"] == "pgd"

    def test_test_adversarial_cw(self, client):
        """Test C&W attack method."""
        request_data = {
            "model_type": "vgg",
            "attack_method": "cw",
            "epsilon": 0.05,
        }

        response = client.post("/api/v1/adversarial/test", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["attack_method"] == "cw"

    def test_test_adversarial_defaults(self, client):
        """Test with default parameters."""
        request_data = {"model_type": "cnn"}

        response = client.post("/api/v1/adversarial/test", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["attack_method"] == "fgsm"

    def test_test_adversarial_missing_model(self, client):
        """Test request missing model_type."""
        response = client.post("/api/v1/adversarial/test", json={})

        assert response.status_code == 422

    def test_test_adversarial_recommendations(self, client):
        """Test that recommendations are returned."""
        request_data = {"model_type": "cnn", "attack_method": "fgsm"}

        response = client.post("/api/v1/adversarial/test", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) > 0


# =============================================================================
# Error Handler Tests
# =============================================================================


class TestErrorHandlers:
    """Test API error handlers."""

    def test_not_found_404(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent/endpoint")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.put("/health")  # Only GET allowed

        assert response.status_code == 405


# =============================================================================
# CORS and Middleware Tests
# =============================================================================


class TestMiddleware:
    """Test middleware configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS should allow the request
        assert response.status_code in [200, 204]


# =============================================================================
# OpenAPI Documentation Tests
# =============================================================================


class TestOpenAPI:
    """Test OpenAPI documentation."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "MedTech AI Security API"
        assert schema["info"]["version"] == "1.1.0"

    def test_openapi_paths(self, client):
        """Test OpenAPI includes all paths."""
        response = client.get("/openapi.json")

        schema = response.json()
        paths = schema["paths"]

        assert "/health" in paths
        assert "/ready" in paths
        assert "/metrics" in paths
        assert "/api/v1/sbom/analyze" in paths
        assert "/api/v1/anomaly/detect" in paths
        assert "/api/v1/cve/{cve_id}" in paths
        assert "/api/v1/adversarial/test" in paths

    def test_swagger_ui(self, client):
        """Test Swagger UI is available."""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_redoc(self, client):
        """Test ReDoc is available."""
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "redoc" in response.text.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestAPIIntegration:
    """Integration tests for API workflows."""

    def test_full_security_assessment_workflow(self, client):
        """Test complete security assessment workflow."""
        # Step 1: Health check
        health = client.get("/health")
        assert health.status_code == 200

        # Step 2: Analyze SBOM
        sbom_response = client.post(
            "/api/v1/sbom/analyze",
            json={"sbom": {"components": [{"name": "openssl", "version": "1.1.1"}]}},
        )
        assert sbom_response.status_code == 200

        # Step 3: Check specific CVE
        cve_response = client.get("/api/v1/cve/CVE-2024-0001")
        assert cve_response.status_code == 200

        # Step 4: Analyze traffic
        traffic_response = client.post(
            "/api/v1/anomaly/detect",
            json={"traffic_data": [{"src_ip": "192.168.1.1"}]},
        )
        assert traffic_response.status_code == 200

        # Step 5: Test model robustness
        adversarial_response = client.post(
            "/api/v1/adversarial/test",
            json={"model_type": "cnn"},
        )
        assert adversarial_response.status_code == 200

    def test_multiple_sbom_formats(self, client):
        """Test multiple SBOM formats in sequence."""
        # CycloneDX
        cyclonedx = client.post(
            "/api/v1/sbom/analyze",
            json={"sbom": {"components": []}, "format": "cyclonedx"},
        )
        assert cyclonedx.status_code == 200

        # SPDX
        spdx = client.post(
            "/api/v1/sbom/analyze",
            json={"sbom": {"packages": []}, "format": "spdx"},
        )
        assert spdx.status_code == 200
