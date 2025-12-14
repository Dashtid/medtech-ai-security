"""
Tests for DefectDojo API Integration

Tests the DefectDojoClient class and related functionality.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from medtech_ai_security.integration.defectdojo import (
    DefectDojoClient,
    DefectDojoConfig,
    Finding,
)


class TestDefectDojoConfig:
    """Tests for DefectDojoConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DefectDojoConfig(
            url="https://defectdojo.example.com",
            api_key="test-api-key",
        )
        assert config.url == "https://defectdojo.example.com"
        assert config.api_key == "test-api-key"
        assert config.verify_ssl is True
        assert config.timeout == 30
        assert config.product_type == "Medical Device"
        assert config.default_severity == "Medium"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DefectDojoConfig(
            url="https://custom.example.com",
            api_key="custom-key",
            verify_ssl=False,
            timeout=60,
            product_type="Custom Type",
            default_severity="High",
        )
        assert config.verify_ssl is False
        assert config.timeout == 60
        assert config.product_type == "Custom Type"
        assert config.default_severity == "High"


class TestFinding:
    """Tests for Finding dataclass."""

    def test_basic_finding(self):
        """Test basic finding creation."""
        finding = Finding(
            title="Test Finding",
            description="Test description",
            severity="High",
        )
        assert finding.title == "Test Finding"
        assert finding.description == "Test description"
        assert finding.severity == "High"
        assert finding.cve is None
        assert finding.verified is False
        assert finding.active is True

    def test_finding_with_all_fields(self):
        """Test finding with all fields populated."""
        finding = Finding(
            title="Complete Finding",
            description="Full description",
            severity="Critical",
            cve="CVE-2024-1234",
            cvss_score=9.8,
            cwe=79,
            references="https://example.com",
            mitigation="Apply patch",
            impact="Remote code execution",
            component_name="vulnerable-lib",
            component_version="1.0.0",
            file_path="/src/app.py",
            line=42,
            verified=True,
            active=True,
            duplicate=False,
            static_finding=True,
            dynamic_finding=False,
            tags=["critical", "rce"],
            endpoints=["https://api.example.com"],
        )
        assert finding.cve == "CVE-2024-1234"
        assert finding.cvss_score == 9.8
        assert finding.cwe == 79
        assert finding.component_name == "vulnerable-lib"
        assert finding.line == 42
        assert finding.verified is True
        assert "critical" in finding.tags

    def test_finding_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        finding = Finding(
            title="Minimal Finding",
            description="Description",
            severity="Low",
        )
        result = finding.to_dict()
        assert result["title"] == "Minimal Finding"
        assert result["description"] == "Description"
        assert result["severity"] == "Low"
        assert result["verified"] is False
        assert result["active"] is True
        assert "cve" not in result
        assert "cvssv3_score" not in result

    def test_finding_to_dict_full(self):
        """Test to_dict with all fields."""
        finding = Finding(
            title="Full Finding",
            description="Full description",
            severity="High",
            cve="CVE-2024-5678",
            cvss_score=7.5,
            cwe=89,
            references="https://nvd.nist.gov",
            mitigation="Update library",
            impact="Data breach",
            component_name="sql-lib",
            component_version="2.0.0",
            file_path="/db/query.py",
            line=100,
            tags=["sqli", "database"],
        )
        result = finding.to_dict()
        assert result["cve"] == "CVE-2024-5678"
        assert result["cvssv3_score"] == 7.5
        assert result["cwe"] == 89
        assert result["references"] == "https://nvd.nist.gov"
        assert result["mitigation"] == "Update library"
        assert result["impact"] == "Data breach"
        assert result["component_name"] == "sql-lib"
        assert result["component_version"] == "2.0.0"
        assert result["file_path"] == "/db/query.py"
        assert result["line"] == 100
        assert result["tags"] == ["sqli", "database"]

    def test_finding_title_truncation(self):
        """Test that long titles are truncated to 200 chars."""
        long_title = "A" * 300
        finding = Finding(
            title=long_title,
            description="Description",
            severity="Medium",
        )
        result = finding.to_dict()
        assert len(result["title"]) == 200


class TestDefectDojoClient:
    """Tests for DefectDojoClient class."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DefectDojoClient(
            url="https://defectdojo.example.com",
            api_key="test-api-key",
            verify_ssl=False,
        )

    @pytest.fixture
    def mock_response(self):
        """Create a mock response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"results": [], "count": 0}
        return response

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.base_url == "https://defectdojo.example.com"
        assert client.api_url == "https://defectdojo.example.com/api/v2"
        assert "Token test-api-key" in client.headers["Authorization"]
        assert client.verify_ssl is False

    def test_client_url_trailing_slash_removed(self):
        """Test that trailing slash is removed from URL."""
        client = DefectDojoClient(
            url="https://defectdojo.example.com/",
            api_key="test-key",
        )
        assert client.base_url == "https://defectdojo.example.com"

    @patch("requests.Session.request")
    def test_request_success(self, mock_request, client):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test"}
        mock_request.return_value = mock_response

        result = client._request("GET", "/products/")
        assert result == {"id": 1, "name": "Test"}
        mock_request.assert_called_once()

    @patch("requests.Session.request")
    def test_request_no_content(self, mock_request, client):
        """Test request with 204 No Content response."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_request.return_value = mock_response

        result = client._request("DELETE", "/products/1/")
        assert result == {"success": True}

    @patch("requests.Session.request")
    def test_request_failure(self, mock_request, client):
        """Test API request failure handling."""
        import requests

        mock_request.side_effect = requests.exceptions.RequestException("Network error")

        result = client._request("GET", "/products/")
        assert result is None

    @patch("requests.Session.request")
    def test_get_products(self, mock_request, client):
        """Test getting products."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"id": 1, "name": "Product A"}],
            "count": 1,
        }
        mock_request.return_value = mock_response

        products = client.get_products()
        assert len(products) == 1
        assert products[0]["name"] == "Product A"

    @patch("requests.Session.request")
    def test_get_products_by_name(self, mock_request, client):
        """Test getting products filtered by name."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"id": 2, "name": "Medical Device"}],
            "count": 1,
        }
        mock_request.return_value = mock_response

        products = client.get_products(name="Medical Device")
        assert len(products) == 1
        # Verify name parameter was passed
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["params"]["name"] == "Medical Device"

    @patch("requests.Session.request")
    def test_create_product(self, mock_request, client):
        """Test creating a product."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 3, "name": "New Product"}
        mock_request.return_value = mock_response

        result = client.create_product(
            name="New Product",
            description="Test description",
            tags=["medical", "device"],
        )
        assert result["id"] == 3
        assert result["name"] == "New Product"

    @patch("requests.Session.request")
    def test_get_or_create_product_existing(self, mock_request, client):
        """Test get_or_create_product when product exists."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"id": 5, "name": "Existing Product"}],
            "count": 1,
        }
        mock_request.return_value = mock_response

        product_id = client.get_or_create_product("Existing Product")
        assert product_id == 5

    @patch("requests.Session.request")
    def test_get_or_create_product_new(self, mock_request, client):
        """Test get_or_create_product when product doesn't exist."""
        # First call returns empty results, second creates product
        responses = [
            MagicMock(status_code=200, json=MagicMock(return_value={"results": [], "count": 0})),
            MagicMock(status_code=201, json=MagicMock(return_value={"id": 10, "name": "New"})),
        ]
        mock_request.side_effect = responses

        product_id = client.get_or_create_product("New")
        assert product_id == 10

    @patch("requests.Session.request")
    def test_create_engagement(self, mock_request, client):
        """Test creating an engagement."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": 1,
            "name": "Security Scan",
            "product": 5,
        }
        mock_request.return_value = mock_response

        result = client.create_engagement(
            product_id=5,
            name="Security Scan",
            description="Automated scan",
        )
        assert result["id"] == 1
        assert result["name"] == "Security Scan"

    @patch("requests.Session.request")
    def test_close_engagement(self, mock_request, client):
        """Test closing an engagement."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "status": "Completed"}
        mock_request.return_value = mock_response

        result = client.close_engagement(1)
        assert result["status"] == "Completed"

    @patch("requests.Session.request")
    def test_create_test(self, mock_request, client):
        """Test creating a test."""
        # First call gets test type, second creates test
        responses = [
            MagicMock(
                status_code=200,
                json=MagicMock(return_value={"results": [{"id": 1, "name": "SBOM Analysis"}]}),
            ),
            MagicMock(
                status_code=201,
                json=MagicMock(return_value={"id": 10, "title": "SBOM Test"}),
            ),
        ]
        mock_request.side_effect = responses

        result = client.create_test(
            engagement_id=1,
            test_type="SBOM Analysis",
            title="SBOM Test",
        )
        assert result["id"] == 10

    @patch("requests.Session.request")
    def test_create_finding(self, mock_request, client):
        """Test creating a finding."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": 100,
            "title": "Test Finding",
            "severity": "High",
        }
        mock_request.return_value = mock_response

        finding = Finding(
            title="Test Finding",
            description="Description",
            severity="High",
            cve="CVE-2024-1234",
        )
        result = client.create_finding(test_id=10, finding=finding)
        assert result["id"] == 100

    @patch("requests.Session.request")
    def test_create_findings_batch(self, mock_request, client):
        """Test creating multiple findings."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1, "title": "Finding"}
        mock_request.return_value = mock_response

        findings = [
            Finding(title="Finding 1", description="Desc 1", severity="High"),
            Finding(title="Finding 2", description="Desc 2", severity="Medium"),
        ]
        created = client.create_findings_batch(test_id=10, findings=findings)
        assert len(created) == 2

    @patch("requests.Session.request")
    def test_test_connection_success(self, mock_request, client):
        """Test successful connection test."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [], "count": 0}
        mock_request.return_value = mock_response

        assert client.test_connection() is True

    @patch("requests.Session.request")
    def test_test_connection_failure(self, mock_request, client):
        """Test failed connection test."""
        import requests

        mock_request.side_effect = requests.exceptions.RequestException("Failed")

        assert client.test_connection() is False


class TestDefectDojoClientImportMethods:
    """Tests for import methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DefectDojoClient(
            url="https://defectdojo.example.com",
            api_key="test-api-key",
        )

    def test_convert_sbom_to_findings(self, client):
        """Test converting SBOM report to findings."""
        report = {
            "risk_report": {
                "package_details": [
                    {
                        "package_name": "vulnerable-lib",
                        "package_version": "1.0.0",
                        "risk_level": "high",
                        "risk_score": 75.0,
                        "vulnerability_count": 2,
                        "breakdown": {
                            "vulnerability": 50.0,
                            "license": 10.0,
                            "dependency": 10.0,
                            "position": 5.0,
                        },
                        "recommendations": ["Update to version 2.0.0"],
                    },
                    {
                        "package_name": "safe-lib",
                        "package_version": "2.0.0",
                        "risk_level": "low",
                        "risk_score": 10.0,
                        "vulnerability_count": 0,
                    },
                ]
            }
        }

        findings = client._convert_sbom_to_findings(report)
        # Should only include packages with vulnerabilities
        assert len(findings) == 1
        assert findings[0].component_name == "vulnerable-lib"
        assert findings[0].severity == "High"
        assert "sbom" in findings[0].tags

    def test_convert_sbom_to_findings_empty(self, client):
        """Test converting empty SBOM report."""
        report = {"risk_report": {"package_details": []}}
        findings = client._convert_sbom_to_findings(report)
        assert len(findings) == 0

    def test_convert_cves_to_findings(self, client):
        """Test converting CVE data to findings."""
        cve_data = [
            {
                "cve_id": "CVE-2024-1234",
                "cvss_score": 9.5,
                "description": "Critical vulnerability",
                "published_date": "2024-01-15",
                "affected_products": ["Device A", "Device B"],
                "clinical_impact": "Patient safety risk",
                "device_type": "infusion-pump",
                "mitigation": "Apply firmware update",
                "references": ["https://nvd.nist.gov/cve-2024-1234"],
            },
            {
                "cve_id": "CVE-2024-5678",
                "cvss_score": 4.5,
                "description": "Low severity issue",
            },
        ]

        findings = client._convert_cves_to_findings(cve_data)
        assert len(findings) == 2

        # Check critical CVE
        critical = findings[0]
        assert critical.cve == "CVE-2024-1234"
        assert critical.severity == "Critical"
        assert critical.cvss_score == 9.5
        assert "threat-intel" in critical.tags

        # Check low CVE
        low = findings[1]
        assert low.cve == "CVE-2024-5678"
        assert low.severity == "Medium"

    def test_severity_mapping(self, client):
        """Test CVSS to severity mapping."""
        # Test different CVSS ranges
        test_cases = [
            ({"cve_id": "CVE-1", "cvss_score": 9.8}, "Critical"),
            ({"cve_id": "CVE-2", "cvss_score": 7.5}, "High"),
            ({"cve_id": "CVE-3", "cvss_score": 5.0}, "Medium"),
            ({"cve_id": "CVE-4", "cvss_score": 2.0}, "Low"),
            ({"cve_id": "CVE-5", "cvss_score": 0.0}, "Low"),
        ]

        for cve_data, expected_severity in test_cases:
            findings = client._convert_cves_to_findings([cve_data])
            assert (
                findings[0].severity == expected_severity
            ), f"Failed for CVSS {cve_data['cvss_score']}"

    @patch.object(DefectDojoClient, "create_engagement")
    @patch.object(DefectDojoClient, "create_test")
    @patch.object(DefectDojoClient, "create_findings_batch")
    @patch.object(DefectDojoClient, "close_engagement")
    def test_import_sbom_findings(self, mock_close, mock_batch, mock_test, mock_engagement, client):
        """Test importing SBOM findings."""
        mock_engagement.return_value = {"id": 1, "name": "Test"}
        mock_test.return_value = {"id": 10, "title": "SBOM"}
        mock_batch.return_value = [{"id": 100}]
        mock_close.return_value = {"status": "Completed"}

        report = {
            "risk_report": {
                "package_details": [
                    {
                        "package_name": "lib",
                        "package_version": "1.0",
                        "risk_level": "high",
                        "risk_score": 80.0,
                        "vulnerability_count": 1,
                        "breakdown": {},
                    }
                ]
            }
        }

        result = client.import_sbom_findings(
            product_id=5,
            analysis_report=report,
            engagement_name="Test SBOM",
        )

        assert result["engagement_id"] == 1
        assert result["test_id"] == 10
        assert result["findings_created"] == 1
        mock_engagement.assert_called_once()
        mock_test.assert_called_once()
        mock_close.assert_called_once()

    @patch.object(DefectDojoClient, "create_engagement")
    def test_import_sbom_findings_engagement_failure(self, mock_engagement, client):
        """Test import failure when engagement creation fails."""
        mock_engagement.return_value = None

        result = client.import_sbom_findings(
            product_id=5,
            analysis_report={},
        )

        assert "error" in result
        assert "engagement" in result["error"].lower()

    @patch.object(DefectDojoClient, "create_engagement")
    @patch.object(DefectDojoClient, "create_test")
    def test_import_sbom_findings_test_failure(self, mock_test, mock_engagement, client):
        """Test import failure when test creation fails."""
        mock_engagement.return_value = {"id": 1}
        mock_test.return_value = None

        result = client.import_sbom_findings(
            product_id=5,
            analysis_report={},
        )

        assert "error" in result
        assert "test" in result["error"].lower()

    @patch.object(DefectDojoClient, "create_engagement")
    @patch.object(DefectDojoClient, "create_test")
    @patch.object(DefectDojoClient, "create_findings_batch")
    @patch.object(DefectDojoClient, "close_engagement")
    def test_import_threat_intel(self, mock_close, mock_batch, mock_test, mock_engagement, client):
        """Test importing threat intelligence CVEs."""
        mock_engagement.return_value = {"id": 2, "name": "Threat Intel"}
        mock_test.return_value = {"id": 20, "title": "CVE Analysis"}
        mock_batch.return_value = [{"id": 200}, {"id": 201}]
        mock_close.return_value = {"status": "Completed"}

        cve_data = [
            {"cve_id": "CVE-2024-1111", "cvss_score": 8.0, "description": "Test 1"},
            {"cve_id": "CVE-2024-2222", "cvss_score": 5.0, "description": "Test 2"},
        ]

        result = client.import_threat_intel(
            product_id=5,
            cve_data=cve_data,
            engagement_name="Threat Analysis",
        )

        assert result["engagement_id"] == 2
        assert result["test_id"] == 20
        assert result["findings_created"] == 2
        assert result["findings_total"] == 2


class TestDefectDojoClientUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DefectDojoClient(
            url="https://defectdojo.example.com",
            api_key="test-api-key",
        )

    @patch("requests.Session.request")
    def test_get_product_types(self, mock_request, client):
        """Test getting product types."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "name": "Medical Device"},
                {"id": 2, "name": "Web Application"},
            ]
        }
        mock_request.return_value = mock_response

        types = client.get_product_types()
        assert len(types) == 2
        assert types[0]["name"] == "Medical Device"

    @patch("requests.Session.request")
    def test_get_findings(self, mock_request, client):
        """Test getting findings with filters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"id": 1, "title": "Finding 1"}]}
        mock_request.return_value = mock_response

        findings = client.get_findings(test_id=10, severity="High", active=True)
        assert len(findings) == 1

        # Verify parameters
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["params"]["test"] == 10
        assert call_kwargs["params"]["severity"] == "High"
        assert call_kwargs["params"]["active"] is True

    @patch("requests.Session.request")
    def test_export_findings_json(self, mock_request, client, tmp_path):
        """Test exporting findings to JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Finding 1"},
                {"id": 2, "title": "Finding 2"},
            ]
        }
        mock_request.return_value = mock_response

        output_file = tmp_path / "findings.json"
        result = client.export_findings_json(test_id=10, output_path=output_file)

        assert result is True
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 2

    @patch("requests.Session.request")
    def test_export_findings_json_empty(self, mock_request, client, tmp_path):
        """Test exporting empty findings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_request.return_value = mock_response

        output_file = tmp_path / "empty.json"
        result = client.export_findings_json(test_id=10, output_path=output_file)

        assert result is False
        assert not output_file.exists()


class TestSeverityMap:
    """Tests for severity mapping."""

    def test_severity_map_values(self):
        """Test all severity map values."""
        expected = {
            "critical": "Critical",
            "high": "High",
            "medium": "Medium",
            "low": "Low",
            "info": "Informational",
            "informational": "Informational",
        }
        assert DefectDojoClient.SEVERITY_MAP == expected

    def test_test_types(self):
        """Test test type constants."""
        assert "sbom_analysis" in DefectDojoClient.TEST_TYPES
        assert "threat_intel" in DefectDojoClient.TEST_TYPES
        assert "adversarial_ml" in DefectDojoClient.TEST_TYPES


class TestCLI:
    """Tests for CLI functionality."""

    @patch(
        "sys.argv", ["medsec-defectdojo", "--url", "https://test.com", "--api-key", "key", "--test"]
    )
    @patch.object(DefectDojoClient, "test_connection")
    def test_cli_test_connection(self, mock_test, capsys):
        """Test CLI connection test."""
        mock_test.return_value = True

        from medtech_ai_security.integration.defectdojo import main

        main()

        captured = capsys.readouterr()
        assert "[OK]" in captured.out

    @patch(
        "sys.argv", ["medsec-defectdojo", "--url", "https://test.com", "--api-key", "key", "--test"]
    )
    @patch.object(DefectDojoClient, "test_connection")
    def test_cli_test_connection_failure(self, mock_test, capsys):
        """Test CLI connection test failure."""
        mock_test.return_value = False

        from medtech_ai_security.integration.defectdojo import main

        main()

        captured = capsys.readouterr()
        assert "[FAIL]" in captured.out

    @patch(
        "sys.argv",
        ["medsec-defectdojo", "--url", "https://test.com", "--api-key", "key", "--list-products"],
    )
    @patch.object(DefectDojoClient, "get_products")
    def test_cli_list_products(self, mock_products, capsys):
        """Test CLI list products."""
        mock_products.return_value = [
            {"id": 1, "name": "Product A"},
            {"id": 2, "name": "Product B"},
        ]

        from medtech_ai_security.integration.defectdojo import main

        main()

        captured = capsys.readouterr()
        assert "Product A" in captured.out
        assert "Product B" in captured.out
        assert "2 products" in captured.out
