"""Unit tests for Phase 1: NLP Threat Intelligence."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from medtech_ai_security.threat_intel.nvd_scraper import (
    CVEEntry,
    MEDICAL_DEVICE_KEYWORDS,
    NVDScraper,
)


class TestCVEEntry:
    """Test CVEEntry dataclass."""

    def test_cve_entry_creation(self):
        """Test creating a CVEEntry with required fields."""
        entry = CVEEntry(
            cve_id="CVE-2021-44228",
            description="Log4j vulnerability",
            published_date="2021-12-10",
            last_modified_date="2021-12-15",
        )

        assert entry.cve_id == "CVE-2021-44228"
        assert entry.description == "Log4j vulnerability"
        assert entry.cvss_v3_score is None
        assert entry.cwe_ids == []

    def test_cve_entry_with_all_fields(self):
        """Test CVEEntry with all optional fields."""
        entry = CVEEntry(
            cve_id="CVE-2021-44228",
            description="Log4j vulnerability",
            published_date="2021-12-10",
            last_modified_date="2021-12-15",
            cvss_v3_score=10.0,
            cvss_v3_severity="CRITICAL",
            cvss_v3_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            cwe_ids=["CWE-502"],
            matched_keywords=["healthcare"],
            device_type="imaging",
            clinical_impact="HIGH",
        )

        assert entry.cvss_v3_score == 10.0
        assert entry.cvss_v3_severity == "CRITICAL"
        assert "CWE-502" in entry.cwe_ids
        assert entry.device_type == "imaging"


class TestNVDScraper:
    """Test NVDScraper functionality."""

    def test_scraper_initialization(self):
        """Test scraper initializes with correct defaults."""
        scraper = NVDScraper()

        assert scraper.api_key is None
        assert scraper.request_delay == 6.0  # No API key = slower rate

    def test_scraper_with_api_key(self):
        """Test scraper with API key has faster rate limit."""
        scraper = NVDScraper(api_key="test-api-key")

        assert scraper.api_key == "test-api-key"
        assert scraper.request_delay == 0.6  # With API key = faster

    def test_matches_medical_keywords(self):
        """Test keyword matching functionality."""
        scraper = NVDScraper()

        # Test with DICOM keyword
        matched = scraper._matches_medical_keywords(
            "Vulnerability in DICOM server allows remote code execution"
        )
        assert "DICOM" in matched

        # Test with multiple keywords
        matched = scraper._matches_medical_keywords(
            "Healthcare PACS system vulnerability in HL7 interface"
        )
        assert len(matched) >= 2

        # Test with no matching keywords
        matched = scraper._matches_medical_keywords(
            "Generic web application vulnerability"
        )
        assert len(matched) == 0

    def test_parse_cve_basic(self):
        """Test parsing a basic CVE response."""
        scraper = NVDScraper()

        cve_item = {
            "cve": {
                "id": "CVE-2024-1234",
                "descriptions": [
                    {"lang": "en", "value": "Test vulnerability in DICOM viewer"}
                ],
                "published": "2024-01-15T10:00:00.000",
                "lastModified": "2024-01-16T10:00:00.000",
                "vulnStatus": "Analyzed",
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "cvssData": {
                                "baseScore": 7.5,
                                "baseSeverity": "HIGH",
                                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
                            }
                        }
                    ]
                },
                "weaknesses": [
                    {"description": [{"value": "CWE-400"}]}
                ],
                "references": [
                    {"url": "https://example.com/advisory"}
                ],
            }
        }

        entry = scraper._parse_cve(cve_item, ["DICOM"])

        assert entry.cve_id == "CVE-2024-1234"
        assert "DICOM" in entry.description
        assert entry.cvss_v3_score == 7.5
        assert entry.cvss_v3_severity == "HIGH"
        assert "CWE-400" in entry.cwe_ids
        assert "DICOM" in entry.matched_keywords

    def test_parse_cve_missing_metrics(self):
        """Test parsing CVE with missing metrics."""
        scraper = NVDScraper()

        cve_item = {
            "cve": {
                "id": "CVE-2024-5678",
                "descriptions": [{"lang": "en", "value": "Test vulnerability"}],
                "published": "2024-01-15T10:00:00.000",
                "lastModified": "2024-01-16T10:00:00.000",
                "metrics": {},
            }
        }

        entry = scraper._parse_cve(cve_item, [])

        assert entry.cve_id == "CVE-2024-5678"
        assert entry.cvss_v3_score is None
        assert entry.cvss_v3_severity is None

    def test_save_results_json(self, tmp_path):
        """Test saving results to JSON file."""
        scraper = NVDScraper()
        output_file = tmp_path / "test_cves.json"

        cves = [
            CVEEntry(
                cve_id="CVE-2024-1234",
                description="Test vulnerability",
                published_date="2024-01-15",
                last_modified_date="2024-01-16",
                cvss_v3_score=7.5,
            )
        ]

        scraper.save_results(cves, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        assert data["metadata"]["total_cves"] == 1
        assert len(data["cves"]) == 1
        assert data["cves"][0]["cve_id"] == "CVE-2024-1234"

    def test_generate_claude_prompt(self):
        """Test Claude prompt generation."""
        scraper = NVDScraper()

        cves = [
            CVEEntry(
                cve_id="CVE-2024-1234",
                description="Vulnerability in medical DICOM viewer",
                published_date="2024-01-15T10:00:00.000",
                last_modified_date="2024-01-16",
                cvss_v3_score=9.8,
                cvss_v3_severity="CRITICAL",
                cwe_ids=["CWE-798"],
                matched_keywords=["DICOM", "medical"],
            )
        ]

        prompt = scraper.generate_claude_prompt(cves)

        assert "CVE-2024-1234" in prompt
        assert "CRITICAL" in prompt
        assert "DICOM" in prompt
        assert "JSON format" in prompt
        assert "device_type" in prompt

    @patch("requests.Session.get")
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": [], "totalResults": 0}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper = NVDScraper()
        scraper.last_request_time = 0  # Skip rate limiting for test

        result = scraper._make_request({"keywordSearch": "test"})

        assert result["totalResults"] == 0
        mock_get.assert_called_once()


class TestMedicalDeviceKeywords:
    """Test medical device keyword list."""

    def test_keywords_not_empty(self):
        """Test that keyword list is populated."""
        assert len(MEDICAL_DEVICE_KEYWORDS) > 0

    def test_keywords_contains_expected(self):
        """Test expected keywords are present."""
        expected = ["DICOM", "HL7", "PACS", "infusion pump", "patient monitor"]
        for keyword in expected:
            assert keyword in MEDICAL_DEVICE_KEYWORDS

    def test_keywords_are_strings(self):
        """Test all keywords are strings."""
        for keyword in MEDICAL_DEVICE_KEYWORDS:
            assert isinstance(keyword, str)
            assert len(keyword) > 0
