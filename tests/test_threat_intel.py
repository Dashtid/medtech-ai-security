"""Unit tests for Phase 1: NLP Threat Intelligence."""

import json
from unittest.mock import Mock, patch

from medtech_ai_security.threat_intel.cisa_scraper import (
    CISAAdvisory,
    CISAScraper,
)
from medtech_ai_security.threat_intel.claude_processor import (
    generate_summary_report,
    load_claude_response,
    merge_analysis,
)
from medtech_ai_security.threat_intel.nvd_scraper import (
    MEDICAL_DEVICE_KEYWORDS,
    CVEEntry,
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


class TestClaudeProcessor:
    """Test Claude response processor functionality."""

    def test_load_claude_response_json(self, tmp_path):
        """Test loading a plain JSON response."""
        response_data = {
            "analyses": [
                {
                    "cve_id": "CVE-2024-1234",
                    "device_type": "imaging",
                    "clinical_impact": "HIGH",
                    "exploitability": "EASY",
                    "reasoning": "Test reasoning",
                }
            ]
        }

        response_file = tmp_path / "response.json"
        with open(response_file, "w") as f:
            json.dump(response_data, f)

        loaded = load_claude_response(response_file)

        assert "analyses" in loaded
        assert loaded["analyses"][0]["cve_id"] == "CVE-2024-1234"

    def test_load_claude_response_markdown(self, tmp_path):
        """Test loading JSON wrapped in markdown code block."""
        response_content = '''```json
{
    "analyses": [
        {
            "cve_id": "CVE-2024-1234",
            "device_type": "imaging"
        }
    ]
}
```'''

        response_file = tmp_path / "response.md"
        with open(response_file, "w") as f:
            f.write(response_content)

        loaded = load_claude_response(response_file)

        assert "analyses" in loaded
        assert loaded["analyses"][0]["cve_id"] == "CVE-2024-1234"

    def test_merge_analysis(self, tmp_path):
        """Test merging Claude analysis into CVE data."""
        # Create CVE file
        cve_data = {
            "metadata": {"generated_at": "2024-01-15T10:00:00Z"},
            "cves": [
                {
                    "cve_id": "CVE-2024-1234",
                    "description": "Test CVE",
                    "cvss_v3_score": 9.8,
                }
            ],
        }
        cve_file = tmp_path / "cves.json"
        with open(cve_file, "w") as f:
            json.dump(cve_data, f)

        # Create response file
        response_data = {
            "analyses": [
                {
                    "cve_id": "CVE-2024-1234",
                    "device_type": "imaging",
                    "clinical_impact": "HIGH",
                    "exploitability": "EASY",
                    "reasoning": "Affects DICOM systems",
                }
            ]
        }
        response_file = tmp_path / "response.json"
        with open(response_file, "w") as f:
            json.dump(response_data, f)

        # Merge
        output_file = tmp_path / "enriched.json"
        result = merge_analysis(cve_file, response_file, output_file)

        # Verify
        assert result["cves"][0]["device_type"] == "imaging"
        assert result["cves"][0]["clinical_impact"] == "HIGH"
        assert result["cves"][0]["ai_reasoning"] == "Affects DICOM systems"
        assert result["metadata"]["enriched_count"] == 1
        assert output_file.exists()

    def test_merge_analysis_default_output(self, tmp_path):
        """Test merge_analysis with default output path."""
        cve_data = {
            "metadata": {"generated_at": "2024-01-15"},
            "cves": [{"cve_id": "CVE-2024-1234", "description": "Test"}],
        }
        cve_file = tmp_path / "test_cves.json"
        with open(cve_file, "w") as f:
            json.dump(cve_data, f)

        response_data = {"analyses": []}
        response_file = tmp_path / "response.json"
        with open(response_file, "w") as f:
            json.dump(response_data, f)

        # Merge without specifying output
        _result = merge_analysis(cve_file, response_file)  # noqa: F841

        # Default output should be *_enriched.json
        expected_output = tmp_path / "test_cves_enriched.json"
        assert expected_output.exists()

    def test_generate_summary_report(self, tmp_path):
        """Test generating summary report from enriched data."""
        enriched_data = {
            "metadata": {
                "generated_at": "2024-01-15T10:00:00Z",
                "enriched_at": "2024-01-16T10:00:00Z",
                "enriched_count": 3,
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
                    "description": "HL7 parser overflow",
                    "cvss_v3_score": 7.5,
                    "cvss_v3_severity": "HIGH",
                    "device_type": "healthcare_it",
                    "clinical_impact": "MEDIUM",
                    "exploitability": "MODERATE",
                },
                {
                    "cve_id": "CVE-2024-1003",
                    "description": "Low severity issue",
                    "cvss_v3_score": 3.0,
                    "cvss_v3_severity": "LOW",
                    "device_type": "monitoring",
                    "clinical_impact": "LOW",
                    "exploitability": "HARD",
                },
            ],
        }

        report_file = tmp_path / "report.txt"
        report = generate_summary_report(enriched_data, report_file)

        # Verify report content
        assert "MEDICAL DEVICE THREAT INTELLIGENCE REPORT" in report
        assert "CRITICAL" in report
        assert "HIGH" in report
        assert "LOW" in report
        assert "imaging" in report
        assert "healthcare_it" in report

        # Verify file was saved
        assert report_file.exists()

    def test_generate_summary_report_high_priority(self, tmp_path):
        """Test report highlights high priority CVEs."""
        enriched_data = {
            "metadata": {
                "generated_at": "2024-01-15",
                "enriched_at": "2024-01-16",
                "enriched_count": 1,
            },
            "cves": [
                {
                    "cve_id": "CVE-2024-1001",
                    "description": "Critical vulnerability with high clinical impact",
                    "cvss_v3_score": 9.8,
                    "cvss_v3_severity": "CRITICAL",
                    "device_type": "imaging",
                    "clinical_impact": "HIGH",
                    "exploitability": "EASY",
                }
            ],
        }

        report = generate_summary_report(enriched_data)

        assert "HIGH PRIORITY" in report
        assert "CVE-2024-1001" in report


class TestCISAAdvisory:
    """Test CISAAdvisory dataclass."""

    def test_advisory_creation(self):
        """Test creating a CISAAdvisory with required fields."""
        advisory = CISAAdvisory(
            advisory_id="ICSMA-24-001-01",
            title="Medical Device Vulnerability",
            url="https://cisa.gov/advisories/icsma-24-001-01",
            release_date="2024-01-15",
        )

        assert advisory.advisory_id == "ICSMA-24-001-01"
        assert advisory.title == "Medical Device Vulnerability"
        assert advisory.advisory_type == "ICSMA"

    def test_advisory_with_all_fields(self):
        """Test CISAAdvisory with all optional fields."""
        advisory = CISAAdvisory(
            advisory_id="ICSMA-24-001-01",
            title="DICOM Server Vulnerability",
            url="https://cisa.gov/advisories/icsma-24-001-01",
            release_date="2024-01-15",
            last_updated="2024-01-20",
            severity="CRITICAL",
            cvss_score=9.8,
            affected_products=["DICOM Server v1.0", "DICOM Server v2.0"],
            cve_ids=["CVE-2024-1234", "CVE-2024-1235"],
            vendor="Medical Devices Inc",
            description="Remote code execution vulnerability",
            mitigation="Update to version 3.0",
            device_type="imaging",
            clinical_impact="HIGH",
            exploitability="EASY",
        )

        assert advisory.cvss_score == 9.8
        assert len(advisory.affected_products) == 2
        assert "CVE-2024-1234" in advisory.cve_ids
        assert advisory.device_type == "imaging"


class TestCISAScraper:
    """Test CISAScraper functionality."""

    def test_scraper_initialization(self):
        """Test scraper initializes with correct defaults."""
        scraper = CISAScraper()

        assert scraper.request_delay == 2.0
        assert scraper.session is not None

    def test_scraper_custom_delay(self):
        """Test scraper with custom request delay."""
        scraper = CISAScraper(request_delay=5.0)

        assert scraper.request_delay == 5.0

    def test_medical_keywords_present(self):
        """Test that medical keywords are defined."""
        assert len(CISAScraper.MEDICAL_KEYWORDS) > 0
        assert "medical" in CISAScraper.MEDICAL_KEYWORDS
        assert "DICOM" in CISAScraper.MEDICAL_KEYWORDS

    @patch.object(CISAScraper, "_make_request")
    def test_parse_advisory_list_page(self, mock_request):
        """Test parsing advisory listing page."""
        from bs4 import BeautifulSoup

        html = '''
        <html>
        <body>
            <a href="/ics-medical-advisories/icsma-24-001-01">
                Medical Device Advisory
            </a>
            <a href="/ics-advisories/icsa-24-002-01">
                General ICS Advisory
            </a>
        </body>
        </html>
        '''

        soup = BeautifulSoup(html, "html.parser")
        scraper = CISAScraper()
        advisories = scraper._parse_advisory_list_page(soup)

        assert len(advisories) == 2
        assert advisories[0]["advisory_id"] == "ICSMA-24-001-01"
        assert advisories[0]["advisory_type"] == "ICSMA"
        assert advisories[1]["advisory_id"] == "ICSA-24-002-01"
        assert advisories[1]["advisory_type"] == "ICSA"

    def test_save_results(self, tmp_path):
        """Test saving advisory results to JSON."""
        scraper = CISAScraper()
        output_file = tmp_path / "cisa_advisories.json"

        advisories = [
            CISAAdvisory(
                advisory_id="ICSMA-24-001-01",
                title="Test Advisory",
                url="https://cisa.gov/test",
                release_date="2024-01-15",
                cvss_score=8.5,
                severity="HIGH",
            )
        ]

        scraper.save_results(advisories, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        assert data["metadata"]["total_advisories"] == 1
        assert data["advisories"][0]["advisory_id"] == "ICSMA-24-001-01"
        assert data["advisories"][0]["cvss_score"] == 8.5

    def test_generate_claude_prompt(self):
        """Test generating Claude prompt from advisories."""
        scraper = CISAScraper()

        advisories = [
            CISAAdvisory(
                advisory_id="ICSMA-24-001-01",
                title="DICOM Server RCE",
                url="https://cisa.gov/test",
                release_date="2024-01-15",
                cvss_score=9.8,
                severity="CRITICAL",
                cve_ids=["CVE-2024-1234"],
                vendor="Medical Devices Inc",
                description="Remote code execution in DICOM server",
            )
        ]

        prompt = scraper.generate_claude_prompt(advisories)

        assert "ICSMA-24-001-01" in prompt
        assert "CRITICAL" in prompt
        assert "CVE-2024-1234" in prompt
        assert "device_type" in prompt
        assert "JSON format" in prompt
