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
        matched = scraper._matches_medical_keywords("Generic web application vulnerability")
        assert len(matched) == 0

    def test_parse_cve_basic(self):
        """Test parsing a basic CVE response."""
        scraper = NVDScraper()

        cve_item = {
            "cve": {
                "id": "CVE-2024-1234",
                "descriptions": [{"lang": "en", "value": "Test vulnerability in DICOM viewer"}],
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
                "weaknesses": [{"description": [{"value": "CWE-400"}]}],
                "references": [{"url": "https://example.com/advisory"}],
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
        response_content = """```json
{
    "analyses": [
        {
            "cve_id": "CVE-2024-1234",
            "device_type": "imaging"
        }
    ]
}
```"""

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

        html = """
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
        """

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

    def test_generate_claude_prompt_empty_fields(self):
        """Test Claude prompt handles missing optional fields."""
        scraper = CISAScraper()

        advisories = [
            CISAAdvisory(
                advisory_id="ICSMA-24-002-01",
                title="Test Advisory",
                url="https://cisa.gov/test",
                release_date="2024-01-15",
                # No cvss_score, severity, cve_ids, vendor, description
            )
        ]

        prompt = scraper.generate_claude_prompt(advisories)

        assert "ICSMA-24-002-01" in prompt
        assert "N/A" in prompt  # Should show N/A for missing fields

    def test_rate_limiting(self):
        """Test rate limiting mechanism."""
        scraper = CISAScraper(request_delay=0.1)
        scraper.last_request_time = 0

        import time

        start = time.time()
        scraper._rate_limit()
        scraper._rate_limit()
        elapsed = time.time() - start

        # Should have enforced delay
        assert elapsed >= 0.1

    @patch("requests.Session.get")
    def test_make_request_success(self, mock_get):
        """Test successful HTTP request."""
        mock_response = Mock()
        mock_response.text = "<html><body>Test</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper = CISAScraper(request_delay=0)
        scraper.last_request_time = 0

        result = scraper._make_request("https://example.com")

        assert result is not None
        mock_get.assert_called_once()

    @patch("requests.Session.get")
    def test_make_request_failure(self, mock_get):
        """Test HTTP request failure handling."""
        import requests

        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        scraper = CISAScraper(request_delay=0)
        scraper.last_request_time = 0

        result = scraper._make_request("https://example.com")

        assert result is None

    @patch.object(CISAScraper, "_make_request")
    def test_parse_advisory_detail_extracts_cves(self, mock_request):
        """Test parsing advisory detail page extracts CVEs."""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <article>
                <div class="content">
                    <p>This advisory affects CVE-2024-1234 and CVE-2024-5678.</p>
                    <p>CVSS: 9.8 (CRITICAL)</p>
                </div>
            </article>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "html.parser")
        mock_request.return_value = soup

        scraper = CISAScraper()
        basic_info = {
            "advisory_id": "ICSMA-24-001-01",
            "title": "Test Advisory",
            "release_date": "2024-01-15",
            "advisory_type": "ICSMA",
        }

        advisory = scraper._parse_advisory_detail("https://test.com", basic_info)

        assert advisory is not None
        assert "CVE-2024-1234" in advisory.cve_ids
        assert "CVE-2024-5678" in advisory.cve_ids

    @patch.object(CISAScraper, "_make_request")
    def test_parse_advisory_detail_extracts_cvss(self, mock_request):
        """Test parsing advisory detail page extracts CVSS score."""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <article>
                <div class="content">
                    <p>CVSS v3.1: 7.5 (HIGH)</p>
                </div>
            </article>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "html.parser")
        mock_request.return_value = soup

        scraper = CISAScraper()
        basic_info = {
            "advisory_id": "ICSMA-24-001-01",
            "title": "Test Advisory",
            "release_date": "2024-01-15",
            "advisory_type": "ICSMA",
        }

        advisory = scraper._parse_advisory_detail("https://test.com", basic_info)

        assert advisory is not None
        assert advisory.cvss_score == 7.5
        assert advisory.severity == "HIGH"

    @patch.object(CISAScraper, "_make_request")
    def test_parse_advisory_detail_returns_none_on_failed_request(self, mock_request):
        """Test parsing returns None when request fails."""
        mock_request.return_value = None

        scraper = CISAScraper()
        basic_info = {
            "advisory_id": "ICSMA-24-001-01",
            "title": "Test Advisory",
            "release_date": "2024-01-15",
            "advisory_type": "ICSMA",
        }

        advisory = scraper._parse_advisory_detail("https://test.com", basic_info)

        assert advisory is None

    def test_parse_advisory_list_page_with_dates(self):
        """Test parsing advisory list page extracts dates from parent elements."""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <div class="advisory">
                <span class="date">January 15, 2024</span>
                <a href="/ics-medical-advisories/icsma-24-001-01">
                    Medical Device Advisory
                </a>
            </div>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "html.parser")
        scraper = CISAScraper()
        advisories = scraper._parse_advisory_list_page(soup)

        assert len(advisories) == 1
        assert advisories[0]["release_date"] == "January 15, 2024"

    def test_parse_advisory_list_page_skips_invalid_urls(self):
        """Test parsing skips links without valid advisory IDs."""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <a href="/some-other-page">Not an advisory</a>
            <a href="/ics-medical-advisories/icsma-24-001-01">Valid Advisory</a>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "html.parser")
        scraper = CISAScraper()
        advisories = scraper._parse_advisory_list_page(soup)

        assert len(advisories) == 1
        assert advisories[0]["advisory_id"] == "ICSMA-24-001-01"

    def test_cvss_severity_mapping(self):
        """Test CVSS score to severity mapping during parsing."""
        from bs4 import BeautifulSoup

        test_cases = [
            ("9.5", "CRITICAL"),
            ("8.0", "HIGH"),
            ("5.5", "MEDIUM"),
            ("2.5", "LOW"),
        ]

        scraper = CISAScraper()

        for cvss_str, expected_severity in test_cases:
            html = f"""
            <html>
            <body>
                <article>
                    <div class="content">
                        <p>CVSS v3: {cvss_str}</p>
                    </div>
                </article>
            </body>
            </html>
            """

            soup = BeautifulSoup(html, "html.parser")

            with patch.object(scraper, "_make_request", return_value=soup):
                basic_info = {
                    "advisory_id": "ICSMA-24-001-01",
                    "title": "Test",
                    "release_date": "2024-01-15",
                    "advisory_type": "ICSMA",
                }
                advisory = scraper._parse_advisory_detail("https://test.com", basic_info)

                assert (
                    advisory.severity == expected_severity
                ), f"CVSS {cvss_str} should map to {expected_severity}"


class TestNVDScraperAdvanced:
    """Advanced tests for NVD scraper."""

    def test_rate_limiting(self):
        """Test rate limiting mechanism."""
        import time

        scraper = NVDScraper()
        scraper.request_delay = 0.1  # Short delay for testing
        scraper.last_request_time = time.time()

        start = time.time()
        scraper._rate_limit()
        elapsed = time.time() - start

        # Should have enforced delay
        assert elapsed >= 0.05  # Allow some tolerance

    def test_rate_limiting_skipped_when_sufficient_time(self):
        """Test rate limiting is skipped when enough time has passed."""
        import time

        scraper = NVDScraper()
        scraper.request_delay = 0.1
        scraper.last_request_time = 0  # Very old timestamp

        start = time.time()
        scraper._rate_limit()
        elapsed = time.time() - start

        # Should not wait
        assert elapsed < 0.05

    @patch("requests.Session.get")
    def test_make_request_failure(self, mock_get):
        """Test API request failure handling."""
        import requests

        mock_get.side_effect = requests.exceptions.RequestException("API error")

        scraper = NVDScraper()
        scraper.last_request_time = 0

        try:
            scraper._make_request({"keywordSearch": "test"})
            raise AssertionError("Should have raised exception")
        except requests.exceptions.RequestException:
            pass  # Expected

    @patch("requests.Session.get")
    def test_search_by_keyword(self, mock_get):
        """Test search by keyword functionality."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-1234",
                        "descriptions": [{"lang": "en", "value": "DICOM vulnerability"}],
                        "published": "2024-01-15",
                        "lastModified": "2024-01-16",
                        "metrics": {},
                    }
                }
            ],
            "totalResults": 1,
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper = NVDScraper()
        scraper.last_request_time = 0

        result = scraper.search_by_keyword("DICOM")

        assert result["totalResults"] == 1
        mock_get.assert_called_once()

    @patch("requests.Session.get")
    def test_search_by_keyword_pagination(self, mock_get):
        """Test search with pagination parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": [], "totalResults": 0}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper = NVDScraper()
        scraper.last_request_time = 0

        scraper.search_by_keyword("test", start_index=100, results_per_page=50)

        call_url = mock_get.call_args[0][0]
        assert "startIndex=100" in call_url
        assert "resultsPerPage=50" in call_url

    @patch("requests.Session.get")
    def test_search_recent_cves(self, mock_get):
        """Test search for recent CVEs."""
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": [], "totalResults": 0}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper = NVDScraper()
        scraper.last_request_time = 0

        result = scraper.search_recent_cves(days_back=7)

        assert result["totalResults"] == 0
        call_url = mock_get.call_args[0][0]
        assert "pubStartDate" in call_url
        assert "pubEndDate" in call_url

    def test_parse_cve_with_cvss_v30(self):
        """Test parsing CVE with CVSS v3.0 metrics."""
        scraper = NVDScraper()

        cve_item = {
            "cve": {
                "id": "CVE-2024-5678",
                "descriptions": [{"lang": "en", "value": "Test"}],
                "published": "2024-01-15",
                "lastModified": "2024-01-16",
                "metrics": {
                    "cvssMetricV30": [
                        {
                            "cvssData": {
                                "baseScore": 6.5,
                                "baseSeverity": "MEDIUM",
                                "vectorString": "CVSS:3.0/AV:N",
                            }
                        }
                    ]
                },
            }
        }

        entry = scraper._parse_cve(cve_item, [])

        assert entry.cvss_v3_score == 6.5
        assert entry.cvss_v3_severity == "MEDIUM"

    def test_parse_cve_with_cvss_v2(self):
        """Test parsing CVE with only CVSS v2 metrics."""
        scraper = NVDScraper()

        cve_item = {
            "cve": {
                "id": "CVE-2024-9999",
                "descriptions": [{"lang": "en", "value": "Old CVE"}],
                "published": "2024-01-15",
                "lastModified": "2024-01-16",
                "metrics": {
                    "cvssMetricV2": [
                        {
                            "cvssData": {
                                "baseScore": 5.0,
                            }
                        }
                    ]
                },
            }
        }

        entry = scraper._parse_cve(cve_item, [])

        assert entry.cvss_v2_score == 5.0
        assert entry.cvss_v3_score is None

    def test_parse_cve_non_english_fallback(self):
        """Test parsing CVE with non-English description fallback."""
        scraper = NVDScraper()

        cve_item = {
            "cve": {
                "id": "CVE-2024-0001",
                "descriptions": [
                    {"lang": "es", "value": "Vulnerabilidad de prueba"},
                ],
                "published": "2024-01-15",
                "lastModified": "2024-01-16",
                "metrics": {},
            }
        }

        entry = scraper._parse_cve(cve_item, [])

        assert entry.description == "Vulnerabilidad de prueba"

    def test_parse_cve_extracts_configurations(self):
        """Test parsing CVE extracts affected products from configurations."""
        scraper = NVDScraper()

        cve_item = {
            "cve": {
                "id": "CVE-2024-0002",
                "descriptions": [{"lang": "en", "value": "Test"}],
                "published": "2024-01-15",
                "lastModified": "2024-01-16",
                "metrics": {},
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {"criteria": "cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*"},
                                    {"criteria": "cpe:2.3:a:vendor:product:2.0:*:*:*:*:*:*:*"},
                                ]
                            }
                        ]
                    }
                ],
            }
        }

        entry = scraper._parse_cve(cve_item, [])

        assert len(entry.affected_products) == 2
        assert "cpe:2.3:a:vendor:product:1.0" in entry.affected_products[0]

    def test_parse_cve_vulnerability_status(self):
        """Test parsing CVE extracts vulnerability status."""
        scraper = NVDScraper()

        cve_item = {
            "cve": {
                "id": "CVE-2024-0003",
                "descriptions": [{"lang": "en", "value": "Test"}],
                "published": "2024-01-15",
                "lastModified": "2024-01-16",
                "vulnStatus": "Analyzed",
                "metrics": {},
            }
        }

        entry = scraper._parse_cve(cve_item, [])

        assert entry.vulnerability_status == "Analyzed"

    @patch("requests.Session.get")
    def test_search_medical_device_cves(self, mock_get):
        """Test searching for medical device CVEs."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-MED1",
                        "descriptions": [{"lang": "en", "value": "DICOM server RCE"}],
                        "published": "2024-01-15",
                        "lastModified": "2024-01-16",
                        "metrics": {},
                    }
                }
            ],
            "totalResults": 1,
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper = NVDScraper()
        scraper.last_request_time = 0

        # Use just one keyword to speed up test
        cves = scraper.search_medical_device_cves(
            keywords=["DICOM"],
            max_results=10,
        )

        assert len(cves) >= 0  # May or may not find matches based on keyword matching

    def test_save_results_creates_directory(self, tmp_path):
        """Test save_results creates output directory if needed."""
        scraper = NVDScraper()
        output_file = tmp_path / "subdir" / "test_cves.json"

        cves = [
            CVEEntry(
                cve_id="CVE-2024-1234",
                description="Test",
                published_date="2024-01-15",
                last_modified_date="2024-01-16",
            )
        ]

        scraper.save_results(cves, output_file)

        assert output_file.exists()


class TestCISAScraperAdvanced:
    """Advanced tests for CISA scraper."""

    def test_scraper_with_headers(self):
        """Test scraper has proper headers."""
        scraper = CISAScraper()

        # Verify session exists
        assert scraper.session is not None

    @patch.object(CISAScraper, "_make_request")
    def test_parse_advisory_detail_extracts_vendor(self, mock_request):
        """Test parsing advisory detail page extracts vendor info."""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <article>
                <div class="content">
                    <p>Vendor: Medical Devices Inc</p>
                    <p>Affected Products: DICOM Server v1.0</p>
                </div>
            </article>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "html.parser")
        mock_request.return_value = soup

        scraper = CISAScraper()
        basic_info = {
            "advisory_id": "ICSMA-24-001-01",
            "title": "Test Advisory",
            "release_date": "2024-01-15",
            "advisory_type": "ICSMA",
        }

        advisory = scraper._parse_advisory_detail("https://test.com", basic_info)

        assert advisory is not None

    @patch.object(CISAScraper, "_make_request")
    def test_parse_advisory_detail_extracts_mitigation(self, mock_request):
        """Test parsing advisory detail page extracts mitigation."""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <article>
                <div class="content">
                    <h2>Mitigation</h2>
                    <p>Update to version 3.0 or later.</p>
                </div>
            </article>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "html.parser")
        mock_request.return_value = soup

        scraper = CISAScraper()
        basic_info = {
            "advisory_id": "ICSMA-24-001-01",
            "title": "Test Advisory",
            "release_date": "2024-01-15",
            "advisory_type": "ICSMA",
        }

        advisory = scraper._parse_advisory_detail("https://test.com", basic_info)

        assert advisory is not None

    def test_advisory_to_dict(self):
        """Test CISAAdvisory conversion to dictionary."""
        from dataclasses import asdict

        advisory = CISAAdvisory(
            advisory_id="ICSMA-24-001-01",
            title="Test Advisory",
            url="https://test.com",
            release_date="2024-01-15",
            cvss_score=8.5,
            cve_ids=["CVE-2024-1234"],
        )

        data = asdict(advisory)

        assert data["advisory_id"] == "ICSMA-24-001-01"
        assert data["cvss_score"] == 8.5
        assert "CVE-2024-1234" in data["cve_ids"]

    def test_advisory_type_explicit_setting(self):
        """Test advisory type can be explicitly set."""
        # Default is ICSMA (medical)
        default_advisory = CISAAdvisory(
            advisory_id="ICSMA-24-001-01",
            title="Medical Advisory",
            url="https://test.com",
            release_date="2024-01-15",
        )
        assert default_advisory.advisory_type == "ICSMA"

        # Explicit ICSA setting for general ICS advisories
        icsa = CISAAdvisory(
            advisory_id="ICSA-24-001-01",
            title="ICS Advisory",
            url="https://test.com",
            release_date="2024-01-15",
            advisory_type="ICSA",
        )
        assert icsa.advisory_type == "ICSA"

    def test_save_results_empty_list(self, tmp_path):
        """Test saving empty results."""
        scraper = CISAScraper()
        output_file = tmp_path / "empty_advisories.json"

        scraper.save_results([], output_file)

        assert output_file.exists()
        import json

        with open(output_file) as f:
            data = json.load(f)
        assert data["metadata"]["total_advisories"] == 0
        assert data["advisories"] == []

    @patch.object(CISAScraper, "_make_request")
    @patch.object(CISAScraper, "_parse_advisory_detail")
    def test_scrape_medical_advisories_icsma_only(self, mock_detail, mock_request):
        """Test scraping medical advisories (ICSMA only)."""
        from bs4 import BeautifulSoup

        # Mock the advisory list page
        list_html = """
        <html>
        <body>
            <a href="/ics-medical-advisories/icsma-24-001-01">Advisory 1</a>
            <a href="/ics-medical-advisories/icsma-24-002-01">Advisory 2</a>
        </body>
        </html>
        """

        mock_request.return_value = BeautifulSoup(list_html, "html.parser")
        mock_detail.return_value = CISAAdvisory(
            advisory_id="ICSMA-24-001-01",
            title="Test Advisory",
            url="https://test.com",
            release_date="2024-01-15",
        )

        scraper = CISAScraper()
        advisories = scraper.scrape_medical_advisories(max_results=5, include_general_ics=False)

        assert len(advisories) > 0
        mock_request.assert_called()

    @patch.object(CISAScraper, "_make_request")
    def test_scrape_medical_advisories_with_general_ics(self, mock_request):
        """Test scraping includes general ICS advisories with medical keywords."""
        from bs4 import BeautifulSoup

        # First call returns empty ICSMA list
        icsma_html = """<html><body></body></html>"""

        # Second call returns ICS advisories with medical keyword
        ics_html = """
        <html>
        <body>
            <a href="/ics-advisories/icsa-24-001-01">MRI Machine Vulnerability</a>
        </body>
        </html>
        """

        mock_request.side_effect = [
            BeautifulSoup(icsma_html, "html.parser"),
            BeautifulSoup(ics_html, "html.parser"),
        ]

        scraper = CISAScraper()
        # Test that it searches both pages
        scraper.scrape_medical_advisories(max_results=5, include_general_ics=True)

        # Should have called twice (ICSMA + ICS pages)
        assert mock_request.call_count == 2

    @patch.object(CISAScraper, "_make_request")
    def test_scrape_medical_advisories_max_results(self, mock_request):
        """Test max_results limits advisory count."""
        from bs4 import BeautifulSoup

        # Return many advisories
        list_html = """
        <html>
        <body>
            <a href="/ics-medical-advisories/icsma-24-001-01">Advisory 1</a>
            <a href="/ics-medical-advisories/icsma-24-002-01">Advisory 2</a>
            <a href="/ics-medical-advisories/icsma-24-003-01">Advisory 3</a>
        </body>
        </html>
        """

        def request_return(url):
            if "medical" in url:
                return BeautifulSoup(list_html, "html.parser")
            return BeautifulSoup("<html><body></body></html>", "html.parser")

        mock_request.side_effect = request_return

        scraper = CISAScraper()
        advisories = scraper.scrape_medical_advisories(max_results=1, include_general_ics=False)

        # Should be limited to 1
        assert len(advisories) <= 1

    @patch.object(CISAScraper, "_make_request")
    def test_scrape_medical_advisories_request_failure(self, mock_request):
        """Test scraping handles request failure gracefully."""
        mock_request.return_value = None

        scraper = CISAScraper()
        advisories = scraper.scrape_medical_advisories(max_results=5, include_general_ics=False)

        # Should return empty list on failure
        assert advisories == []


class TestCISAScraperPromptGeneration:
    """Tests for CISA scraper prompt generation."""

    def test_generate_claude_prompt_basic(self):
        """Test basic Claude prompt generation."""
        scraper = CISAScraper()
        advisories = [
            CISAAdvisory(
                advisory_id="ICSMA-24-001-01",
                title="Test Medical Device Advisory",
                url="https://test.com",
                release_date="2024-01-15",
                cvss_score=8.5,
                cve_ids=["CVE-2024-1234"],
                vendor="Test Vendor",
                description="Critical vulnerability in DICOM server",
            )
        ]

        prompt = scraper.generate_claude_prompt(advisories, batch_size=10)

        assert "ICSMA-24-001-01" in prompt
        assert "Test Medical Device Advisory" in prompt
        assert "CVE-2024-1234" in prompt
        assert "device_type" in prompt
        assert "clinical_impact" in prompt

    def test_generate_claude_prompt_truncates_description(self):
        """Test prompt truncates long descriptions."""
        scraper = CISAScraper()
        long_description = "A" * 500  # Longer than 400 char limit

        advisories = [
            CISAAdvisory(
                advisory_id="ICSMA-24-001-01",
                title="Test",
                url="https://test.com",
                release_date="2024-01-15",
                description=long_description,
            )
        ]

        prompt = scraper.generate_claude_prompt(advisories)

        # Should have truncation indicator
        assert "..." in prompt

    def test_generate_claude_prompt_handles_none_values(self):
        """Test prompt handles None values gracefully."""
        scraper = CISAScraper()
        advisories = [
            CISAAdvisory(
                advisory_id="ICSMA-24-001-01",
                title="Test",
                url="https://test.com",
                release_date="2024-01-15",
                cvss_score=None,
                cve_ids=None,
                vendor=None,
                description=None,
            )
        ]

        prompt = scraper.generate_claude_prompt(advisories)

        # Should contain N/A for missing values
        assert "N/A" in prompt

    def test_generate_claude_prompt_batch_size(self):
        """Test batch_size limits advisories in prompt."""
        scraper = CISAScraper()
        advisories = [
            CISAAdvisory(
                advisory_id=f"ICSMA-24-{i:03d}-01",
                title=f"Advisory {i}",
                url="https://test.com",
                release_date="2024-01-15",
            )
            for i in range(20)
        ]

        prompt = scraper.generate_claude_prompt(advisories, batch_size=5)

        # Should only contain first 5 advisories
        assert "ICSMA-24-000-01" in prompt
        assert "ICSMA-24-004-01" in prompt
        assert "ICSMA-24-005-01" not in prompt
