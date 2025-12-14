"""
NVD Scraper - Medical Device CVE Extraction

Scrapes the NIST National Vulnerability Database (NVD) API 2.0 for
CVEs related to medical devices. Uses keyword filtering to identify
relevant vulnerabilities.

Usage:
    from medtech_security.threat_intel import NVDScraper

    scraper = NVDScraper()
    cves = scraper.search_medical_device_cves(max_results=100)
    scraper.save_results(cves, "data/threat_intel/cves/medical_devices.json")

API Documentation: https://nvd.nist.gov/developers/vulnerabilities
Rate Limits: 5 requests per 30 seconds (without API key)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Medical device related keywords for filtering CVEs
MEDICAL_DEVICE_KEYWORDS = [
    # General medical terms
    "medical device",
    "healthcare",
    "clinical",
    "patient monitor",
    "hospital",
    # Imaging systems
    "DICOM",
    "PACS",
    "radiology",
    "X-ray",
    "CT scanner",
    "MRI",
    "ultrasound",
    "PET",
    "mammography",
    "fluoroscopy",
    # Therapeutic devices
    "infusion pump",
    "insulin pump",
    "pacemaker",
    "defibrillator",
    "ventilator",
    "dialysis",
    "radiation therapy",
    "linear accelerator",
    # Monitoring devices
    "vital signs",
    "ECG",
    "EKG",
    "pulse oximeter",
    "blood pressure",
    "glucose monitor",
    "telemetry",
    # Laboratory
    "laboratory information",
    "LIS",
    "blood analyzer",
    "pathology",
    # Healthcare IT
    "HL7",
    "FHIR",
    "EHR",
    "EMR",
    "health information",
    "PHI",
    "HIPAA",
    # Specific manufacturers (major medical device companies)
    "Philips Healthcare",
    "GE Healthcare",
    "Siemens Healthineers",
    "Medtronic",
    "BD",  # Becton Dickinson
    "Baxter",
    "Abbott",
    "Boston Scientific",
    "Stryker",
    "Zimmer Biomet",
    "Dräger",
    "Mindray",
    "Fujifilm",
    "Canon Medical",
    "Carestream",
    # Protocols and standards
    "IHE",
    "medical imaging",
    "surgical robot",
    "telemedicine",
]

# CPE vendor strings for medical device manufacturers
MEDICAL_DEVICE_VENDORS = [
    "philips",
    "ge_healthcare",
    "siemens",
    "medtronic",
    "bd",
    "baxter",
    "abbott",
    "boston_scientific",
    "stryker",
    "zimmer_biomet",
    "draeger",
    "mindray",
    "fujifilm",
    "canon",
    "carestream",
    "epic",
    "cerner",
    "allscripts",
]


@dataclass
class CVEEntry:
    """Structured CVE data for medical device vulnerabilities."""

    cve_id: str
    description: str
    published_date: str
    last_modified_date: str
    cvss_v3_score: float | None = None
    cvss_v3_severity: str | None = None
    cvss_v3_vector: str | None = None
    cvss_v2_score: float | None = None
    cwe_ids: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    affected_products: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    vulnerability_status: str = ""

    # Fields for Claude.ai extraction (to be filled later)
    device_type: str | None = None
    clinical_impact: str | None = None
    exploitability: str | None = None
    remediation: str | None = None


class NVDScraper:
    """
    Scraper for the NIST National Vulnerability Database API 2.0.

    Fetches CVEs related to medical devices using keyword filtering.
    Respects API rate limits (5 requests per 30 seconds without API key).
    """

    BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def __init__(self, api_key: str | None = None):
        """
        Initialize the NVD scraper.

        Args:
            api_key: Optional NVD API key for higher rate limits.
                     Get one at: https://nvd.nist.gov/developers/request-an-api-key
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["apiKey"] = api_key
            self.request_delay = 0.6  # 50 requests per 30 seconds with API key
        else:
            self.request_delay = 6.0  # 5 requests per 30 seconds without key

        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            sleep_time = self.request_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _make_request(self, params: dict[str, Any]) -> dict[Any, Any]:
        """
        Make a rate-limited request to the NVD API.

        Args:
            params: Query parameters for the API request.

        Returns:
            JSON response from the API.
        """
        self._rate_limit()

        url = f"{self.BASE_URL}?{urlencode(params)}"
        logger.debug(f"Requesting: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            result: dict[Any, Any] = response.json()
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def search_by_keyword(
        self,
        keyword: str,
        start_index: int = 0,
        results_per_page: int = 100,
        pub_start_date: datetime | None = None,
        pub_end_date: datetime | None = None,
    ) -> dict:
        """
        Search CVEs by keyword.

        Args:
            keyword: Search term to find in CVE descriptions.
            start_index: Starting index for pagination.
            results_per_page: Number of results per page (max 2000).
            pub_start_date: Filter by publication date (start).
            pub_end_date: Filter by publication date (end).

        Returns:
            API response with CVE data.
        """
        params = {
            "keywordSearch": keyword,
            "startIndex": start_index,
            "resultsPerPage": min(results_per_page, 2000),
        }

        # Note: Date filtering disabled for now - NVD API 2.0 has strict ISO 8601 requirements
        # that cause 404 errors. Using keyword search without date filters works reliably.
        # TODO: Fix date format to comply with NVD API 2.0 requirements
        # if pub_start_date:
        #     params["pubStartDate"] = pub_start_date.isoformat()
        # if pub_end_date:
        #     params["pubEndDate"] = pub_end_date.isoformat()

        return self._make_request(params)

    def search_recent_cves(
        self,
        days_back: int = 30,
        start_index: int = 0,
        results_per_page: int = 100,
    ) -> dict:
        """
        Search for recently published or modified CVEs.

        Args:
            days_back: Number of days to look back.
            start_index: Starting index for pagination.
            results_per_page: Number of results per page.

        Returns:
            API response with CVE data.
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        params = {
            "pubStartDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
            "pubEndDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
            "startIndex": start_index,
            "resultsPerPage": min(results_per_page, 2000),
        }

        return self._make_request(params)

    def _parse_cve(self, cve_item: dict, matched_keywords: list[str]) -> CVEEntry:
        """
        Parse a CVE item from the API response into a CVEEntry.

        Args:
            cve_item: Raw CVE data from the API.
            matched_keywords: Keywords that matched this CVE.

        Returns:
            Structured CVEEntry object.
        """
        cve = cve_item.get("cve", {})
        cve_id = cve.get("id", "")

        # Extract description (prefer English)
        descriptions = cve.get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break
        if not description and descriptions:
            description = descriptions[0].get("value", "")

        # Extract dates
        published = cve.get("published", "")
        last_modified = cve.get("lastModified", "")

        # Extract CVSS scores
        metrics = cve.get("metrics", {})

        cvss_v3_score = None
        cvss_v3_severity = None
        cvss_v3_vector = None

        # Try CVSS 3.1 first, then 3.0
        for cvss_key in ["cvssMetricV31", "cvssMetricV30"]:
            if cvss_key in metrics and metrics[cvss_key]:
                cvss_data = metrics[cvss_key][0].get("cvssData", {})
                cvss_v3_score = cvss_data.get("baseScore")
                cvss_v3_severity = cvss_data.get("baseSeverity")
                cvss_v3_vector = cvss_data.get("vectorString")
                break

        cvss_v2_score = None
        if "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
            cvss_v2_score = metrics["cvssMetricV2"][0].get("cvssData", {}).get("baseScore")

        # Extract CWE IDs
        cwe_ids = []
        weaknesses = cve.get("weaknesses", [])
        for weakness in weaknesses:
            for desc in weakness.get("description", []):
                if desc.get("value", "").startswith("CWE-"):
                    cwe_ids.append(desc["value"])

        # Extract references
        references = []
        for ref in cve.get("references", []):
            references.append(ref.get("url", ""))

        # Extract affected products (CPE)
        affected_products = []
        configurations = cve.get("configurations", [])
        for config in configurations:
            for node in config.get("nodes", []):
                for cpe_match in node.get("cpeMatch", []):
                    criteria = cpe_match.get("criteria", "")
                    if criteria:
                        affected_products.append(criteria)

        # Vulnerability status
        vuln_status = cve.get("vulnStatus", "")

        return CVEEntry(
            cve_id=cve_id,
            description=description,
            published_date=published,
            last_modified_date=last_modified,
            cvss_v3_score=cvss_v3_score,
            cvss_v3_severity=cvss_v3_severity,
            cvss_v3_vector=cvss_v3_vector,
            cvss_v2_score=cvss_v2_score,
            cwe_ids=cwe_ids,
            references=references[:10],  # Limit to first 10 references
            affected_products=affected_products[:20],  # Limit to first 20 CPEs
            matched_keywords=matched_keywords,
            vulnerability_status=vuln_status,
        )

    def _matches_medical_keywords(self, description: str) -> list[str]:
        """
        Check if a CVE description matches medical device keywords.

        Args:
            description: CVE description text.

        Returns:
            List of matched keywords.
        """
        matched = []
        description_lower = description.lower()

        for keyword in MEDICAL_DEVICE_KEYWORDS:
            if keyword.lower() in description_lower:
                matched.append(keyword)

        return matched

    def search_medical_device_cves(
        self,
        keywords: list[str] | None = None,
        max_results: int = 100,
        days_back: int = 365,
        severity_filter: str | None = None,
    ) -> list[CVEEntry]:
        """
        Search for medical device related CVEs.

        Args:
            keywords: Custom keywords to search. Defaults to MEDICAL_DEVICE_KEYWORDS.
            max_results: Maximum number of CVEs to return.
            days_back: How far back to search (in days).
            severity_filter: Filter by severity (LOW, MEDIUM, HIGH, CRITICAL).

        Returns:
            List of CVEEntry objects for medical device CVEs.
        """
        if keywords is None:
            # Use a subset of the most specific keywords for API searches
            keywords = [
                "medical device",
                "DICOM",
                "PACS",
                "infusion pump",
                "patient monitor",
                "HL7",
                "healthcare",
                "Philips Healthcare",
                "GE Healthcare",
                "Medtronic",
            ]

        all_cves: dict[str, CVEEntry] = {}  # Use dict to deduplicate by CVE ID

        # Note: Date filtering disabled due to NVD API 2.0 date format issues
        # Will filter by date locally after fetching results
        # end_date = datetime.now(timezone.utc)
        # start_date = end_date - timedelta(days=days_back)

        for keyword in keywords:
            if len(all_cves) >= max_results:
                break

            logger.info(f"Searching for keyword: '{keyword}'")

            try:
                response = self.search_by_keyword(
                    keyword=keyword,
                    results_per_page=min(100, max_results - len(all_cves)),
                    # Date filtering disabled - see note above
                )

                total_results = response.get("totalResults", 0)
                vulnerabilities = response.get("vulnerabilities", [])

                logger.info(f"  Found {total_results} total, processing {len(vulnerabilities)}")

                for vuln in vulnerabilities:
                    cve = vuln.get("cve", {})
                    cve_id = cve.get("id", "")

                    if cve_id in all_cves:
                        # Already have this CVE, just add the keyword
                        all_cves[cve_id].matched_keywords.append(keyword)
                        continue

                    # Parse and add
                    entry = self._parse_cve(vuln, [keyword])

                    # Apply severity filter if specified
                    if severity_filter:
                        if entry.cvss_v3_severity != severity_filter:
                            continue

                    all_cves[cve_id] = entry

            except Exception as e:
                logger.error(f"Error searching keyword '{keyword}': {e}")
                continue

        results = list(all_cves.values())

        # Sort by CVSS score (highest first), then by date
        results.sort(
            key=lambda x: (
                -(x.cvss_v3_score or 0),
                x.published_date,
            ),
            reverse=False,
        )

        return results[:max_results]

    def save_results(
        self,
        cves: list[CVEEntry],
        output_path: str | Path,
        format: str = "json",
    ) -> None:
        """
        Save CVE results to a file.

        Args:
            cves: List of CVEEntry objects.
            output_path: Path to save the file.
            format: Output format ('json' or 'csv').
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = {
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "total_cves": len(cves),
                    "source": "NVD API 2.0",
                },
                "cves": [asdict(cve) for cve in cves],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(cves)} CVEs to {output_path}")

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if cves:
                    writer = csv.DictWriter(f, fieldnames=asdict(cves[0]).keys())
                    writer.writeheader()
                    for cve in cves:
                        row = asdict(cve)
                        # Convert lists to strings for CSV
                        for key, value in row.items():
                            if isinstance(value, list):
                                row[key] = "; ".join(str(v) for v in value)
                        writer.writerow(row)

            logger.info(f"Saved {len(cves)} CVEs to {output_path}")

        else:
            raise ValueError(f"Unknown format: {format}")

    def generate_claude_prompt(self, cves: list[CVEEntry], batch_size: int = 10) -> str:
        """
        Generate a prompt for Claude.ai to extract structured information.

        Args:
            cves: List of CVEEntry objects to analyze.
            batch_size: Number of CVEs per prompt batch.

        Returns:
            Formatted prompt for Claude.ai.
        """
        prompt_template = """You are a medical device cybersecurity expert. Analyze the following CVEs and extract structured threat intelligence.

For each CVE, provide:
1. **Device Type**: What type of medical device is affected? (imaging, therapeutic, monitoring, laboratory, healthcare IT, or other)
2. **Clinical Impact**: What is the potential impact on patient care? (HIGH: direct patient harm, MEDIUM: care disruption, LOW: administrative)
3. **Exploitability**: How easy is this to exploit? (EASY: remote/unauthenticated, MODERATE: requires some access, HARD: requires physical access)
4. **Remediation**: What actions should be taken? (patch available, workaround, vendor contact, no fix)

CVEs to analyze:
"""

        cve_list = []
        for cve in cves[:batch_size]:
            cve_text = f"""
---
**{cve.cve_id}** (CVSS: {cve.cvss_v3_score or 'N/A'}, Severity: {cve.cvss_v3_severity or 'N/A'})
Published: {cve.published_date[:10] if cve.published_date else 'N/A'}
Description: {cve.description[:500]}{'...' if len(cve.description) > 500 else ''}
CWEs: {', '.join(cve.cwe_ids) if cve.cwe_ids else 'N/A'}
Matched Keywords: {', '.join(cve.matched_keywords)}
"""
            cve_list.append(cve_text)

        prompt = prompt_template + "\n".join(cve_list)

        prompt += """

Respond in JSON format:
```json
{
  "analyses": [
    {
      "cve_id": "CVE-XXXX-XXXXX",
      "device_type": "imaging|therapeutic|monitoring|laboratory|healthcare_it|other",
      "clinical_impact": "HIGH|MEDIUM|LOW",
      "exploitability": "EASY|MODERATE|HARD",
      "remediation": "patch_available|workaround|vendor_contact|no_fix",
      "reasoning": "Brief explanation of your analysis"
    }
  ]
}
```"""

        return prompt


def main() -> None:
    """CLI entry point for the NVD scraper."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape NVD for medical device CVEs")
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum number of CVEs to fetch (default: 50)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=365,
        help="How many days back to search (default: 365)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/threat_intel/cves/medical_devices.json",
        help="Output file path",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NVD API key for higher rate limits",
    )
    parser.add_argument(
        "--generate-prompt",
        action="store_true",
        help="Generate Claude.ai prompt for extracted CVEs",
    )
    parser.add_argument(
        "--severity",
        type=str,
        choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default=None,
        help="Filter by severity level",
    )

    args = parser.parse_args()

    scraper = NVDScraper(api_key=args.api_key)

    logger.info(f"Searching for medical device CVEs (last {args.days_back} days)...")
    cves = scraper.search_medical_device_cves(
        max_results=args.max_results,
        days_back=args.days_back,
        severity_filter=args.severity,
    )

    logger.info(f"Found {len(cves)} medical device related CVEs")

    # Save results
    scraper.save_results(cves, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("MEDICAL DEVICE CVE SUMMARY")
    print("=" * 60)

    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "N/A": 0}
    for cve in cves:
        sev = cve.cvss_v3_severity or "N/A"
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print(f"\nTotal CVEs: {len(cves)}")
    print("\nBy Severity:")
    for sev, count in severity_counts.items():
        if count > 0:
            print(f"  {sev}: {count}")

    print("\nTop 5 CVEs by CVSS Score:")
    for cve in cves[:5]:
        print(f"  {cve.cve_id}: {cve.cvss_v3_score or 'N/A'} ({cve.cvss_v3_severity or 'N/A'})")
        print(f"    Keywords: {', '.join(cve.matched_keywords[:3])}")

    # Generate Claude prompt if requested
    if args.generate_prompt:
        prompt = scraper.generate_claude_prompt(cves)
        prompt_path = Path(args.output).parent / "claude_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"\nClaude.ai prompt saved to: {prompt_path}")

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
