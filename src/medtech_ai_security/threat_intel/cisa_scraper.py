"""
CISA ICS-CERT Advisory Scraper - Medical Device Security Advisories

Scrapes CISA (formerly ICS-CERT) advisories for medical device
security vulnerabilities. CISA publishes ICS Medical Advisories (ICSMA)
specifically for healthcare and medical device security issues.

Sources:
- https://www.cisa.gov/news-events/ics-medical-advisories
- https://www.cisa.gov/news-events/ics-advisories

Note: CISA does not have a public API. This scraper parses the public
advisory listing pages and individual advisory pages.

Usage:
    from medtech_ai_security.threat_intel import CISAScraper

    scraper = CISAScraper()
    advisories = scraper.scrape_medical_advisories(max_results=50)
    scraper.save_results(advisories, "data/threat_intel/advisories/cisa_medical.json")
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CISAAdvisory:
    """Structured CISA advisory data."""

    advisory_id: str
    title: str
    url: str
    release_date: str
    last_updated: str | None = None
    severity: str | None = None
    cvss_score: float | None = None
    affected_products: list[str] = field(default_factory=list)
    cve_ids: list[str] = field(default_factory=list)
    vendor: str | None = None
    description: str | None = None
    mitigation: str | None = None
    advisory_type: str = "ICSMA"  # ICSMA for medical, ICSA for general ICS

    # Fields for Claude.ai extraction
    device_type: str | None = None
    clinical_impact: str | None = None
    exploitability: str | None = None


class CISAScraper:
    """
    Scraper for CISA ICS-CERT advisories.

    Focuses on medical device advisories (ICSMA prefix) but can also
    scrape general ICS advisories (ICSA prefix) with medical keywords.
    """

    # CISA advisory listing pages
    MEDICAL_ADVISORIES_URL = "https://www.cisa.gov/news-events/ics-medical-advisories"
    ICS_ADVISORIES_URL = "https://www.cisa.gov/news-events/ics-advisories"
    BASE_URL = "https://www.cisa.gov"

    # Medical device keywords for filtering general ICS advisories
    MEDICAL_KEYWORDS = [
        "medical",
        "healthcare",
        "hospital",
        "patient",
        "clinical",
        "health",
        "diagnostic",
        "imaging",
        "DICOM",
        "PACS",
        "infusion",
        "ventilator",
        "monitor",
        "therapeutic",
        "surgical",
        "laboratory",
        "pharmacy",
        "EHR",
        "EMR",
    ]

    def __init__(self, request_delay: float = 2.0):
        """
        Initialize the CISA scraper.

        Args:
            request_delay: Delay between requests in seconds (be respectful).
        """
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MedTech-AI-Security/1.0 (Medical Device Security Research)"
        })
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str) -> BeautifulSoup | None:
        """
        Make a rate-limited request and parse HTML.

        Args:
            url: URL to fetch.

        Returns:
            BeautifulSoup object or None on error.
        """
        self._rate_limit()

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def _parse_advisory_list_page(self, soup: BeautifulSoup) -> list[dict]:
        """
        Parse advisory listing page to extract basic info.

        Args:
            soup: BeautifulSoup object of listing page.

        Returns:
            List of advisory dicts with id, title, url, date.
        """
        advisories = []

        # CISA uses various list structures - try to find advisory links
        # Look for links containing ICSMA or ICSA in the URL
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")

            # Match advisory URLs
            if "/ics-medical-advisories/icsma-" in href or "/ics-advisories/icsa-" in href:
                title = link.get_text(strip=True)

                # Extract advisory ID from URL
                match = re.search(r"(icsma-[\d-]+|icsa-[\d-]+)", href, re.I)
                if match:
                    advisory_id = match.group(1).upper()
                else:
                    continue

                # Build full URL
                full_url = urljoin(self.BASE_URL, href)

                # Try to find associated date
                date_str = ""
                parent = link.find_parent(["div", "li", "article", "tr"])
                if parent:
                    date_elem = parent.find(["time", "span"], class_=re.compile(r"date|time", re.I))
                    if date_elem:
                        date_str = date_elem.get_text(strip=True)

                advisories.append({
                    "advisory_id": advisory_id,
                    "title": title,
                    "url": full_url,
                    "release_date": date_str,
                    "advisory_type": "ICSMA" if "icsma" in advisory_id.lower() else "ICSA",
                })

        return advisories

    def _parse_advisory_detail(self, url: str, basic_info: dict) -> CISAAdvisory | None:
        """
        Parse individual advisory page for detailed information.

        Args:
            url: Advisory page URL.
            basic_info: Basic info from listing page.

        Returns:
            CISAAdvisory object or None on error.
        """
        soup = self._make_request(url)
        if not soup:
            return None

        # Extract content
        content = soup.find("article") or soup.find("main") or soup

        # Get full description
        description = ""
        desc_section = content.find(["div", "section"], class_=re.compile(r"content|body", re.I))
        if desc_section:
            # Get first few paragraphs
            paragraphs = desc_section.find_all("p", limit=5)
            description = " ".join(p.get_text(strip=True) for p in paragraphs)

        # Extract CVE IDs
        cve_ids = []
        cve_pattern = re.compile(r"CVE-\d{4}-\d{4,}")
        for text in content.stripped_strings:
            matches = cve_pattern.findall(text)
            cve_ids.extend(matches)
        cve_ids = list(set(cve_ids))  # Deduplicate

        # Extract CVSS score
        cvss_score = None
        cvss_pattern = re.compile(r"CVSS[:\s]+v?3\.?[01]?[:\s]+(\d+\.?\d*)", re.I)
        for text in content.stripped_strings:
            match = cvss_pattern.search(text)
            if match:
                try:
                    cvss_score = float(match.group(1))
                    break
                except ValueError:
                    pass

        # Extract severity
        severity = None
        if cvss_score:
            if cvss_score >= 9.0:
                severity = "CRITICAL"
            elif cvss_score >= 7.0:
                severity = "HIGH"
            elif cvss_score >= 4.0:
                severity = "MEDIUM"
            else:
                severity = "LOW"

        # Extract vendor
        vendor = None
        vendor_section = content.find(string=re.compile(r"vendor|manufacturer", re.I))
        if vendor_section:
            parent = vendor_section.find_parent()
            if parent:
                vendor = parent.get_text(strip=True)[:100]

        # Extract affected products
        affected_products = []
        affected_section = content.find(string=re.compile(r"affected products?", re.I))
        if affected_section:
            parent = affected_section.find_parent()
            if parent:
                next_list = parent.find_next(["ul", "ol"])
                if next_list:
                    for li in next_list.find_all("li", limit=10):
                        affected_products.append(li.get_text(strip=True))

        # Extract mitigation
        mitigation = None
        mitigation_section = content.find(string=re.compile(r"mitigation|recommendation", re.I))
        if mitigation_section:
            parent = mitigation_section.find_parent()
            if parent:
                next_elem = parent.find_next(["p", "ul"])
                if next_elem:
                    mitigation = next_elem.get_text(strip=True)[:500]

        return CISAAdvisory(
            advisory_id=basic_info["advisory_id"],
            title=basic_info["title"],
            url=url,
            release_date=basic_info.get("release_date", ""),
            severity=severity,
            cvss_score=cvss_score,
            affected_products=affected_products,
            cve_ids=cve_ids,
            vendor=vendor,
            description=description[:1000] if description else None,
            mitigation=mitigation,
            advisory_type=basic_info.get("advisory_type", "ICSMA"),
        )

    def scrape_medical_advisories(
        self,
        max_results: int = 50,
        include_general_ics: bool = True,
    ) -> list[CISAAdvisory]:
        """
        Scrape CISA medical device advisories.

        Args:
            max_results: Maximum number of advisories to return.
            include_general_ics: Also search general ICS advisories for medical keywords.

        Returns:
            List of CISAAdvisory objects.
        """
        advisories = []

        # First, get medical-specific advisories (ICSMA)
        logger.info("Fetching CISA medical advisories (ICSMA)...")
        soup = self._make_request(self.MEDICAL_ADVISORIES_URL)
        if soup:
            medical_list = self._parse_advisory_list_page(soup)
            logger.info(f"Found {len(medical_list)} medical advisories")

            for basic_info in medical_list[:max_results]:
                logger.info(f"  Fetching details: {basic_info['advisory_id']}")
                advisory = self._parse_advisory_detail(basic_info["url"], basic_info)
                if advisory:
                    advisories.append(advisory)

                if len(advisories) >= max_results:
                    break

        # Optionally search general ICS advisories for medical keywords
        if include_general_ics and len(advisories) < max_results:
            logger.info("Searching general ICS advisories for medical keywords...")
            soup = self._make_request(self.ICS_ADVISORIES_URL)
            if soup:
                ics_list = self._parse_advisory_list_page(soup)

                for basic_info in ics_list:
                    if len(advisories) >= max_results:
                        break

                    # Check if title contains medical keywords
                    title_lower = basic_info["title"].lower()
                    if any(kw.lower() in title_lower for kw in self.MEDICAL_KEYWORDS):
                        # Skip if we already have this advisory
                        if any(a.advisory_id == basic_info["advisory_id"] for a in advisories):
                            continue

                        logger.info(f"  Fetching details: {basic_info['advisory_id']}")
                        advisory = self._parse_advisory_detail(basic_info["url"], basic_info)
                        if advisory:
                            advisories.append(advisory)

        logger.info(f"Total advisories collected: {len(advisories)}")
        return advisories

    def save_results(
        self,
        advisories: list[CISAAdvisory],
        output_path: str | Path,
    ) -> None:
        """
        Save advisory results to a JSON file.

        Args:
            advisories: List of CISAAdvisory objects.
            output_path: Path to save the file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_advisories": len(advisories),
                "source": "CISA ICS-CERT",
            },
            "advisories": [asdict(adv) for adv in advisories],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(advisories)} advisories to {output_path}")

    def generate_claude_prompt(
        self,
        advisories: list[CISAAdvisory],
        batch_size: int = 10,
    ) -> str:
        """
        Generate a prompt for Claude.ai to analyze advisories.

        Args:
            advisories: List of CISAAdvisory objects.
            batch_size: Number of advisories per prompt.

        Returns:
            Formatted prompt for Claude.ai.
        """
        prompt_template = """You are a medical device cybersecurity expert. Analyze the following CISA ICS-CERT advisories and extract structured threat intelligence.

For each advisory, provide:
1. **Device Type**: What type of medical device is affected? (imaging, therapeutic, monitoring, laboratory, healthcare_it, or other)
2. **Clinical Impact**: What is the potential impact on patient care? (HIGH: direct patient harm, MEDIUM: care disruption, LOW: administrative)
3. **Exploitability**: How easy is this to exploit? (EASY: remote/unauthenticated, MODERATE: requires some access, HARD: requires physical access)

Advisories to analyze:
"""

        advisory_list = []
        for adv in advisories[:batch_size]:
            adv_text = f"""
---
**{adv.advisory_id}**: {adv.title}
CVSS: {adv.cvss_score or 'N/A'} ({adv.severity or 'N/A'})
CVEs: {', '.join(adv.cve_ids) if adv.cve_ids else 'N/A'}
Vendor: {adv.vendor or 'N/A'}
Description: {(adv.description or '')[:400]}{'...' if adv.description and len(adv.description) > 400 else ''}
"""
            advisory_list.append(adv_text)

        prompt = prompt_template + "\n".join(advisory_list)

        prompt += """

Respond in JSON format:
```json
{
  "analyses": [
    {
      "advisory_id": "ICSMA-XX-XXX-XX",
      "device_type": "imaging|therapeutic|monitoring|laboratory|healthcare_it|other",
      "clinical_impact": "HIGH|MEDIUM|LOW",
      "exploitability": "EASY|MODERATE|HARD",
      "reasoning": "Brief explanation of your analysis"
    }
  ]
}
```"""

        return prompt


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape CISA ICS-CERT medical device advisories"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of advisories to fetch (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/threat_intel/advisories/cisa_medical.json",
        help="Output file path",
    )
    parser.add_argument(
        "--generate-prompt",
        action="store_true",
        help="Generate Claude.ai prompt for analysis",
    )
    parser.add_argument(
        "--include-general-ics",
        action="store_true",
        default=True,
        help="Also search general ICS advisories for medical keywords",
    )

    args = parser.parse_args()

    scraper = CISAScraper()

    logger.info("Scraping CISA medical device advisories...")
    advisories = scraper.scrape_medical_advisories(
        max_results=args.max_results,
        include_general_ics=args.include_general_ics,
    )

    # Save results
    scraper.save_results(advisories, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("CISA MEDICAL DEVICE ADVISORY SUMMARY")
    print("=" * 60)

    print(f"\nTotal Advisories: {len(advisories)}")

    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "N/A": 0}
    for adv in advisories:
        sev = adv.severity or "N/A"
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print("\nBy Severity:")
    for sev, count in severity_counts.items():
        if count > 0:
            print(f"  {sev}: {count}")

    print("\nTop 5 Advisories:")
    sorted_advisories = sorted(advisories, key=lambda x: -(x.cvss_score or 0))
    for adv in sorted_advisories[:5]:
        print(f"  {adv.advisory_id}: {adv.cvss_score or 'N/A'} - {adv.title[:50]}...")

    # Generate Claude prompt if requested
    if args.generate_prompt:
        prompt = scraper.generate_claude_prompt(advisories)
        prompt_path = Path(args.output).parent / "cisa_claude_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"\nClaude.ai prompt saved to: {prompt_path}")

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
