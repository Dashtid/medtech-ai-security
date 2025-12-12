"""
DefectDojo API Integration

Provides integration with DefectDojo for vulnerability management:
- Create/update products and engagements
- Import SBOM analysis findings
- Import threat intelligence CVEs
- Create test results from ML predictions

DefectDojo API v2 Reference:
https://demo.defectdojo.org/api/v2/doc/

Usage:
    from medtech_ai_security.integration.defectdojo import DefectDojoClient

    client = DefectDojoClient(
        url="https://defectdojo.example.com",
        api_key="your-api-key"
    )

    # Create or get product
    product_id = client.get_or_create_product("Medical Device Firmware")

    # Import SBOM findings
    client.import_sbom_findings(product_id, sbom_analysis_report)

    # Import threat intel CVEs
    client.import_threat_intel(product_id, enriched_cves)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class DefectDojoConfig:
    """Configuration for DefectDojo connection."""

    url: str
    api_key: str
    verify_ssl: bool = True
    timeout: int = 30
    product_type: str = "Medical Device"
    default_severity: str = "Medium"


@dataclass
class Finding:
    """Represents a DefectDojo finding."""

    title: str
    description: str
    severity: str
    cve: str | None = None
    cvss_score: float | None = None
    cwe: int | None = None
    references: str = ""
    mitigation: str = ""
    impact: str = ""
    component_name: str = ""
    component_version: str = ""
    file_path: str = ""
    line: int | None = None
    verified: bool = False
    active: bool = True
    duplicate: bool = False
    static_finding: bool = True
    dynamic_finding: bool = False
    tags: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to DefectDojo API format."""
        data = {
            "title": self.title[:200],  # DefectDojo has a 200 char limit
            "description": self.description,
            "severity": self.severity,
            "verified": self.verified,
            "active": self.active,
            "duplicate": self.duplicate,
            "static_finding": self.static_finding,
            "dynamic_finding": self.dynamic_finding,
        }

        if self.cve:
            data["cve"] = self.cve

        if self.cvss_score is not None:
            data["cvssv3_score"] = self.cvss_score

        if self.cwe:
            data["cwe"] = self.cwe

        if self.references:
            data["references"] = self.references

        if self.mitigation:
            data["mitigation"] = self.mitigation

        if self.impact:
            data["impact"] = self.impact

        if self.component_name:
            data["component_name"] = self.component_name

        if self.component_version:
            data["component_version"] = self.component_version

        if self.file_path:
            data["file_path"] = self.file_path

        if self.line is not None:
            data["line"] = self.line

        if self.tags:
            data["tags"] = self.tags

        return data


class DefectDojoClient:
    """
    Client for DefectDojo API v2.

    Provides methods for:
    - Product and engagement management
    - Finding import from various sources
    - Test result creation

    Example:
        client = DefectDojoClient(
            url="https://defectdojo.example.com",
            api_key="your-api-key"
        )
        product_id = client.get_or_create_product("My Device")
        engagement_id = client.create_engagement(product_id, "Security Scan Q1")
        test_id = client.create_test(engagement_id, "SBOM Analysis")
        client.create_finding(test_id, finding)
    """

    # Severity mapping
    SEVERITY_MAP = {
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "info": "Informational",
        "informational": "Informational",
    }

    # Test type IDs (common defaults in DefectDojo)
    TEST_TYPES = {
        "sbom_analysis": "SBOM Analysis",
        "threat_intel": "Threat Intelligence",
        "adversarial_ml": "Adversarial ML Testing",
        "anomaly_detection": "Anomaly Detection",
        "risk_assessment": "Risk Assessment",
        "manual": "Manual Testing",
    }

    def __init__(
        self,
        url: str,
        api_key: str,
        verify_ssl: bool = True,
        timeout: int = 30,
    ):
        """
        Initialize DefectDojo client.

        Args:
            url: DefectDojo instance URL (e.g., https://defectdojo.example.com)
            api_key: API key (generate from DefectDojo UI)
            verify_ssl: Verify SSL certificates
            timeout: Request timeout in seconds
        """
        self.base_url = url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v2"
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict | None:
        """
        Make an API request.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            endpoint: API endpoint (e.g., /products/)
            data: Request body data
            params: Query parameters

        Returns:
            Response JSON or None on error
        """
        url = f"{self.api_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            response.raise_for_status()

            if response.status_code == 204:  # No content
                return {"success": True}

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"DefectDojo API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    # Product Management

    def get_products(self, name: str | None = None) -> list[dict]:
        """
        Get list of products.

        Args:
            name: Optional filter by product name

        Returns:
            List of product dictionaries
        """
        params = {}
        if name:
            params["name"] = name

        result = self._request("GET", "/products/", params=params)
        return result.get("results", []) if result else []

    def get_product(self, product_id: int) -> dict | None:
        """Get product by ID."""
        return self._request("GET", f"/products/{product_id}/")

    def create_product(
        self,
        name: str,
        description: str = "",
        prod_type: int = 1,
        tags: list[str] | None = None,
    ) -> dict | None:
        """
        Create a new product.

        Args:
            name: Product name
            description: Product description
            prod_type: Product type ID (default: 1)
            tags: Optional list of tags

        Returns:
            Created product data or None
        """
        data = {
            "name": name,
            "description": description or f"Medical device: {name}",
            "prod_type": prod_type,
        }

        if tags:
            data["tags"] = tags

        return self._request("POST", "/products/", data=data)

    def get_or_create_product(
        self,
        name: str,
        description: str = "",
        prod_type: int = 1,
    ) -> int | None:
        """
        Get existing product or create new one.

        Args:
            name: Product name
            description: Product description
            prod_type: Product type ID

        Returns:
            Product ID or None on error
        """
        products = self.get_products(name=name)

        for product in products:
            if product.get("name") == name:
                logger.info(f"Found existing product: {name} (ID: {product['id']})")
                return product["id"]

        # Create new product
        result = self.create_product(name, description, prod_type)
        if result:
            logger.info(f"Created new product: {name} (ID: {result['id']})")
            return result["id"]

        return None

    # Engagement Management

    def create_engagement(
        self,
        product_id: int,
        name: str,
        description: str = "",
        engagement_type: str = "CI/CD",
        status: str = "In Progress",
        deduplication_on_engagement: bool = True,
    ) -> dict | None:
        """
        Create a new engagement.

        Args:
            product_id: Product ID
            name: Engagement name
            description: Engagement description
            engagement_type: Type (CI/CD, Interactive)
            status: Status (Not Started, In Progress, Completed)
            deduplication_on_engagement: Enable finding deduplication

        Returns:
            Created engagement data or None
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        data = {
            "product": product_id,
            "name": name,
            "description": description or f"Automated security scan: {name}",
            "engagement_type": engagement_type,
            "status": status,
            "target_start": today,
            "target_end": today,
            "deduplication_on_engagement": deduplication_on_engagement,
        }

        return self._request("POST", "/engagements/", data=data)

    def get_engagements(
        self,
        product_id: int | None = None,
        name: str | None = None,
    ) -> list[dict]:
        """Get list of engagements."""
        params: dict[str, int | str] = {}
        if product_id:
            params["product"] = product_id
        if name:
            params["name"] = name

        result = self._request("GET", "/engagements/", params=params)
        return result.get("results", []) if result else []

    def close_engagement(self, engagement_id: int) -> dict | None:
        """Close an engagement."""
        data = {"status": "Completed"}
        return self._request("PATCH", f"/engagements/{engagement_id}/", data=data)

    # Test Management

    def create_test(
        self,
        engagement_id: int,
        test_type: str = "SBOM Analysis",
        title: str = "",
        description: str = "",
    ) -> dict | None:
        """
        Create a new test within an engagement.

        Args:
            engagement_id: Engagement ID
            test_type: Test type name
            title: Test title
            description: Test description

        Returns:
            Created test data or None
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Get or create test type
        test_type_id = self._get_or_create_test_type(test_type)
        if not test_type_id:
            logger.error(f"Could not get/create test type: {test_type}")
            return None

        data = {
            "engagement": engagement_id,
            "test_type": test_type_id,
            "title": title or test_type,
            "description": description,
            "target_start": today,
            "target_end": today,
        }

        return self._request("POST", "/tests/", data=data)

    def _get_or_create_test_type(self, name: str) -> int | None:
        """Get or create a test type."""
        # Search for existing
        result = self._request("GET", "/test_types/", params={"name": name})
        if result and result.get("results"):
            return result["results"][0]["id"]

        # Create new
        result = self._request("POST", "/test_types/", data={"name": name})
        if result:
            return result["id"]

        # Use default
        return 1

    # Finding Management

    def create_finding(
        self,
        test_id: int,
        finding: Finding,
    ) -> dict | None:
        """
        Create a new finding.

        Args:
            test_id: Test ID
            finding: Finding object

        Returns:
            Created finding data or None
        """
        data = finding.to_dict()
        data["test"] = test_id
        data["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return self._request("POST", "/findings/", data=data)

    def create_findings_batch(
        self,
        test_id: int,
        findings: list[Finding],
    ) -> list[dict]:
        """
        Create multiple findings.

        Args:
            test_id: Test ID
            findings: List of Finding objects

        Returns:
            List of created findings
        """
        created = []
        for finding in findings:
            result = self.create_finding(test_id, finding)
            if result:
                created.append(result)
                logger.debug(f"Created finding: {finding.title}")
            else:
                logger.warning(f"Failed to create finding: {finding.title}")

        logger.info(f"Created {len(created)}/{len(findings)} findings")
        return created

    def get_findings(
        self,
        test_id: int | None = None,
        severity: str | None = None,
        active: bool | None = None,
    ) -> list[dict]:
        """Get list of findings."""
        params: dict[str, int | str | bool] = {}
        if test_id:
            params["test"] = test_id
        if severity:
            params["severity"] = severity
        if active is not None:
            params["active"] = active

        result = self._request("GET", "/findings/", params=params)
        return result.get("results", []) if result else []

    # Import Methods for MedTech AI Security

    def import_sbom_findings(
        self,
        product_id: int,
        analysis_report: dict,
        engagement_name: str = "SBOM Security Analysis",
    ) -> dict:
        """
        Import findings from SBOM analysis report.

        Args:
            product_id: Product ID
            analysis_report: SBOM analysis report dict
            engagement_name: Name for the engagement

        Returns:
            Import summary
        """
        # Create engagement
        engagement = self.create_engagement(
            product_id=product_id,
            name=engagement_name,
            description="Automated SBOM supply chain security analysis",
        )
        if not engagement:
            return {"error": "Failed to create engagement"}

        # Create test
        test = self.create_test(
            engagement_id=engagement["id"],
            test_type="SBOM Analysis",
            title="Supply Chain Risk Assessment",
        )
        if not test:
            return {"error": "Failed to create test"}

        # Convert analysis report to findings
        findings = self._convert_sbom_to_findings(analysis_report)

        # Create findings
        created = self.create_findings_batch(test["id"], findings)

        # Close engagement
        self.close_engagement(engagement["id"])

        return {
            "engagement_id": engagement["id"],
            "test_id": test["id"],
            "findings_created": len(created),
            "findings_total": len(findings),
        }

    def _convert_sbom_to_findings(self, report: dict) -> list[Finding]:
        """Convert SBOM analysis report to DefectDojo findings."""
        findings = []

        # Get package details from risk report
        risk_report = report.get("risk_report", {})
        package_details = risk_report.get("package_details", [])

        for pkg in package_details:
            # Skip low-risk packages without vulnerabilities
            if pkg.get("vulnerability_count", 0) == 0:
                continue

            severity = self.SEVERITY_MAP.get(
                pkg.get("risk_level", "medium").lower(), "Medium"
            )

            finding = Finding(
                title=f"Vulnerable dependency: {pkg.get('package_name', 'Unknown')}@{pkg.get('package_version', '')}",
                description=self._build_sbom_description(pkg),
                severity=severity,
                component_name=pkg.get("package_name", ""),
                component_version=pkg.get("package_version", ""),
                mitigation="Update to a patched version or apply vendor mitigation",
                impact=f"Risk score: {pkg.get('risk_score', 0):.1f}/100",
                tags=["sbom", "supply-chain", "dependency"],
            )

            findings.append(finding)

        return findings

    def _build_sbom_description(self, pkg: dict) -> str:
        """Build finding description from package data."""
        lines = [
            f"## Package: {pkg.get('package_name', 'Unknown')}",
            f"**Version**: {pkg.get('package_version', 'Unknown')}",
            f"**Risk Level**: {pkg.get('risk_level', 'Unknown').upper()}",
            f"**Risk Score**: {pkg.get('risk_score', 0):.1f}/100",
            "",
            "### Risk Breakdown",
            f"- Vulnerability Score: {pkg.get('breakdown', {}).get('vulnerability', 0):.1f}",
            f"- License Score: {pkg.get('breakdown', {}).get('license', 0):.1f}",
            f"- Dependency Score: {pkg.get('breakdown', {}).get('dependency', 0):.1f}",
            f"- Position Score: {pkg.get('breakdown', {}).get('position', 0):.1f}",
            "",
        ]

        if pkg.get("recommendations"):
            lines.append("### Recommendations")
            for rec in pkg.get("recommendations", []):
                lines.append(f"- {rec}")

        return "\n".join(lines)

    def import_threat_intel(
        self,
        product_id: int,
        cve_data: list[dict],
        engagement_name: str = "Threat Intelligence Analysis",
    ) -> dict:
        """
        Import findings from threat intelligence CVE data.

        Args:
            product_id: Product ID
            cve_data: List of enriched CVE dictionaries
            engagement_name: Name for the engagement

        Returns:
            Import summary
        """
        # Create engagement
        engagement = self.create_engagement(
            product_id=product_id,
            name=engagement_name,
            description="Automated threat intelligence analysis for medical devices",
        )
        if not engagement:
            return {"error": "Failed to create engagement"}

        # Create test
        test = self.create_test(
            engagement_id=engagement["id"],
            test_type="Threat Intelligence",
            title="CVE Analysis",
        )
        if not test:
            return {"error": "Failed to create test"}

        # Convert CVEs to findings
        findings = self._convert_cves_to_findings(cve_data)

        # Create findings
        created = self.create_findings_batch(test["id"], findings)

        # Close engagement
        self.close_engagement(engagement["id"])

        return {
            "engagement_id": engagement["id"],
            "test_id": test["id"],
            "findings_created": len(created),
            "findings_total": len(findings),
        }

    def _convert_cves_to_findings(self, cve_data: list[dict]) -> list[Finding]:
        """Convert CVE data to DefectDojo findings."""
        findings = []

        for cve in cve_data:
            cve_id = cve.get("cve_id", "Unknown")
            cvss = cve.get("cvss_score", 0.0)

            # Determine severity from CVSS
            if cvss >= 9.0:
                severity = "Critical"
            elif cvss >= 7.0:
                severity = "High"
            elif cvss >= 4.0:
                severity = "Medium"
            else:
                severity = "Low"

            finding = Finding(
                title=f"{cve_id}: {cve.get('description', 'No description')[:150]}",
                description=self._build_cve_description(cve),
                severity=severity,
                cve=cve_id,
                cvss_score=cvss,
                cwe=cve.get("cwe_id"),
                references="\n".join(cve.get("references", [])),
                mitigation=cve.get("mitigation", "Apply vendor patches"),
                impact=cve.get("clinical_impact", ""),
                tags=["threat-intel", "cve", cve.get("device_type", "medical-device")],
            )

            findings.append(finding)

        return findings

    def _build_cve_description(self, cve: dict) -> str:
        """Build finding description from CVE data."""
        lines = [
            f"## {cve.get('cve_id', 'Unknown CVE')}",
            "",
            f"**CVSS Score**: {cve.get('cvss_score', 'N/A')}",
            f"**Published**: {cve.get('published_date', 'Unknown')}",
            "",
            "### Description",
            cve.get("description", "No description available"),
            "",
        ]

        if cve.get("affected_products"):
            lines.append("### Affected Products")
            for product in cve.get("affected_products", []):
                lines.append(f"- {product}")
            lines.append("")

        if cve.get("clinical_impact"):
            lines.append("### Clinical Impact")
            lines.append(cve.get("clinical_impact"))
            lines.append("")

        if cve.get("device_type"):
            lines.append(f"**Device Type**: {cve.get('device_type')}")

        if cve.get("exploitability"):
            lines.append(f"**Exploitability**: {cve.get('exploitability')}")

        return "\n".join(lines)

    # Utility Methods

    def test_connection(self) -> bool:
        """Test connection to DefectDojo."""
        result = self._request("GET", "/products/", params={"limit": 1})
        return result is not None

    def get_product_types(self) -> list[dict]:
        """Get list of product types."""
        result = self._request("GET", "/product_types/")
        return result.get("results", []) if result else []

    def export_findings_json(self, test_id: int, output_path: str | Path) -> bool:
        """Export findings to JSON file."""
        findings = self.get_findings(test_id=test_id)

        if not findings:
            logger.warning("No findings to export")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(findings, f, indent=2)

        logger.info(f"Exported {len(findings)} findings to {output_path}")
        return True


def main() -> None:
    """CLI for DefectDojo integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DefectDojo API integration for MedTech AI Security"
    )
    parser.add_argument("--url", required=True, help="DefectDojo instance URL")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--test", action="store_true", help="Test connection")
    parser.add_argument("--list-products", action="store_true", help="List products")

    args = parser.parse_args()

    client = DefectDojoClient(url=args.url, api_key=args.api_key)

    if args.test:
        if client.test_connection():
            print("[OK] Connection successful")
        else:
            print("[FAIL] Connection failed")
            return

    if args.list_products:
        products = client.get_products()
        print(f"Found {len(products)} products:")
        for p in products:
            print(f"  - {p['name']} (ID: {p['id']})")


if __name__ == "__main__":
    main()
