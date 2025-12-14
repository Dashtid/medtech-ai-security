"""
IEC 62443 Security Assessment Module.

Provides security level assessment per IEC 62443 framework for industrial
automation and control systems, recognized by FDA as a consensus standard
for medical device cybersecurity.

References:
- IEC 62443-3-3: System security requirements and security levels
- IEC 62443-4-2: Technical security requirements for IACS components
- FDA Guidance on Medical Device Cybersecurity (2023)
- IEC 81001-5-1: Security for health software and health IT systems

Security Levels (SL):
- SL 1: Protection against casual or coincidental violation
- SL 2: Protection against intentional violation using simple means
- SL 3: Protection against sophisticated attacks with moderate resources
- SL 4: Protection against state-sponsored attacks with extensive resources
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SecurityLevel(Enum):
    """IEC 62443 Security Levels."""

    SL_0 = 0  # No specific requirements
    SL_1 = 1  # Protection against casual/coincidental violation
    SL_2 = 2  # Protection against intentional violation, simple means
    SL_3 = 3  # Protection against sophisticated attacks, moderate resources
    SL_4 = 4  # Protection against state-sponsored attacks


class SecurityLevelTarget(Enum):
    """Types of Security Level targets."""

    SL_T = "target"  # Target Security Level (desired)
    SL_C = "capability"  # Capability Security Level (component capability)
    SL_A = "achieved"  # Achieved Security Level (actual implementation)


class FoundationalRequirement(Enum):
    """IEC 62443-3-3 Foundational Requirements (FR)."""

    FR_1 = "Identification and Authentication Control (IAC)"
    FR_2 = "Use Control (UC)"
    FR_3 = "System Integrity (SI)"
    FR_4 = "Data Confidentiality (DC)"
    FR_5 = "Restricted Data Flow (RDF)"
    FR_6 = "Timely Response to Events (TRE)"
    FR_7 = "Resource Availability (RA)"


class ComplianceStatus(Enum):
    """Compliance status for requirements."""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    NOT_ASSESSED = "not_assessed"


@dataclass
class SystemRequirement:
    """Individual system requirement from IEC 62443-3-3."""

    sr_id: str  # e.g., "SR 1.1"
    name: str
    description: str
    foundational_requirement: FoundationalRequirement
    security_levels: dict[SecurityLevel, str]  # SL -> requirement text
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    evidence: str = ""
    notes: str = ""
    assessed_date: str | None = None


@dataclass
class RequirementEnhancement:
    """Requirement Enhancement (RE) for higher security levels."""

    re_id: str  # e.g., "SR 1.1 RE 1"
    parent_sr: str
    name: str
    description: str
    required_sl: SecurityLevel  # Minimum SL where this RE is required
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    evidence: str = ""


@dataclass
class Zone:
    """Security zone in the IEC 62443 zone and conduit model."""

    zone_id: str
    name: str
    description: str
    target_sl: SecurityLevel
    achieved_sl: SecurityLevel = SecurityLevel.SL_0
    assets: list[str] = field(default_factory=list)
    security_policies: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ZoneConduit:
    """Conduit connecting two zones."""

    conduit_id: str
    name: str
    source_zone: str  # Zone ID
    destination_zone: str  # Zone ID
    communication_type: str  # e.g., "TCP/IP", "Serial", "Wireless"
    target_sl: SecurityLevel
    achieved_sl: SecurityLevel = SecurityLevel.SL_0
    security_controls: list[str] = field(default_factory=list)
    data_flows: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SOUPComponent:
    """Software of Unknown Provenance (SOUP) per IEC 62304."""

    name: str
    version: str
    vendor: str
    purpose: str
    risk_class: str  # A, B, or C per IEC 62304
    security_assessment: str
    known_vulnerabilities: list[str] = field(default_factory=list)
    update_mechanism: str = ""
    sbom_reference: str = ""


@dataclass
class AssessmentReport:
    """Complete IEC 62443 security assessment report."""

    report_id: str
    system_name: str
    system_description: str
    assessment_date: str
    assessor: str
    target_sl: SecurityLevel
    achieved_sl: SecurityLevel
    zones: list[Zone]
    conduits: list[ZoneConduit]
    requirement_results: dict[str, ComplianceStatus]
    soup_inventory: list[SOUPComponent]
    gaps: list[str]
    recommendations: list[str]
    executive_summary: str = ""

    def to_markdown(self) -> str:
        """Generate markdown assessment report."""
        lines = [
            "# IEC 62443 Security Assessment Report",
            "",
            "## Executive Summary",
            "",
            self.executive_summary if self.executive_summary else "_No executive summary provided._",
            "",
            "## System Information",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Report ID | {self.report_id} |",
            f"| System Name | {self.system_name} |",
            f"| Assessment Date | {self.assessment_date} |",
            f"| Assessor | {self.assessor} |",
            f"| Target Security Level | SL-{self.target_sl.value} |",
            f"| Achieved Security Level | SL-{self.achieved_sl.value} |",
            "",
            "### System Description",
            "",
            self.system_description,
            "",
            "## Security Level Assessment",
            "",
            "### Target vs Achieved",
            "",
            f"- **Target SL**: SL-{self.target_sl.value} - {_sl_description(self.target_sl)}",
            f"- **Achieved SL**: SL-{self.achieved_sl.value} - {_sl_description(self.achieved_sl)}",
            "",
        ]

        # Gap status
        if self.achieved_sl.value >= self.target_sl.value:
            lines.append("[+] **Status**: Target security level ACHIEVED")
        else:
            gap = self.target_sl.value - self.achieved_sl.value
            lines.append(f"[!] **Status**: Gap of {gap} security level(s) to target")

        lines.extend([
            "",
            "## Zone and Conduit Model",
            "",
            "### Zones",
            "",
        ])

        if self.zones:
            lines.append("| Zone ID | Name | Target SL | Achieved SL | Assets |")
            lines.append("|---------|------|-----------|-------------|--------|")
            for zone in self.zones:
                assets = ", ".join(zone.assets[:3])
                if len(zone.assets) > 3:
                    assets += f" (+{len(zone.assets) - 3} more)"
                lines.append(
                    f"| {zone.zone_id} | {zone.name} | SL-{zone.target_sl.value} | "
                    f"SL-{zone.achieved_sl.value} | {assets} |"
                )
        else:
            lines.append("_No zones defined._")

        lines.extend([
            "",
            "### Conduits",
            "",
        ])

        if self.conduits:
            lines.append("| Conduit ID | Name | Source | Destination | Type | SL |")
            lines.append("|------------|------|--------|-------------|------|-----|")
            for conduit in self.conduits:
                lines.append(
                    f"| {conduit.conduit_id} | {conduit.name} | {conduit.source_zone} | "
                    f"{conduit.destination_zone} | {conduit.communication_type} | "
                    f"SL-{conduit.achieved_sl.value} |"
                )
        else:
            lines.append("_No conduits defined._")

        lines.extend([
            "",
            "## Foundational Requirements Assessment",
            "",
        ])

        # Group results by FR
        fr_results: dict[str, list[tuple[str, ComplianceStatus]]] = {}
        for sr_id, status in self.requirement_results.items():
            fr = sr_id.split(".")[0] if "." in sr_id else "Other"
            if fr not in fr_results:
                fr_results[fr] = []
            fr_results[fr].append((sr_id, status))

        for fr, results in sorted(fr_results.items()):
            fr_name = _get_fr_name(fr)
            lines.append(f"### {fr}: {fr_name}")
            lines.append("")
            lines.append("| Requirement | Status |")
            lines.append("|-------------|--------|")
            for sr_id, status in sorted(results):
                status_icon = _status_icon(status)
                lines.append(f"| {sr_id} | {status_icon} {status.value.replace('_', ' ').title()} |")
            lines.append("")

        lines.extend([
            "## SOUP Inventory (IEC 62304)",
            "",
        ])

        if self.soup_inventory:
            lines.append("| Component | Version | Vendor | Risk Class | Known Vulns |")
            lines.append("|-----------|---------|--------|------------|-------------|")
            for soup in self.soup_inventory:
                vuln_count = len(soup.known_vulnerabilities)
                vuln_status = f"{vuln_count} known" if vuln_count > 0 else "None"
                lines.append(
                    f"| {soup.name} | {soup.version} | {soup.vendor} | "
                    f"Class {soup.risk_class} | {vuln_status} |"
                )
        else:
            lines.append("_No SOUP components documented._")

        lines.extend([
            "",
            "## Identified Gaps",
            "",
        ])

        if self.gaps:
            for i, gap in enumerate(self.gaps, 1):
                lines.append(f"{i}. {gap}")
        else:
            lines.append("[+] No gaps identified.")

        lines.extend([
            "",
            "## Recommendations",
            "",
        ])

        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
        else:
            lines.append("_No specific recommendations._")

        lines.extend([
            "",
            "---",
            "",
            "## FDA Compliance Note",
            "",
            "This assessment follows IEC 62443 standards recognized by FDA as consensus ",
            "standards for medical device cybersecurity. The security level framework ",
            "aligns with FDA's risk-based approach to premarket cybersecurity review.",
            "",
            "### Regulatory References",
            "",
            "- FDA Guidance: Content of Premarket Submissions for Management of ",
            "  Cybersecurity in Medical Devices (2023)",
            "- IEC 62443-3-3: System security requirements and security levels",
            "- IEC 62443-4-2: Technical security requirements for IACS components",
            "- IEC 81001-5-1: Security for health software and health IT systems",
            "- IEC 62304: Medical device software life cycle processes",
            "",
            f"_Report generated: {datetime.now(timezone.utc).isoformat()}_",
        ])

        return "\n".join(lines)


def _sl_description(sl: SecurityLevel) -> str:
    """Get description for security level."""
    descriptions = {
        SecurityLevel.SL_0: "No specific security requirements",
        SecurityLevel.SL_1: "Protection against casual/coincidental violation",
        SecurityLevel.SL_2: "Protection against intentional violation using simple means",
        SecurityLevel.SL_3: "Protection against sophisticated attacks with moderate resources",
        SecurityLevel.SL_4: "Protection against state-sponsored attacks with extensive resources",
    }
    return descriptions.get(sl, "Unknown")


def _status_icon(status: ComplianceStatus) -> str:
    """Get icon for compliance status."""
    icons = {
        ComplianceStatus.COMPLIANT: "[+]",
        ComplianceStatus.PARTIAL: "[~]",
        ComplianceStatus.NON_COMPLIANT: "[-]",
        ComplianceStatus.NOT_APPLICABLE: "[N/A]",
        ComplianceStatus.NOT_ASSESSED: "[?]",
    }
    return icons.get(status, "[?]")


def _get_fr_name(fr_id: str) -> str:
    """Get foundational requirement name from ID."""
    fr_names = {
        "SR 1": "Identification and Authentication Control",
        "SR 2": "Use Control",
        "SR 3": "System Integrity",
        "SR 4": "Data Confidentiality",
        "SR 5": "Restricted Data Flow",
        "SR 6": "Timely Response to Events",
        "SR 7": "Resource Availability",
    }
    return fr_names.get(fr_id, "Unknown")


# =============================================================================
# IEC 62443-3-3 System Requirements Database
# =============================================================================

# Core system requirements from IEC 62443-3-3
SYSTEM_REQUIREMENTS: dict[str, dict[str, Any]] = {
    # FR 1: Identification and Authentication Control
    "SR 1.1": {
        "name": "Human user identification and authentication",
        "fr": FoundationalRequirement.FR_1,
        "sl_requirements": {
            SecurityLevel.SL_1: "Identify and authenticate all human users",
            SecurityLevel.SL_2: "Unique identification for all human users",
            SecurityLevel.SL_3: "Multi-factor authentication for all users",
            SecurityLevel.SL_4: "Hardware-based authentication tokens",
        },
    },
    "SR 1.2": {
        "name": "Software process and device identification and authentication",
        "fr": FoundationalRequirement.FR_1,
        "sl_requirements": {
            SecurityLevel.SL_1: "Identify software processes",
            SecurityLevel.SL_2: "Authenticate software processes",
            SecurityLevel.SL_3: "Cryptographic authentication of processes",
            SecurityLevel.SL_4: "Certificate-based authentication",
        },
    },
    "SR 1.3": {
        "name": "Account management",
        "fr": FoundationalRequirement.FR_1,
        "sl_requirements": {
            SecurityLevel.SL_1: "Support account management",
            SecurityLevel.SL_2: "Automated account management",
            SecurityLevel.SL_3: "Integration with enterprise identity management",
            SecurityLevel.SL_4: "Continuous account validation",
        },
    },
    "SR 1.4": {
        "name": "Identifier management",
        "fr": FoundationalRequirement.FR_1,
        "sl_requirements": {
            SecurityLevel.SL_1: "Support unique identifiers",
            SecurityLevel.SL_2: "Automated identifier lifecycle management",
            SecurityLevel.SL_3: "Centralized identifier management",
            SecurityLevel.SL_4: "Cryptographically protected identifiers",
        },
    },
    "SR 1.5": {
        "name": "Authenticator management",
        "fr": FoundationalRequirement.FR_1,
        "sl_requirements": {
            SecurityLevel.SL_1: "Manage authenticators",
            SecurityLevel.SL_2: "Enforce authenticator strength",
            SecurityLevel.SL_3: "Hardware-protected authenticator storage",
            SecurityLevel.SL_4: "HSM-based authenticator management",
        },
    },
    # FR 2: Use Control
    "SR 2.1": {
        "name": "Authorization enforcement",
        "fr": FoundationalRequirement.FR_2,
        "sl_requirements": {
            SecurityLevel.SL_1: "Enforce assigned authorizations",
            SecurityLevel.SL_2: "Role-based access control",
            SecurityLevel.SL_3: "Attribute-based access control",
            SecurityLevel.SL_4: "Multi-level security enforcement",
        },
    },
    "SR 2.2": {
        "name": "Wireless use control",
        "fr": FoundationalRequirement.FR_2,
        "sl_requirements": {
            SecurityLevel.SL_1: "Control wireless access",
            SecurityLevel.SL_2: "Encrypted wireless communications",
            SecurityLevel.SL_3: "Strong wireless authentication",
            SecurityLevel.SL_4: "Wireless intrusion detection",
        },
    },
    "SR 2.3": {
        "name": "Use control for portable/mobile devices",
        "fr": FoundationalRequirement.FR_2,
        "sl_requirements": {
            SecurityLevel.SL_1: "Control portable device connections",
            SecurityLevel.SL_2: "Authenticate portable devices",
            SecurityLevel.SL_3: "Encrypted portable device storage",
            SecurityLevel.SL_4: "Remote wipe capability",
        },
    },
    # FR 3: System Integrity
    "SR 3.1": {
        "name": "Communication integrity",
        "fr": FoundationalRequirement.FR_3,
        "sl_requirements": {
            SecurityLevel.SL_1: "Protect communication integrity",
            SecurityLevel.SL_2: "Cryptographic integrity protection",
            SecurityLevel.SL_3: "Strong cryptographic algorithms",
            SecurityLevel.SL_4: "Quantum-resistant algorithms",
        },
    },
    "SR 3.2": {
        "name": "Malicious code protection",
        "fr": FoundationalRequirement.FR_3,
        "sl_requirements": {
            SecurityLevel.SL_1: "Malware protection mechanisms",
            SecurityLevel.SL_2: "Automated malware updates",
            SecurityLevel.SL_3: "Application whitelisting",
            SecurityLevel.SL_4: "Behavioral malware detection",
        },
    },
    "SR 3.3": {
        "name": "Security functionality verification",
        "fr": FoundationalRequirement.FR_3,
        "sl_requirements": {
            SecurityLevel.SL_1: "Verify security functions operate correctly",
            SecurityLevel.SL_2: "Automated security function testing",
            SecurityLevel.SL_3: "Continuous security monitoring",
            SecurityLevel.SL_4: "Self-healing security mechanisms",
        },
    },
    "SR 3.4": {
        "name": "Software and information integrity",
        "fr": FoundationalRequirement.FR_3,
        "sl_requirements": {
            SecurityLevel.SL_1: "Detect software changes",
            SecurityLevel.SL_2: "Cryptographic integrity verification",
            SecurityLevel.SL_3: "Secure boot and measured launch",
            SecurityLevel.SL_4: "Hardware root of trust",
        },
    },
    # FR 4: Data Confidentiality
    "SR 4.1": {
        "name": "Information confidentiality",
        "fr": FoundationalRequirement.FR_4,
        "sl_requirements": {
            SecurityLevel.SL_1: "Protect confidential information",
            SecurityLevel.SL_2: "Encrypt confidential data",
            SecurityLevel.SL_3: "Strong encryption (AES-256)",
            SecurityLevel.SL_4: "Hardware-based encryption",
        },
    },
    "SR 4.2": {
        "name": "Information persistence",
        "fr": FoundationalRequirement.FR_4,
        "sl_requirements": {
            SecurityLevel.SL_1: "Protect stored information",
            SecurityLevel.SL_2: "Secure deletion mechanisms",
            SecurityLevel.SL_3: "Cryptographic erasure",
            SecurityLevel.SL_4: "Physical destruction support",
        },
    },
    # FR 5: Restricted Data Flow
    "SR 5.1": {
        "name": "Network segmentation",
        "fr": FoundationalRequirement.FR_5,
        "sl_requirements": {
            SecurityLevel.SL_1: "Segment control system networks",
            SecurityLevel.SL_2: "Firewall-based segmentation",
            SecurityLevel.SL_3: "Deep packet inspection",
            SecurityLevel.SL_4: "Micro-segmentation",
        },
    },
    "SR 5.2": {
        "name": "Zone boundary protection",
        "fr": FoundationalRequirement.FR_5,
        "sl_requirements": {
            SecurityLevel.SL_1: "Monitor zone boundaries",
            SecurityLevel.SL_2: "Filter communications at boundaries",
            SecurityLevel.SL_3: "Protocol-aware filtering",
            SecurityLevel.SL_4: "Content inspection and filtering",
        },
    },
    # FR 6: Timely Response to Events
    "SR 6.1": {
        "name": "Audit log accessibility",
        "fr": FoundationalRequirement.FR_6,
        "sl_requirements": {
            SecurityLevel.SL_1: "Provide read access to audit logs",
            SecurityLevel.SL_2: "Searchable audit logs",
            SecurityLevel.SL_3: "Centralized log management",
            SecurityLevel.SL_4: "Real-time log analysis",
        },
    },
    "SR 6.2": {
        "name": "Continuous monitoring",
        "fr": FoundationalRequirement.FR_6,
        "sl_requirements": {
            SecurityLevel.SL_1: "Monitor security-relevant events",
            SecurityLevel.SL_2: "Automated event correlation",
            SecurityLevel.SL_3: "Security information and event management",
            SecurityLevel.SL_4: "AI-based anomaly detection",
        },
    },
    # FR 7: Resource Availability
    "SR 7.1": {
        "name": "Denial of service protection",
        "fr": FoundationalRequirement.FR_7,
        "sl_requirements": {
            SecurityLevel.SL_1: "Detect DoS conditions",
            SecurityLevel.SL_2: "Mitigate DoS attacks",
            SecurityLevel.SL_3: "Automated DoS response",
            SecurityLevel.SL_4: "Distributed DoS protection",
        },
    },
    "SR 7.2": {
        "name": "Resource management",
        "fr": FoundationalRequirement.FR_7,
        "sl_requirements": {
            SecurityLevel.SL_1: "Manage system resources",
            SecurityLevel.SL_2: "Resource quotas and limits",
            SecurityLevel.SL_3: "Dynamic resource allocation",
            SecurityLevel.SL_4: "Predictive resource management",
        },
    },
    "SR 7.3": {
        "name": "Control system backup",
        "fr": FoundationalRequirement.FR_7,
        "sl_requirements": {
            SecurityLevel.SL_1: "Backup control system data",
            SecurityLevel.SL_2: "Automated backups",
            SecurityLevel.SL_3: "Encrypted backups",
            SecurityLevel.SL_4: "Immutable backup storage",
        },
    },
    "SR 7.4": {
        "name": "Control system recovery and reconstitution",
        "fr": FoundationalRequirement.FR_7,
        "sl_requirements": {
            SecurityLevel.SL_1: "Recover from failures",
            SecurityLevel.SL_2: "Documented recovery procedures",
            SecurityLevel.SL_3: "Automated recovery",
            SecurityLevel.SL_4: "Hot standby with automatic failover",
        },
    },
}


class IEC62443Assessor:
    """
    IEC 62443 Security Level Assessor.

    Performs security assessments per IEC 62443-3-3 system requirements
    and generates compliance reports for medical device cybersecurity.

    Example:
        >>> assessor = IEC62443Assessor(
        ...     system_name="Medical Device Controller",
        ...     target_sl=SecurityLevel.SL_2,
        ... )
        >>> assessor.add_zone(Zone(
        ...     zone_id="Z1",
        ...     name="Control Zone",
        ...     description="Primary control system zone",
        ...     target_sl=SecurityLevel.SL_2,
        ... ))
        >>> assessor.assess_requirement("SR 1.1", ComplianceStatus.COMPLIANT, "MFA implemented")
        >>> report = assessor.generate_report(assessor="Security Team")
    """

    def __init__(
        self,
        system_name: str,
        system_description: str = "",
        target_sl: SecurityLevel = SecurityLevel.SL_2,
    ) -> None:
        """
        Initialize the IEC 62443 assessor.

        Args:
            system_name: Name of the system being assessed
            system_description: Description of the system
            target_sl: Target security level for the assessment
        """
        self.system_name = system_name
        self.system_description = system_description
        self.target_sl = target_sl
        self.zones: list[Zone] = []
        self.conduits: list[ZoneConduit] = []
        self.soup_inventory: list[SOUPComponent] = []
        self.requirement_results: dict[str, tuple[ComplianceStatus, str]] = {}
        self._initialized = datetime.now(timezone.utc).isoformat()

    def add_zone(self, zone: Zone) -> None:
        """Add a security zone to the assessment."""
        self.zones.append(zone)

    def add_conduit(self, conduit: ZoneConduit) -> None:
        """Add a conduit to the assessment."""
        self.conduits.append(conduit)

    def add_soup(self, soup: SOUPComponent) -> None:
        """Add a SOUP component to the inventory."""
        self.soup_inventory.append(soup)

    def assess_requirement(
        self,
        sr_id: str,
        status: ComplianceStatus,
        evidence: str = "",
    ) -> None:
        """
        Assess a specific system requirement.

        Args:
            sr_id: System requirement ID (e.g., "SR 1.1")
            status: Compliance status
            evidence: Evidence supporting the assessment
        """
        if sr_id not in SYSTEM_REQUIREMENTS:
            raise ValueError(f"Unknown system requirement: {sr_id}")
        self.requirement_results[sr_id] = (status, evidence)

    def get_unassessed_requirements(self) -> list[str]:
        """Get list of requirements not yet assessed."""
        all_srs = set(SYSTEM_REQUIREMENTS.keys())
        assessed = set(self.requirement_results.keys())
        return sorted(all_srs - assessed)

    def get_requirements_for_sl(self, sl: SecurityLevel) -> list[str]:
        """Get requirements applicable for a given security level."""
        applicable = []
        for sr_id, sr_data in SYSTEM_REQUIREMENTS.items():
            if sl in sr_data["sl_requirements"]:
                applicable.append(sr_id)
        return sorted(applicable)

    def calculate_achieved_sl(self) -> SecurityLevel:
        """
        Calculate achieved security level based on assessments.

        Returns the highest SL where all applicable requirements are compliant.
        """
        for sl in [SecurityLevel.SL_4, SecurityLevel.SL_3, SecurityLevel.SL_2, SecurityLevel.SL_1]:
            applicable_reqs = self.get_requirements_for_sl(sl)
            all_compliant = True

            for sr_id in applicable_reqs:
                if sr_id in self.requirement_results:
                    status, _ = self.requirement_results[sr_id]
                    if status not in [ComplianceStatus.COMPLIANT, ComplianceStatus.NOT_APPLICABLE]:
                        all_compliant = False
                        break
                else:
                    all_compliant = False
                    break

            if all_compliant:
                return sl

        return SecurityLevel.SL_0

    def identify_gaps(self) -> list[str]:
        """Identify gaps between target and achieved security levels."""
        gaps = []
        applicable_reqs = self.get_requirements_for_sl(self.target_sl)

        for sr_id in applicable_reqs:
            if sr_id not in self.requirement_results:
                sr_name = SYSTEM_REQUIREMENTS[sr_id]["name"]
                gaps.append(f"{sr_id} ({sr_name}): Not assessed")
            else:
                status, _ = self.requirement_results[sr_id]
                if status == ComplianceStatus.NON_COMPLIANT:
                    sr_name = SYSTEM_REQUIREMENTS[sr_id]["name"]
                    gaps.append(f"{sr_id} ({sr_name}): Non-compliant")
                elif status == ComplianceStatus.PARTIAL:
                    sr_name = SYSTEM_REQUIREMENTS[sr_id]["name"]
                    gaps.append(f"{sr_id} ({sr_name}): Partially compliant")

        # Zone gaps
        for zone in self.zones:
            if zone.achieved_sl.value < zone.target_sl.value:
                gaps.append(
                    f"Zone {zone.zone_id} ({zone.name}): Achieved SL-{zone.achieved_sl.value} "
                    f"below target SL-{zone.target_sl.value}"
                )

        return gaps

    def generate_recommendations(self) -> list[str]:
        """Generate recommendations based on assessment results."""
        recommendations = []
        gaps = self.identify_gaps()

        if gaps:
            recommendations.append(
                "Address identified gaps to achieve target security level"
            )

        # Check SOUP inventory
        high_risk_soup = [s for s in self.soup_inventory if s.risk_class == "C"]
        if high_risk_soup:
            recommendations.append(
                f"Review {len(high_risk_soup)} high-risk (Class C) SOUP components for vulnerabilities"
            )

        # Check for known vulnerabilities
        vuln_soup = [s for s in self.soup_inventory if s.known_vulnerabilities]
        if vuln_soup:
            total_vulns = sum(len(s.known_vulnerabilities) for s in vuln_soup)
            recommendations.append(
                f"Remediate {total_vulns} known vulnerabilities in SOUP components"
            )

        # Zone recommendations
        for zone in self.zones:
            if not zone.security_policies:
                recommendations.append(
                    f"Document security policies for zone {zone.zone_id} ({zone.name})"
                )

        # Conduit recommendations
        for conduit in self.conduits:
            if not conduit.security_controls:
                recommendations.append(
                    f"Implement security controls for conduit {conduit.conduit_id}"
                )

        # General recommendations based on target SL
        if self.target_sl.value >= 2:
            recommendations.append(
                "Implement continuous security monitoring per FR 6"
            )
        if self.target_sl.value >= 3:
            recommendations.append(
                "Consider third-party penetration testing to validate security controls"
            )

        return recommendations

    def generate_report(
        self,
        assessor: str,
        report_id: str | None = None,
        executive_summary: str = "",
    ) -> AssessmentReport:
        """
        Generate complete assessment report.

        Args:
            assessor: Name/ID of the assessor
            report_id: Optional report identifier
            executive_summary: Executive summary for the report

        Returns:
            Complete assessment report
        """
        achieved_sl = self.calculate_achieved_sl()
        gaps = self.identify_gaps()
        recommendations = self.generate_recommendations()

        if not report_id:
            report_id = f"IEC62443-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        # Convert requirement results to simple status dict
        req_status = {sr_id: status for sr_id, (status, _) in self.requirement_results.items()}

        return AssessmentReport(
            report_id=report_id,
            system_name=self.system_name,
            system_description=self.system_description,
            assessment_date=datetime.now(timezone.utc).isoformat(),
            assessor=assessor,
            target_sl=self.target_sl,
            achieved_sl=achieved_sl,
            zones=self.zones,
            conduits=self.conduits,
            requirement_results=req_status,
            soup_inventory=self.soup_inventory,
            gaps=gaps,
            recommendations=recommendations,
            executive_summary=executive_summary,
        )

    def export_checklist(self) -> str:
        """
        Export assessment as a markdown checklist.

        Returns:
            Markdown checklist for manual assessment
        """
        lines = [
            f"# IEC 62443 Assessment Checklist",
            f"",
            f"**System**: {self.system_name}",
            f"**Target Security Level**: SL-{self.target_sl.value}",
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
            f"",
        ]

        # Group by foundational requirement
        by_fr: dict[FoundationalRequirement, list[str]] = {}
        for sr_id, sr_data in SYSTEM_REQUIREMENTS.items():
            fr = sr_data["fr"]
            if fr not in by_fr:
                by_fr[fr] = []
            by_fr[fr].append(sr_id)

        for fr in FoundationalRequirement:
            lines.append(f"## {fr.value}")
            lines.append("")

            if fr not in by_fr:
                continue

            for sr_id in sorted(by_fr[fr]):
                sr_data = SYSTEM_REQUIREMENTS[sr_id]
                sr_name = sr_data["name"]
                sl_req = sr_data["sl_requirements"].get(self.target_sl, "N/A")

                status_mark = " "
                if sr_id in self.requirement_results:
                    status, _ = self.requirement_results[sr_id]
                    if status == ComplianceStatus.COMPLIANT:
                        status_mark = "x"
                    elif status == ComplianceStatus.PARTIAL:
                        status_mark = "~"
                    elif status == ComplianceStatus.NOT_APPLICABLE:
                        status_mark = "-"

                lines.append(f"- [{status_mark}] **{sr_id}**: {sr_name}")
                lines.append(f"  - Requirement (SL-{self.target_sl.value}): {sl_req}")
                lines.append(f"  - Evidence: _________________")
                lines.append("")

        return "\n".join(lines)
