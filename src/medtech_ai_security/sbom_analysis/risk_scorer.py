"""
Supply Chain Risk Scorer for Medical Device SBOM Analysis.

Implements comprehensive risk scoring that considers:
- Direct and transitive vulnerabilities
- Package criticality based on dependency depth
- Medical device regulatory context
- License compliance risks
- Supply chain attack surface

Based on FDA SBOM guidance and 2025 medical device security standards.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from medtech_ai_security.sbom_analysis.parser import (
    DependencyGraph,
    Package,
    VulnerabilityInfo,
)

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RiskCategory(Enum):
    """Categories of supply chain risk."""

    VULNERABILITY = "vulnerability"
    LICENSE = "license"
    DEPENDENCY_DEPTH = "dependency_depth"
    UNMAINTAINED = "unmaintained"
    TRANSITIVE_EXPOSURE = "transitive_exposure"
    ATTACK_SURFACE = "attack_surface"
    REGULATORY = "regulatory"


@dataclass
class PackageRisk:
    """Risk assessment for a single package."""

    package_id: str
    package_name: str
    package_version: str
    risk_level: RiskLevel = RiskLevel.INFO
    risk_score: float = 0.0  # 0-100 scale

    # Risk breakdown
    vulnerability_score: float = 0.0
    license_score: float = 0.0
    dependency_score: float = 0.0
    position_score: float = 0.0

    # Details
    vulnerabilities: list[VulnerabilityInfo] = field(default_factory=list)
    transitive_vulnerabilities: int = 0
    dependent_count: int = 0  # How many packages depend on this
    depth_from_root: int = 0

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "package_id": self.package_id,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "breakdown": {
                "vulnerability": self.vulnerability_score,
                "license": self.license_score,
                "dependency": self.dependency_score,
                "position": self.position_score,
            },
            "vulnerability_count": len(self.vulnerabilities),
            "transitive_vulnerabilities": self.transitive_vulnerabilities,
            "dependent_count": self.dependent_count,
            "depth_from_root": self.depth_from_root,
            "recommendations": self.recommendations,
        }


@dataclass
class RiskReport:
    """Complete supply chain risk report."""

    # Overall metrics
    overall_risk_level: RiskLevel = RiskLevel.INFO
    overall_risk_score: float = 0.0

    # Package counts
    total_packages: int = 0
    vulnerable_packages: int = 0
    critical_packages: int = 0
    high_risk_packages: int = 0

    # Vulnerability counts
    total_vulnerabilities: int = 0
    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    medium_vulnerabilities: int = 0
    low_vulnerabilities: int = 0

    # Per-package risks
    package_risks: list[PackageRisk] = field(default_factory=list)

    # Attack surface metrics
    direct_dependencies: int = 0
    transitive_dependencies: int = 0
    max_dependency_depth: int = 0
    avg_dependency_depth: float = 0.0

    # License risks
    license_issues: list[str] = field(default_factory=list)

    # Summary and recommendations
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)

    # Medical device specific
    fda_compliance_notes: list[str] = field(default_factory=list)
    eu_mdr_compliance_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall": {
                "risk_level": self.overall_risk_level.value,
                "risk_score": self.overall_risk_score,
            },
            "packages": {
                "total": self.total_packages,
                "vulnerable": self.vulnerable_packages,
                "critical": self.critical_packages,
                "high_risk": self.high_risk_packages,
            },
            "vulnerabilities": {
                "total": self.total_vulnerabilities,
                "critical": self.critical_vulnerabilities,
                "high": self.high_vulnerabilities,
                "medium": self.medium_vulnerabilities,
                "low": self.low_vulnerabilities,
            },
            "attack_surface": {
                "direct_dependencies": self.direct_dependencies,
                "transitive_dependencies": self.transitive_dependencies,
                "max_depth": self.max_dependency_depth,
                "avg_depth": self.avg_dependency_depth,
            },
            "license_issues": self.license_issues,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "compliance": {
                "fda": self.fda_compliance_notes,
                "eu_mdr": self.eu_mdr_compliance_notes,
            },
            "package_details": [pr.to_dict() for pr in self.package_risks],
        }


class SupplyChainRiskScorer:
    """Calculates comprehensive supply chain risk scores for medical devices."""

    # Risk weights for medical device context
    WEIGHTS = {
        "vulnerability": 0.40,  # Known CVEs
        "license": 0.10,  # License compliance
        "dependency": 0.20,  # Dependency depth/complexity
        "position": 0.30,  # Position in supply chain (criticality)
    }

    # CVSS to risk score mapping
    CVSS_MULTIPLIER = {
        "critical": 10.0,  # CVSS >= 9.0
        "high": 7.0,  # CVSS >= 7.0
        "medium": 4.0,  # CVSS >= 4.0
        "low": 1.0,  # CVSS < 4.0
    }

    # Problematic licenses for medical devices
    RISKY_LICENSES = {
        "GPL-3.0": RiskLevel.HIGH,
        "GPL-2.0": RiskLevel.HIGH,
        "AGPL-3.0": RiskLevel.CRITICAL,
        "LGPL-3.0": RiskLevel.MEDIUM,
        "LGPL-2.1": RiskLevel.MEDIUM,
        "SSPL": RiskLevel.CRITICAL,
        "BUSL": RiskLevel.HIGH,
        "CC-BY-NC": RiskLevel.HIGH,
        "NOASSERTION": RiskLevel.MEDIUM,
        "UNKNOWN": RiskLevel.MEDIUM,
    }

    # Permissive licenses (preferred for medical devices)
    PERMISSIVE_LICENSES = {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "CC0-1.0",
        "0BSD",
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        medical_context: bool = True,
    ):
        """Initialize risk scorer.

        Args:
            weights: Custom risk weights
            medical_context: Apply medical device-specific adjustments
        """
        self.weights = weights or self.WEIGHTS.copy()
        self.medical_context = medical_context

    def score(self, dep_graph: DependencyGraph) -> RiskReport:
        """Calculate comprehensive risk score for an SBOM.

        Args:
            dep_graph: Parsed dependency graph

        Returns:
            Complete risk report
        """
        report = RiskReport()
        report.total_packages = dep_graph.package_count

        # Compute depths and dependency counts
        depths = self._compute_depths(dep_graph)
        dependents = self._compute_dependents(dep_graph)

        # Analyze each package
        for pkg_id, pkg in dep_graph.packages.items():
            pkg_risk = self._score_package(pkg, dep_graph, depths, dependents)
            report.package_risks.append(pkg_risk)

            # Aggregate statistics
            if pkg.vulnerabilities:
                report.vulnerable_packages += 1

            if pkg_risk.risk_level == RiskLevel.CRITICAL:
                report.critical_packages += 1
            elif pkg_risk.risk_level == RiskLevel.HIGH:
                report.high_risk_packages += 1

        # Aggregate vulnerability counts
        self._aggregate_vulnerabilities(report, dep_graph)

        # Calculate attack surface metrics
        self._calculate_attack_surface(report, dep_graph, depths)

        # Identify license issues
        self._check_licenses(report, dep_graph)

        # Calculate overall risk
        self._calculate_overall_risk(report)

        # Generate recommendations
        self._generate_recommendations(report, dep_graph)

        # Add compliance notes for medical devices
        if self.medical_context:
            self._add_compliance_notes(report, dep_graph)

        # Generate summary
        report.summary = self._generate_summary(report)

        return report

    def _score_package(
        self,
        pkg: Package,
        dep_graph: DependencyGraph,
        depths: dict[str, int],
        dependents: dict[str, int],
    ) -> PackageRisk:
        """Score a single package."""
        pkg_risk = PackageRisk(
            package_id=pkg.id,
            package_name=pkg.name,
            package_version=pkg.version,
            vulnerabilities=pkg.vulnerabilities.copy(),
            depth_from_root=depths.get(pkg.id, 0),
            dependent_count=dependents.get(pkg.id, 0),
        )

        # 1. Vulnerability score (0-100)
        vuln_score = 0.0
        for vuln in pkg.vulnerabilities:
            severity = (vuln.severity or "").upper()
            cvss = vuln.cvss_score

            if severity == "CRITICAL" or cvss >= 9.0:
                vuln_score += self.CVSS_MULTIPLIER["critical"] * cvss
            elif severity == "HIGH" or cvss >= 7.0:
                vuln_score += self.CVSS_MULTIPLIER["high"] * cvss
            elif severity == "MEDIUM" or cvss >= 4.0:
                vuln_score += self.CVSS_MULTIPLIER["medium"] * cvss
            else:
                vuln_score += self.CVSS_MULTIPLIER["low"] * max(cvss, 1.0)

        pkg_risk.vulnerability_score = min(vuln_score, 100.0)

        # Check transitive vulnerabilities
        trans_deps = dep_graph.get_transitive_dependencies(pkg.id)
        trans_vuln_count = sum(len(d.vulnerabilities) for d in trans_deps)
        pkg_risk.transitive_vulnerabilities = trans_vuln_count

        # Add transitive exposure to score
        if trans_vuln_count > 0:
            pkg_risk.vulnerability_score = min(
                pkg_risk.vulnerability_score + (trans_vuln_count * 2), 100.0
            )

        # 2. License score (0-100)
        pkg_risk.license_score = self._score_license(pkg.license)

        # 3. Dependency score (0-100) - complexity and depth
        direct_deps = len(dep_graph.get_direct_dependencies(pkg.id))
        trans_deps_count = len(trans_deps)
        depth = depths.get(pkg.id, 0)

        # More dependencies = higher risk
        dep_complexity = min((direct_deps * 5) + (trans_deps_count * 2), 50)
        depth_risk = min(depth * 10, 50)
        pkg_risk.dependency_score = dep_complexity + depth_risk

        # 4. Position score (0-100) - criticality based on dependents
        # Packages with many dependents are more critical
        num_dependents = dependents.get(pkg.id, 0)
        is_root = dep_graph.root_package == pkg.id

        if is_root:
            pkg_risk.position_score = 100.0  # Root is most critical
        elif num_dependents > 10:
            pkg_risk.position_score = 80.0
        elif num_dependents > 5:
            pkg_risk.position_score = 60.0
        elif num_dependents > 2:
            pkg_risk.position_score = 40.0
        elif num_dependents > 0:
            pkg_risk.position_score = 20.0
        else:
            pkg_risk.position_score = 10.0

        # Calculate weighted risk score
        pkg_risk.risk_score = (
            self.weights["vulnerability"] * pkg_risk.vulnerability_score
            + self.weights["license"] * pkg_risk.license_score
            + self.weights["dependency"] * pkg_risk.dependency_score
            + self.weights["position"] * pkg_risk.position_score
        )

        # Determine risk level
        pkg_risk.risk_level = self._score_to_level(pkg_risk.risk_score)

        # Add package-specific recommendations
        self._add_package_recommendations(pkg_risk, pkg)

        return pkg_risk

    def _score_license(self, license_str: str) -> float:
        """Score license risk (0-100, higher = riskier)."""
        if not license_str:
            return 30.0  # Unknown license is moderate risk

        license_upper = license_str.upper()

        # Check for permissive licenses
        for permissive in self.PERMISSIVE_LICENSES:
            if permissive.upper() in license_upper:
                return 0.0

        # Check for risky licenses
        for risky, level in self.RISKY_LICENSES.items():
            if risky.upper() in license_upper:
                if level == RiskLevel.CRITICAL:
                    return 100.0
                elif level == RiskLevel.HIGH:
                    return 70.0
                elif level == RiskLevel.MEDIUM:
                    return 40.0
                else:
                    return 20.0

        # Unknown license - moderate risk
        return 30.0

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score >= 70:
            return RiskLevel.CRITICAL
        elif score >= 50:
            return RiskLevel.HIGH
        elif score >= 30:
            return RiskLevel.MEDIUM
        elif score >= 10:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFO

    def _compute_depths(self, dep_graph: DependencyGraph) -> dict[str, int]:
        """Compute depth from root for each package."""
        depths: dict[str, int] = {}

        # Find roots
        roots = set(dep_graph.packages.keys())
        for dep in dep_graph.dependencies:
            roots.discard(dep.target)

        if dep_graph.root_package:
            roots = {dep_graph.root_package}

        # BFS from roots
        queue = list(roots)
        for root in roots:
            depths[root] = 0

        while queue:
            current = queue.pop(0)
            current_depth = depths.get(current, 0)

            for dep in dep_graph.dependencies:
                if dep.source == current and dep.target not in depths:
                    depths[dep.target] = current_depth + 1
                    queue.append(dep.target)

        # Handle disconnected components
        for pkg_id in dep_graph.packages:
            if pkg_id not in depths:
                depths[pkg_id] = 0

        return depths

    def _compute_dependents(self, dep_graph: DependencyGraph) -> dict[str, int]:
        """Count how many packages depend on each package."""
        dependents: dict[str, int] = dict.fromkeys(dep_graph.packages, 0)

        for dep in dep_graph.dependencies:
            if dep.target in dependents:
                dependents[dep.target] += 1

        return dependents

    def _aggregate_vulnerabilities(self, report: RiskReport, dep_graph: DependencyGraph) -> None:
        """Aggregate vulnerability counts."""
        for pkg in dep_graph.packages.values():
            for vuln in pkg.vulnerabilities:
                report.total_vulnerabilities += 1
                severity = (vuln.severity or "").upper()
                cvss = vuln.cvss_score

                if severity == "CRITICAL" or cvss >= 9.0:
                    report.critical_vulnerabilities += 1
                elif severity == "HIGH" or cvss >= 7.0:
                    report.high_vulnerabilities += 1
                elif severity == "MEDIUM" or cvss >= 4.0:
                    report.medium_vulnerabilities += 1
                else:
                    report.low_vulnerabilities += 1

    def _calculate_attack_surface(
        self,
        report: RiskReport,
        dep_graph: DependencyGraph,
        depths: dict[str, int],
    ) -> None:
        """Calculate attack surface metrics."""
        # Direct vs transitive dependencies
        direct_deps = set()
        if dep_graph.root_package:
            for dep in dep_graph.dependencies:
                if dep.source == dep_graph.root_package:
                    direct_deps.add(dep.target)

        report.direct_dependencies = len(direct_deps)
        report.transitive_dependencies = dep_graph.package_count - len(direct_deps) - 1

        # Depth metrics
        if depths:
            report.max_dependency_depth = max(depths.values())
            report.avg_dependency_depth = sum(depths.values()) / len(depths)
        else:
            report.max_dependency_depth = 0
            report.avg_dependency_depth = 0.0

    def _check_licenses(self, report: RiskReport, dep_graph: DependencyGraph) -> None:
        """Check for license issues."""
        for pkg in dep_graph.packages.values():
            if not pkg.license:
                report.license_issues.append(f"{pkg.name}@{pkg.version}: No license declared")
                continue

            for risky, level in self.RISKY_LICENSES.items():
                if risky.upper() in pkg.license.upper():
                    report.license_issues.append(
                        f"{pkg.name}@{pkg.version}: {level.value.upper()} - "
                        f"Uses {risky} license"
                    )
                    break

    def _calculate_overall_risk(self, report: RiskReport) -> None:
        """Calculate overall supply chain risk."""
        if not report.package_risks:
            report.overall_risk_score = 0.0
            report.overall_risk_level = RiskLevel.INFO
            return

        # Weighted by position (critical packages matter more)
        total_weight = 0.0
        weighted_score = 0.0

        for pkg_risk in report.package_risks:
            weight = pkg_risk.position_score / 100.0 + 0.1  # Minimum weight 0.1
            weighted_score += pkg_risk.risk_score * weight
            total_weight += weight

        if total_weight > 0:
            report.overall_risk_score = weighted_score / total_weight
        else:
            report.overall_risk_score = 0.0

        # Boost for critical vulnerabilities
        if report.critical_vulnerabilities > 0:
            report.overall_risk_score = min(report.overall_risk_score + 20, 100.0)

        report.overall_risk_level = self._score_to_level(report.overall_risk_score)

    def _add_package_recommendations(self, pkg_risk: PackageRisk, pkg: Package) -> None:
        """Add recommendations for a specific package."""
        if pkg_risk.vulnerability_score > 50:
            critical_vulns = [
                v
                for v in pkg.vulnerabilities
                if v.cvss_score >= 9.0 or (v.severity or "").upper() == "CRITICAL"
            ]
            if critical_vulns:
                pkg_risk.recommendations.append(
                    f"URGENT: Update to fix {len(critical_vulns)} critical vulnerabilities"
                )
            else:
                pkg_risk.recommendations.append(
                    "Update to latest version to address known vulnerabilities"
                )

        if pkg_risk.transitive_vulnerabilities > 5:
            pkg_risk.recommendations.append(
                f"Review {pkg_risk.transitive_vulnerabilities} transitive vulnerabilities"
            )

        if pkg_risk.license_score > 50:
            pkg_risk.recommendations.append(f"Review license compliance: {pkg.license}")

    def _generate_recommendations(self, report: RiskReport, dep_graph: DependencyGraph) -> None:
        """Generate overall recommendations."""
        recommendations = []

        # Critical vulnerabilities
        if report.critical_vulnerabilities > 0:
            recommendations.append(
                f"[CRITICAL] Address {report.critical_vulnerabilities} critical "
                f"vulnerabilities immediately"
            )

        # High vulnerabilities
        if report.high_vulnerabilities > 0:
            recommendations.append(
                f"[HIGH] Review and remediate {report.high_vulnerabilities} "
                f"high-severity vulnerabilities"
            )

        # Deep dependency chains
        if report.max_dependency_depth > 5:
            recommendations.append(
                f"[MEDIUM] Consider reducing dependency depth "
                f"(current max: {report.max_dependency_depth})"
            )

        # Large attack surface
        if report.transitive_dependencies > 50:
            recommendations.append(
                f"[MEDIUM] Large transitive dependency count "
                f"({report.transitive_dependencies}) increases attack surface"
            )

        # License issues
        if len(report.license_issues) > 0:
            recommendations.append(f"[INFO] Review {len(report.license_issues)} license issues")

        # If no major issues
        if not recommendations:
            recommendations.append("[OK] No critical supply chain risks identified")

        report.recommendations = recommendations

    def _add_compliance_notes(self, report: RiskReport, dep_graph: DependencyGraph) -> None:
        """Add medical device compliance notes."""
        # FDA SBOM requirements
        if report.critical_vulnerabilities > 0:
            report.fda_compliance_notes.append(
                "FDA: Critical vulnerabilities require immediate disclosure and remediation"
            )

        if report.vulnerable_packages > 0:
            report.fda_compliance_notes.append(
                f"FDA: SBOM shows {report.vulnerable_packages} packages with known vulnerabilities"
            )

        report.fda_compliance_notes.append(
            f"FDA: SBOM contains {report.total_packages} components for 510(k) submission"
        )

        # EU MDR requirements
        if report.critical_vulnerabilities > 0:
            report.eu_mdr_compliance_notes.append(
                "EU MDR: Critical vulnerabilities must be reported to notified body"
            )

        if report.overall_risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            report.eu_mdr_compliance_notes.append(
                "EU MDR: High supply chain risk - additional risk management required"
            )

        report.eu_mdr_compliance_notes.append(
            "EU MDR: Maintain SBOM as part of technical documentation"
        )

    def _generate_summary(self, report: RiskReport) -> str:
        """Generate executive summary."""
        level = report.overall_risk_level.value.upper()
        score = report.overall_risk_score

        summary = f"Supply Chain Risk: {level} ({score:.1f}/100)\n\n"
        summary += f"Analysis of {report.total_packages} packages:\n"
        summary += f"- {report.vulnerable_packages} packages with known vulnerabilities\n"
        summary += f"- {report.total_vulnerabilities} total vulnerabilities "
        summary += f"({report.critical_vulnerabilities} critical, "
        summary += f"{report.high_vulnerabilities} high)\n"
        summary += f"- Dependency depth: max {report.max_dependency_depth}, "
        summary += f"avg {report.avg_dependency_depth:.1f}\n"

        if report.license_issues:
            summary += f"- {len(report.license_issues)} license compliance issues\n"

        return summary


if __name__ == "__main__":
    # Test risk scoring
    from medtech_ai_security.sbom_analysis.parser import SBOMParser, create_sample_sbom

    sample_sbom = create_sample_sbom()
    parser = SBOMParser()
    dep_graph = parser.parse_json(sample_sbom)

    scorer = SupplyChainRiskScorer()
    report = scorer.score(dep_graph)

    print("[+] Supply Chain Risk Report")
    print("=" * 50)
    print(report.summary)

    print("\n[+] Per-Package Risks:")
    for pkg_risk in sorted(report.package_risks, key=lambda x: x.risk_score, reverse=True):
        print(
            f"    {pkg_risk.package_name}@{pkg_risk.package_version}: "
            f"{pkg_risk.risk_level.value.upper()} ({pkg_risk.risk_score:.1f})"
        )
        for rec in pkg_risk.recommendations:
            print(f"        -> {rec}")

    print("\n[+] Recommendations:")
    for rec in report.recommendations:
        print(f"    {rec}")

    print("\n[+] FDA Compliance Notes:")
    for note in report.fda_compliance_notes:
        print(f"    {note}")
