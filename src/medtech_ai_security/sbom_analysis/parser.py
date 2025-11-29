"""
SBOM Parser for CycloneDX and SPDX formats.

Parses Software Bill of Materials files and extracts package and dependency
information for graph-based analysis.

Supported formats:
- CycloneDX JSON (1.4, 1.5, 1.6)
- CycloneDX XML
- SPDX JSON (2.2, 2.3)
- SPDX Tag-Value
"""

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SBOMFormat(Enum):
    """Supported SBOM formats."""

    CYCLONEDX_JSON = "cyclonedx-json"
    CYCLONEDX_XML = "cyclonedx-xml"
    SPDX_JSON = "spdx-json"
    SPDX_TAGVALUE = "spdx-tagvalue"
    UNKNOWN = "unknown"


class PackageType(Enum):
    """Package ecosystem types."""

    NPM = "npm"
    PYPI = "pypi"
    MAVEN = "maven"
    NUGET = "nuget"
    CARGO = "cargo"
    GO = "go"
    APK = "apk"
    DEB = "deb"
    RPM = "rpm"
    CONTAINER = "container"
    UNKNOWN = "unknown"


@dataclass
class VulnerabilityInfo:
    """Information about a known vulnerability."""

    cve_id: str
    cvss_score: float = 0.0
    cvss_vector: str = ""
    severity: str = "UNKNOWN"
    description: str = ""
    fixed_version: Optional[str] = None
    source: str = ""


@dataclass
class Package:
    """Represents a software package in the SBOM."""

    name: str
    version: str
    purl: str = ""  # Package URL (purl)
    package_type: PackageType = PackageType.UNKNOWN
    license: str = ""
    description: str = ""
    supplier: str = ""
    checksum: str = ""
    external_refs: list[str] = field(default_factory=list)
    vulnerabilities: list[VulnerabilityInfo] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique identifier for the package."""
        if self.purl:
            return self.purl
        return f"{self.name}@{self.version}"

    @property
    def ecosystem(self) -> str:
        """Get package ecosystem from purl."""
        if self.purl:
            # Parse purl: pkg:type/namespace/name@version
            match = re.match(r"pkg:(\w+)/", self.purl)
            if match:
                return match.group(1)
        return self.package_type.value


@dataclass
class Dependency:
    """Represents a dependency relationship between packages."""

    source: str  # Package ID of the dependent
    target: str  # Package ID of the dependency
    dependency_type: str = "direct"  # direct, transitive, dev, optional
    scope: str = ""  # compile, runtime, test, etc.
    version_constraint: str = ""  # Version range constraint


@dataclass
class DependencyGraph:
    """Graph representation of SBOM dependencies."""

    packages: dict[str, Package] = field(default_factory=dict)
    dependencies: list[Dependency] = field(default_factory=list)
    root_package: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_package(self, package: Package) -> None:
        """Add a package to the graph."""
        self.packages[package.id] = package

    def add_dependency(self, dependency: Dependency) -> None:
        """Add a dependency relationship."""
        self.dependencies.append(dependency)

    def get_direct_dependencies(self, package_id: str) -> list[Package]:
        """Get direct dependencies of a package."""
        deps = []
        for dep in self.dependencies:
            if dep.source == package_id:
                if dep.target in self.packages:
                    deps.append(self.packages[dep.target])
        return deps

    def get_transitive_dependencies(
        self, package_id: str, visited: Optional[set[str]] = None
    ) -> list[Package]:
        """Get all transitive dependencies of a package."""
        if visited is None:
            visited = set()

        if package_id in visited:
            return []

        visited.add(package_id)
        result = []

        for pkg in self.get_direct_dependencies(package_id):
            result.append(pkg)
            result.extend(self.get_transitive_dependencies(pkg.id, visited))

        return result

    def get_dependents(self, package_id: str) -> list[Package]:
        """Get packages that depend on this package."""
        dependents = []
        for dep in self.dependencies:
            if dep.target == package_id:
                if dep.source in self.packages:
                    dependents.append(self.packages[dep.source])
        return dependents

    def get_vulnerable_packages(self) -> list[Package]:
        """Get all packages with known vulnerabilities."""
        return [pkg for pkg in self.packages.values() if pkg.vulnerabilities]

    @property
    def package_count(self) -> int:
        """Total number of packages."""
        return len(self.packages)

    @property
    def dependency_count(self) -> int:
        """Total number of dependency relationships."""
        return len(self.dependencies)

    @property
    def vulnerability_count(self) -> int:
        """Total number of known vulnerabilities."""
        return sum(len(pkg.vulnerabilities) for pkg in self.packages.values())


class SBOMParser:
    """Parser for SBOM files in various formats."""

    def __init__(self, vuln_db: Optional[dict[str, list[VulnerabilityInfo]]] = None):
        """Initialize parser with optional vulnerability database.

        Args:
            vuln_db: Optional mapping of package purls to known vulnerabilities
        """
        self.vuln_db = vuln_db or {}

    def parse(self, file_path: str | Path) -> DependencyGraph:
        """Parse an SBOM file and return a dependency graph.

        Args:
            file_path: Path to the SBOM file

        Returns:
            DependencyGraph with parsed packages and dependencies
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SBOM file not found: {path}")

        content = path.read_text(encoding="utf-8")
        sbom_format = self._detect_format(path, content)

        logger.info(f"Parsing SBOM file: {path} (format: {sbom_format.value})")

        if sbom_format == SBOMFormat.CYCLONEDX_JSON:
            return self._parse_cyclonedx_json(content)
        elif sbom_format == SBOMFormat.CYCLONEDX_XML:
            return self._parse_cyclonedx_xml(content)
        elif sbom_format == SBOMFormat.SPDX_JSON:
            return self._parse_spdx_json(content)
        elif sbom_format == SBOMFormat.SPDX_TAGVALUE:
            return self._parse_spdx_tagvalue(content)
        else:
            raise ValueError(f"Unknown SBOM format for file: {path}")

    def parse_json(self, content: str) -> DependencyGraph:
        """Parse SBOM from JSON string."""
        try:
            data = json.loads(content)
            if "bomFormat" in data or "components" in data:
                return self._parse_cyclonedx_json(content)
            elif "spdxVersion" in data:
                return self._parse_spdx_json(content)
            else:
                raise ValueError("Cannot determine SBOM format from JSON")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}")

    def _detect_format(self, path: Path, content: str) -> SBOMFormat:
        """Detect the SBOM format from file extension and content."""
        suffix = path.suffix.lower()

        if suffix == ".json":
            try:
                data = json.loads(content)
                if "bomFormat" in data and data["bomFormat"] == "CycloneDX":
                    return SBOMFormat.CYCLONEDX_JSON
                elif "spdxVersion" in data:
                    return SBOMFormat.SPDX_JSON
                # Check for CycloneDX without explicit bomFormat
                if "components" in data and ("specVersion" in data or "version" in data):
                    return SBOMFormat.CYCLONEDX_JSON
            except json.JSONDecodeError:
                pass

        elif suffix == ".xml":
            if "cyclonedx" in content.lower() or "xmlns" in content and "bom" in content:
                return SBOMFormat.CYCLONEDX_XML

        elif suffix in [".spdx", ".tv"]:
            return SBOMFormat.SPDX_TAGVALUE

        # Content-based detection
        if content.strip().startswith("{"):
            try:
                data = json.loads(content)
                if "bomFormat" in data or "components" in data:
                    return SBOMFormat.CYCLONEDX_JSON
                elif "spdxVersion" in data:
                    return SBOMFormat.SPDX_JSON
            except json.JSONDecodeError:
                pass
        elif content.strip().startswith("<"):
            if "cyclonedx" in content.lower():
                return SBOMFormat.CYCLONEDX_XML
        elif "SPDXVersion:" in content:
            return SBOMFormat.SPDX_TAGVALUE

        return SBOMFormat.UNKNOWN

    def _parse_cyclonedx_json(self, content: str) -> DependencyGraph:
        """Parse CycloneDX JSON format."""
        data = json.loads(content)
        graph = DependencyGraph()

        # Extract metadata
        graph.metadata = {
            "format": "CycloneDX",
            "spec_version": data.get("specVersion", "unknown"),
            "serial_number": data.get("serialNumber", ""),
            "version": data.get("version", 1),
        }

        # Parse metadata component (root package)
        if "metadata" in data and "component" in data["metadata"]:
            root_comp = data["metadata"]["component"]
            root_pkg = self._parse_cyclonedx_component(root_comp)
            graph.add_package(root_pkg)
            graph.root_package = root_pkg.id

        # Parse components
        for component in data.get("components", []):
            pkg = self._parse_cyclonedx_component(component)
            self._enrich_with_vulnerabilities(pkg)
            graph.add_package(pkg)

        # Parse dependencies
        for dep_entry in data.get("dependencies", []):
            ref = dep_entry.get("ref", "")
            for dep_ref in dep_entry.get("dependsOn", []):
                graph.add_dependency(
                    Dependency(
                        source=ref,
                        target=dep_ref,
                        dependency_type="direct",
                    )
                )

        # Parse vulnerabilities (CycloneDX 1.4+)
        for vuln in data.get("vulnerabilities", []):
            vuln_info = VulnerabilityInfo(
                cve_id=vuln.get("id", ""),
                description=vuln.get("description", ""),
                source=vuln.get("source", {}).get("name", ""),
            )
            # Find ratings
            for rating in vuln.get("ratings", []):
                if rating.get("method", "").lower() == "cvssv3":
                    vuln_info.cvss_score = rating.get("score", 0.0)
                    vuln_info.cvss_vector = rating.get("vector", "")
                    vuln_info.severity = rating.get("severity", "UNKNOWN")
                    break
            # Link to affected components
            for affect in vuln.get("affects", []):
                ref = affect.get("ref", "")
                if ref in graph.packages:
                    graph.packages[ref].vulnerabilities.append(vuln_info)

        logger.info(
            f"Parsed CycloneDX SBOM: {graph.package_count} packages, "
            f"{graph.dependency_count} dependencies, "
            f"{graph.vulnerability_count} vulnerabilities"
        )

        return graph

    def _parse_cyclonedx_component(self, component: dict) -> Package:
        """Parse a single CycloneDX component."""
        purl = component.get("purl", "")
        name = component.get("name", "")
        version = component.get("version", "")

        # Determine package type from purl
        pkg_type = PackageType.UNKNOWN
        if purl:
            type_match = re.match(r"pkg:(\w+)/", purl)
            if type_match:
                type_str = type_match.group(1).lower()
                try:
                    pkg_type = PackageType(type_str)
                except ValueError:
                    pkg_type = PackageType.UNKNOWN

        # Extract license
        license_str = ""
        if "licenses" in component:
            licenses = component["licenses"]
            if licenses:
                first_lic = licenses[0]
                if "license" in first_lic:
                    license_str = first_lic["license"].get(
                        "id", first_lic["license"].get("name", "")
                    )
                elif "expression" in first_lic:
                    license_str = first_lic["expression"]

        # Extract checksums
        checksum = ""
        for hash_entry in component.get("hashes", []):
            if hash_entry.get("alg", "").upper() == "SHA-256":
                checksum = hash_entry.get("content", "")
                break

        return Package(
            name=name,
            version=version,
            purl=purl,
            package_type=pkg_type,
            license=license_str,
            description=component.get("description", ""),
            supplier=component.get("supplier", {}).get("name", ""),
            checksum=checksum,
            external_refs=[ref.get("url", "") for ref in component.get("externalReferences", [])],
            properties={prop.get("name"): prop.get("value") for prop in component.get("properties", [])},
        )

    def _parse_cyclonedx_xml(self, content: str) -> DependencyGraph:
        """Parse CycloneDX XML format."""
        graph = DependencyGraph()

        # Parse XML
        root = ET.fromstring(content)
        ns = {"cdx": "http://cyclonedx.org/schema/bom/1.4"}

        # Try different namespace versions
        for version in ["1.6", "1.5", "1.4", "1.3", "1.2"]:
            ns["cdx"] = f"http://cyclonedx.org/schema/bom/{version}"
            components = root.findall(".//cdx:component", ns)
            if components:
                break

        # If no namespace works, try without
        if not components:
            components = root.findall(".//component")

        graph.metadata = {"format": "CycloneDX-XML"}

        for comp in components:
            name = comp.findtext("name", default="", namespaces=ns) or comp.findtext("name", default="")
            version = comp.findtext("version", default="", namespaces=ns) or comp.findtext("version", default="")
            purl = comp.findtext("purl", default="", namespaces=ns) or comp.findtext("purl", default="")

            pkg = Package(
                name=name,
                version=version,
                purl=purl,
            )
            self._enrich_with_vulnerabilities(pkg)
            graph.add_package(pkg)

        # Parse dependencies
        deps = root.findall(".//cdx:dependency", ns) or root.findall(".//dependency")
        for dep in deps:
            ref = dep.get("ref", "")
            for sub_dep in dep.findall("cdx:dependency", ns) or dep.findall("dependency"):
                graph.add_dependency(
                    Dependency(source=ref, target=sub_dep.get("ref", ""))
                )

        logger.info(f"Parsed CycloneDX XML: {graph.package_count} packages")
        return graph

    def _parse_spdx_json(self, content: str) -> DependencyGraph:
        """Parse SPDX JSON format."""
        data = json.loads(content)
        graph = DependencyGraph()

        graph.metadata = {
            "format": "SPDX",
            "spdx_version": data.get("spdxVersion", ""),
            "name": data.get("name", ""),
            "document_namespace": data.get("documentNamespace", ""),
        }

        # Parse packages
        for pkg_data in data.get("packages", []):
            spdx_id = pkg_data.get("SPDXID", "")
            name = pkg_data.get("name", "")
            version = pkg_data.get("versionInfo", "")

            # Extract purl from external refs
            purl = ""
            for ext_ref in pkg_data.get("externalRefs", []):
                if ext_ref.get("referenceType", "") == "purl":
                    purl = ext_ref.get("referenceLocator", "")
                    break

            # Extract license
            license_str = pkg_data.get("licenseConcluded", "")
            if license_str == "NOASSERTION":
                license_str = pkg_data.get("licenseDeclared", "")

            pkg = Package(
                name=name,
                version=version,
                purl=purl or spdx_id,
                license=license_str,
                description=pkg_data.get("description", ""),
                supplier=pkg_data.get("supplier", ""),
            )
            self._enrich_with_vulnerabilities(pkg)
            graph.add_package(pkg)

            # Store SPDXID mapping
            pkg.properties["spdx_id"] = spdx_id

        # Parse relationships
        spdx_to_purl = {
            pkg.properties.get("spdx_id", ""): pkg.id
            for pkg in graph.packages.values()
        }

        for rel in data.get("relationships", []):
            rel_type = rel.get("relationshipType", "")
            if rel_type in ["DEPENDS_ON", "CONTAINS", "BUILD_TOOL_OF"]:
                source_id = rel.get("spdxElementId", "")
                target_id = rel.get("relatedSpdxElement", "")

                source = spdx_to_purl.get(source_id, source_id)
                target = spdx_to_purl.get(target_id, target_id)

                dep_type = "direct" if rel_type == "DEPENDS_ON" else "transitive"
                graph.add_dependency(
                    Dependency(source=source, target=target, dependency_type=dep_type)
                )

        logger.info(f"Parsed SPDX JSON: {graph.package_count} packages")
        return graph

    def _parse_spdx_tagvalue(self, content: str) -> DependencyGraph:
        """Parse SPDX Tag-Value format."""
        graph = DependencyGraph()
        graph.metadata = {"format": "SPDX-TagValue"}

        current_pkg: Optional[dict] = None
        packages_by_id: dict[str, Package] = {}

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                continue

            tag, value = line.split(":", 1)
            tag = tag.strip()
            value = value.strip()

            if tag == "PackageName":
                if current_pkg:
                    pkg = Package(
                        name=current_pkg.get("name", ""),
                        version=current_pkg.get("version", ""),
                        purl=current_pkg.get("purl", current_pkg.get("spdx_id", "")),
                        license=current_pkg.get("license", ""),
                    )
                    self._enrich_with_vulnerabilities(pkg)
                    graph.add_package(pkg)
                    if current_pkg.get("spdx_id"):
                        packages_by_id[current_pkg["spdx_id"]] = pkg

                current_pkg = {"name": value}

            elif current_pkg is not None:
                if tag == "SPDXID":
                    current_pkg["spdx_id"] = value
                elif tag == "PackageVersion":
                    current_pkg["version"] = value
                elif tag == "ExternalRef" and "purl" in value.lower():
                    parts = value.split()
                    if len(parts) >= 3:
                        current_pkg["purl"] = parts[2]
                elif tag == "PackageLicenseConcluded":
                    current_pkg["license"] = value

            if tag == "Relationship" and "DEPENDS_ON" in value:
                parts = value.split()
                if len(parts) >= 3:
                    source = parts[0]
                    target = parts[2]
                    graph.add_dependency(
                        Dependency(source=source, target=target, dependency_type="direct")
                    )

        # Add last package
        if current_pkg:
            pkg = Package(
                name=current_pkg.get("name", ""),
                version=current_pkg.get("version", ""),
                purl=current_pkg.get("purl", current_pkg.get("spdx_id", "")),
                license=current_pkg.get("license", ""),
            )
            self._enrich_with_vulnerabilities(pkg)
            graph.add_package(pkg)

        logger.info(f"Parsed SPDX Tag-Value: {graph.package_count} packages")
        return graph

    def _enrich_with_vulnerabilities(self, package: Package) -> None:
        """Enrich package with known vulnerabilities from database."""
        if package.purl and package.purl in self.vuln_db:
            package.vulnerabilities.extend(self.vuln_db[package.purl])
        elif package.id in self.vuln_db:
            package.vulnerabilities.extend(self.vuln_db[package.id])


def create_sample_sbom() -> str:
    """Create a sample CycloneDX SBOM for testing."""
    sample = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "name": "medical-device-app",
                "version": "1.0.0",
                "type": "application",
                "purl": "pkg:npm/medical-device-app@1.0.0",
            }
        },
        "components": [
            {
                "name": "express",
                "version": "4.17.1",
                "purl": "pkg:npm/express@4.17.1",
                "type": "library",
            },
            {
                "name": "lodash",
                "version": "4.17.20",
                "purl": "pkg:npm/lodash@4.17.20",
                "type": "library",
            },
            {
                "name": "axios",
                "version": "0.21.0",
                "purl": "pkg:npm/axios@0.21.0",
                "type": "library",
            },
            {
                "name": "log4j-core",
                "version": "2.14.0",
                "purl": "pkg:maven/org.apache.logging.log4j/log4j-core@2.14.0",
                "type": "library",
            },
        ],
        "dependencies": [
            {
                "ref": "pkg:npm/medical-device-app@1.0.0",
                "dependsOn": [
                    "pkg:npm/express@4.17.1",
                    "pkg:npm/lodash@4.17.20",
                ],
            },
            {
                "ref": "pkg:npm/express@4.17.1",
                "dependsOn": ["pkg:npm/lodash@4.17.20"],
            },
        ],
        "vulnerabilities": [
            {
                "id": "CVE-2021-44228",
                "description": "Log4Shell - Remote code execution vulnerability",
                "source": {"name": "NVD"},
                "ratings": [
                    {
                        "method": "CVSSv3",
                        "score": 10.0,
                        "severity": "critical",
                        "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    }
                ],
                "affects": [
                    {"ref": "pkg:maven/org.apache.logging.log4j/log4j-core@2.14.0"}
                ],
            },
            {
                "id": "CVE-2021-23337",
                "description": "Lodash command injection vulnerability",
                "source": {"name": "NVD"},
                "ratings": [
                    {
                        "method": "CVSSv3",
                        "score": 7.2,
                        "severity": "high",
                    }
                ],
                "affects": [{"ref": "pkg:npm/lodash@4.17.20"}],
            },
        ],
    }
    return json.dumps(sample, indent=2)


if __name__ == "__main__":
    # Test parsing
    sample_sbom = create_sample_sbom()
    parser = SBOMParser()
    graph = parser.parse_json(sample_sbom)

    print(f"[+] Parsed SBOM:")
    print(f"    Packages: {graph.package_count}")
    print(f"    Dependencies: {graph.dependency_count}")
    print(f"    Vulnerabilities: {graph.vulnerability_count}")

    print("\n[+] Packages:")
    for pkg_id, pkg in graph.packages.items():
        vuln_count = len(pkg.vulnerabilities)
        vuln_str = f" [{vuln_count} vulns]" if vuln_count > 0 else ""
        print(f"    - {pkg.name}@{pkg.version}{vuln_str}")

    print("\n[+] Vulnerable packages:")
    for pkg in graph.get_vulnerable_packages():
        for vuln in pkg.vulnerabilities:
            print(f"    - {pkg.name}: {vuln.cve_id} (CVSS: {vuln.cvss_score})")
