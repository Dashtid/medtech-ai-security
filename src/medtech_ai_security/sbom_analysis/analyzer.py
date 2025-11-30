"""
SBOM Analyzer - Main entry point for SBOM analysis.

Provides comprehensive analysis including:
- SBOM parsing and validation
- GNN-based vulnerability prediction
- Supply chain risk scoring
- Dependency visualization
- CLI interface

For medical device cybersecurity per FDA and EU MDR requirements.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from medtech_ai_security.sbom_analysis.graph_builder import (
    GraphData,
    SBOMGraphBuilder,
)
from medtech_ai_security.sbom_analysis.parser import (
    DependencyGraph,
    SBOMParser,
    create_sample_sbom,
)
from medtech_ai_security.sbom_analysis.risk_scorer import (
    RiskLevel,
    RiskReport,
    SupplyChainRiskScorer,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    """Complete SBOM analysis report."""

    # Source info
    sbom_file: str = ""
    sbom_format: str = ""

    # Risk assessment
    risk_report: RiskReport | None = None

    # GNN predictions (if model available)
    gnn_predictions: dict[str, Any] | None = None

    # Graph statistics
    graph_stats: dict[str, Any] = field(default_factory=dict)

    # Visualization data
    visualization_data: dict[str, Any] | None = None

    # Raw data
    dependency_graph: DependencyGraph | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "sbom_file": self.sbom_file,
            "sbom_format": self.sbom_format,
            "graph_stats": self.graph_stats,
        }

        if self.risk_report:
            result["risk_report"] = self.risk_report.to_dict()

        if self.gnn_predictions:
            result["gnn_predictions"] = self.gnn_predictions

        if self.visualization_data:
            result["visualization"] = self.visualization_data

        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class SBOMAnalyzer:
    """Main analyzer for SBOM supply chain security."""

    def __init__(
        self,
        use_gnn: bool = True,
        medical_context: bool = True,
        vuln_db: dict | None = None,
    ):
        """Initialize analyzer.

        Args:
            use_gnn: Whether to use GNN for predictions
            medical_context: Apply medical device context to risk scoring
            vuln_db: Optional vulnerability database
        """
        self.use_gnn = use_gnn
        self.medical_context = medical_context
        self.parser = SBOMParser(vuln_db=vuln_db)
        self.graph_builder = SBOMGraphBuilder()
        self.risk_scorer = SupplyChainRiskScorer(medical_context=medical_context)
        self.gnn_model = None

        # Try to load GNN model if available
        if use_gnn:
            self._load_gnn_model()

    def _load_gnn_model(self) -> None:
        """Load pre-trained GNN model if available."""
        try:
            from medtech_ai_security.sbom_analysis.gnn_model import (
                TF_AVAILABLE,
                GNNConfig,
                VulnerabilityGNN,
            )

            if TF_AVAILABLE:
                config = GNNConfig()
                self.gnn_model = VulnerabilityGNN(config)
                logger.info("GNN model initialized")
            else:
                logger.warning("TensorFlow not available, GNN disabled")
        except ImportError as e:
            logger.warning(f"Could not load GNN model: {e}")

    def analyze(self, sbom_path: str | Path) -> AnalysisReport:
        """Perform complete SBOM analysis.

        Args:
            sbom_path: Path to SBOM file

        Returns:
            Complete analysis report
        """
        path = Path(sbom_path)
        report = AnalysisReport(sbom_file=str(path))

        # Parse SBOM
        logger.info(f"Parsing SBOM: {path}")
        dep_graph = self.parser.parse(path)
        report.dependency_graph = dep_graph
        report.sbom_format = dep_graph.metadata.get("format", "unknown")

        # Build graph representation
        logger.info("Building graph representation")
        graph_data = self.graph_builder.build(dep_graph)

        # Graph statistics
        report.graph_stats = {
            "num_packages": graph_data.num_nodes,
            "num_dependencies": graph_data.num_edges,
            "feature_dim": graph_data.node_features.shape[1] if graph_data.num_nodes > 0 else 0,
        }

        # Risk scoring
        logger.info("Calculating risk scores")
        report.risk_report = self.risk_scorer.score(dep_graph)

        # GNN predictions (if model available and trained)
        if self.gnn_model is not None:
            logger.info("Running GNN predictions")
            report.gnn_predictions = self._run_gnn_predictions(graph_data, dep_graph)

        # Generate visualization data
        logger.info("Generating visualization data")
        report.visualization_data = self._generate_visualization(dep_graph, report.risk_report)

        return report

    def analyze_json(self, sbom_content: str) -> AnalysisReport:
        """Analyze SBOM from JSON string.

        Args:
            sbom_content: SBOM JSON content

        Returns:
            Complete analysis report
        """
        report = AnalysisReport(sbom_file="<inline>")

        # Parse SBOM
        dep_graph = self.parser.parse_json(sbom_content)
        report.dependency_graph = dep_graph
        report.sbom_format = dep_graph.metadata.get("format", "unknown")

        # Build graph
        graph_data = self.graph_builder.build(dep_graph)
        report.graph_stats = {
            "num_packages": graph_data.num_nodes,
            "num_dependencies": graph_data.num_edges,
        }

        # Risk scoring
        report.risk_report = self.risk_scorer.score(dep_graph)

        # GNN predictions
        if self.gnn_model is not None:
            report.gnn_predictions = self._run_gnn_predictions(graph_data, dep_graph)

        # Visualization
        report.visualization_data = self._generate_visualization(dep_graph, report.risk_report)

        return report

    def _run_gnn_predictions(
        self, graph_data: GraphData, dep_graph: DependencyGraph
    ) -> dict[str, Any]:
        """Run GNN model for predictions."""
        try:
            probs = self.gnn_model.predict_proba(graph_data)
            predictions = np.argmax(probs, axis=-1)

            result = {
                "predictions": {},
                "summary": {
                    "predicted_vulnerable": int(np.sum(predictions == 1)),
                    "predicted_transitive": int(np.sum(predictions == 2)),
                    "predicted_clean": int(np.sum(predictions == 0)),
                },
            }

            for i, node_id in enumerate(graph_data.node_ids):
                result["predictions"][node_id] = {
                    "label": int(predictions[i]),
                    "label_name": ["clean", "vulnerable", "transitive"][predictions[i]],
                    "probabilities": {
                        "clean": float(probs[i, 0]),
                        "vulnerable": float(probs[i, 1]),
                        "transitive": float(probs[i, 2]),
                    },
                }

            return result

        except Exception as e:
            logger.error(f"GNN prediction failed: {e}")
            return {"error": str(e)}

    def _generate_visualization(
        self, dep_graph: DependencyGraph, risk_report: RiskReport
    ) -> dict[str, Any]:
        """Generate visualization-ready data structure.

        Returns data compatible with D3.js force-directed graph.
        """
        # Build risk lookup
        risk_lookup = {pr.package_id: pr for pr in risk_report.package_risks}

        # Create nodes
        nodes = []
        for pkg_id, pkg in dep_graph.packages.items():
            pkg_risk = risk_lookup.get(pkg_id)

            node = {
                "id": pkg_id,
                "name": pkg.name,
                "version": pkg.version,
                "ecosystem": pkg.ecosystem,
                "is_root": pkg_id == dep_graph.root_package,
                "vulnerability_count": len(pkg.vulnerabilities),
                "risk_level": pkg_risk.risk_level.value if pkg_risk else "info",
                "risk_score": pkg_risk.risk_score if pkg_risk else 0.0,
            }

            # Color coding based on risk
            if pkg_risk:
                if pkg_risk.risk_level == RiskLevel.CRITICAL:
                    node["color"] = "#dc3545"  # Red
                elif pkg_risk.risk_level == RiskLevel.HIGH:
                    node["color"] = "#fd7e14"  # Orange
                elif pkg_risk.risk_level == RiskLevel.MEDIUM:
                    node["color"] = "#ffc107"  # Yellow
                elif pkg_risk.risk_level == RiskLevel.LOW:
                    node["color"] = "#28a745"  # Green
                else:
                    node["color"] = "#6c757d"  # Gray
            else:
                node["color"] = "#6c757d"

            # Size based on dependents
            node["size"] = 5 + min(len(dep_graph.get_dependents(pkg_id)) * 2, 20)

            nodes.append(node)

        # Create links (edges)
        links = []
        for dep in dep_graph.dependencies:
            links.append({
                "source": dep.source,
                "target": dep.target,
                "type": dep.dependency_type,
            })

        # Cluster by ecosystem
        ecosystems = {}
        for node in nodes:
            eco = node.get("ecosystem", "unknown")
            if eco not in ecosystems:
                ecosystems[eco] = []
            ecosystems[eco].append(node["id"])

        return {
            "nodes": nodes,
            "links": links,
            "ecosystems": ecosystems,
            "statistics": {
                "total_nodes": len(nodes),
                "total_links": len(links),
                "ecosystems": list(ecosystems.keys()),
            },
        }

    def generate_html_report(
        self, report: AnalysisReport, output_path: str | None = None
    ) -> str:
        """Generate an HTML visualization report.

        Args:
            report: Analysis report
            output_path: Optional path to save HTML file

        Returns:
            HTML content
        """
        viz_data = json.dumps(report.visualization_data) if report.visualization_data else "{}"
        _risk_data = json.dumps(report.risk_report.to_dict()) if report.risk_report else "{}"  # noqa: F841

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SBOM Analysis Report</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #fff; margin-bottom: 15px; }}
        .header {{ text-align: center; padding: 30px 0; border-bottom: 1px solid #333; }}
        .risk-badge {{ display: inline-block; padding: 8px 16px; border-radius: 4px; font-weight: bold; text-transform: uppercase; }}
        .risk-critical {{ background: #dc3545; }}
        .risk-high {{ background: #fd7e14; }}
        .risk-medium {{ background: #ffc107; color: #000; }}
        .risk-low {{ background: #28a745; }}
        .risk-info {{ background: #6c757d; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: #16213e; border-radius: 8px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #00d4ff; }}
        .stat-label {{ color: #aaa; margin-top: 5px; }}
        .graph-container {{ background: #16213e; border-radius: 8px; padding: 20px; margin: 30px 0; }}
        #graph {{ width: 100%; height: 500px; }}
        .node-tooltip {{ position: absolute; background: #0f0f1a; border: 1px solid #333; border-radius: 4px; padding: 10px; pointer-events: none; font-size: 12px; z-index: 100; }}
        .recommendations {{ background: #16213e; border-radius: 8px; padding: 20px; margin: 30px 0; }}
        .rec-item {{ padding: 10px 0; border-bottom: 1px solid #333; }}
        .rec-item:last-child {{ border-bottom: none; }}
        .packages-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .packages-table th, .packages-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        .packages-table th {{ background: #0f0f1a; color: #00d4ff; }}
        .packages-table tr:hover {{ background: #1f2937; }}
        .legend {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SBOM Supply Chain Analysis</h1>
            <p style="color: #aaa; margin-top: 10px;">Medical Device Cybersecurity Assessment</p>
            <div style="margin-top: 20px;">
                <span class="risk-badge risk-{report.risk_report.overall_risk_level.value if report.risk_report else 'info'}">
                    {report.risk_report.overall_risk_level.value.upper() if report.risk_report else 'N/A'} RISK
                </span>
                <span style="margin-left: 10px; color: #aaa;">
                    Score: {report.risk_report.overall_risk_score:.1f if report.risk_report else 0}/100
                </span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{report.graph_stats.get('num_packages', 0)}</div>
                <div class="stat-label">Total Packages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{report.risk_report.vulnerable_packages if report.risk_report else 0}</div>
                <div class="stat-label">Vulnerable Packages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{report.risk_report.total_vulnerabilities if report.risk_report else 0}</div>
                <div class="stat-label">Total Vulnerabilities</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{report.risk_report.critical_vulnerabilities if report.risk_report else 0}</div>
                <div class="stat-label">Critical CVEs</div>
            </div>
        </div>

        <div class="graph-container">
            <h2>Dependency Graph</h2>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #dc3545;"></div> Critical</div>
                <div class="legend-item"><div class="legend-color" style="background: #fd7e14;"></div> High</div>
                <div class="legend-item"><div class="legend-color" style="background: #ffc107;"></div> Medium</div>
                <div class="legend-item"><div class="legend-color" style="background: #28a745;"></div> Low</div>
                <div class="legend-item"><div class="legend-color" style="background: #6c757d;"></div> Info</div>
            </div>
            <div id="graph"></div>
        </div>

        <div class="recommendations">
            <h2>Recommendations</h2>
            {''.join(f'<div class="rec-item">{rec}</div>' for rec in (report.risk_report.recommendations if report.risk_report else []))}
        </div>

        <div class="packages-section">
            <h2>Package Details</h2>
            <table class="packages-table">
                <thead>
                    <tr>
                        <th>Package</th>
                        <th>Version</th>
                        <th>Risk Level</th>
                        <th>Risk Score</th>
                        <th>Vulnerabilities</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''<tr>
                        <td>{pr.package_name}</td>
                        <td>{pr.package_version}</td>
                        <td><span class="risk-badge risk-{pr.risk_level.value}">{pr.risk_level.value}</span></td>
                        <td>{pr.risk_score:.1f}</td>
                        <td>{len(pr.vulnerabilities)}</td>
                    </tr>''' for pr in sorted((report.risk_report.package_risks if report.risk_report else []), key=lambda x: x.risk_score, reverse=True))}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const vizData = {viz_data};

        if (vizData.nodes && vizData.nodes.length > 0) {{
            const width = document.getElementById('graph').clientWidth;
            const height = 500;

            const svg = d3.select('#graph')
                .append('svg')
                .attr('width', width)
                .attr('height', height);

            const simulation = d3.forceSimulation(vizData.nodes)
                .force('link', d3.forceLink(vizData.links).id(d => d.id).distance(80))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2));

            const link = svg.append('g')
                .selectAll('line')
                .data(vizData.links)
                .join('line')
                .attr('stroke', '#444')
                .attr('stroke-width', 1);

            const node = svg.append('g')
                .selectAll('circle')
                .data(vizData.nodes)
                .join('circle')
                .attr('r', d => d.size)
                .attr('fill', d => d.color)
                .attr('stroke', '#fff')
                .attr('stroke-width', d => d.is_root ? 3 : 1)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            node.append('title')
                .text(d => `${{d.name}}@${{d.version}}\\nRisk: ${{d.risk_level}} (${{d.risk_score.toFixed(1)}})\\nVulnerabilities: ${{d.vulnerability_count}}`);

            simulation.on('tick', () => {{
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
            }});

            function dragstarted(event) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }}

            function dragged(event) {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }}

            function dragended(event) {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }}
        }}
    </script>
</body>
</html>"""

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info(f"HTML report saved to {output_path}")

        return html


def main():
    """CLI entry point for SBOM analysis."""
    parser = argparse.ArgumentParser(
        description="SBOM Supply Chain Security Analyzer for Medical Devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  medsec-sbom analyze sbom.json
  medsec-sbom analyze sbom.json --output report.json
  medsec-sbom analyze sbom.json --html report.html
  medsec-sbom demo
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze an SBOM file")
    analyze_parser.add_argument("sbom_file", help="Path to SBOM file (CycloneDX or SPDX)")
    analyze_parser.add_argument(
        "--output", "-o", help="Output file for JSON report"
    )
    analyze_parser.add_argument(
        "--html", help="Generate HTML visualization report"
    )
    analyze_parser.add_argument(
        "--no-gnn", action="store_true", help="Disable GNN predictions"
    )
    analyze_parser.add_argument(
        "--no-medical", action="store_true", help="Disable medical device context"
    )
    analyze_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample SBOM")
    demo_parser.add_argument(
        "--html", help="Generate HTML report to file"
    )

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse and display SBOM structure")
    parse_parser.add_argument("sbom_file", help="Path to SBOM file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(message)s",
    )

    if args.command == "analyze":
        # Analyze SBOM
        print(f"[+] Analyzing SBOM: {args.sbom_file}")

        analyzer = SBOMAnalyzer(
            use_gnn=not args.no_gnn,
            medical_context=not args.no_medical,
        )

        try:
            report = analyzer.analyze(args.sbom_file)
        except FileNotFoundError:
            print(f"[-] File not found: {args.sbom_file}")
            sys.exit(1)
        except Exception as e:
            print(f"[-] Analysis failed: {e}")
            sys.exit(1)

        # Print summary
        if report.risk_report:
            print(f"\n{report.risk_report.summary}")

            print("\n[+] Top Risk Packages:")
            for pkg in sorted(
                report.risk_report.package_risks,
                key=lambda x: x.risk_score,
                reverse=True,
            )[:5]:
                print(
                    f"    {pkg.package_name}@{pkg.package_version}: "
                    f"{pkg.risk_level.value.upper()} ({pkg.risk_score:.1f})"
                )

            print("\n[+] Recommendations:")
            for rec in report.risk_report.recommendations:
                print(f"    {rec}")

        # Save outputs
        if args.output:
            Path(args.output).write_text(report.to_json(), encoding="utf-8")
            print(f"\n[+] JSON report saved to: {args.output}")

        if args.html:
            analyzer.generate_html_report(report, args.html)
            print(f"[+] HTML report saved to: {args.html}")

    elif args.command == "demo":
        # Run demo
        print("[+] Running SBOM Analysis Demo")
        print("=" * 50)

        sample_sbom = create_sample_sbom()
        analyzer = SBOMAnalyzer(use_gnn=False)  # GNN needs training data
        report = analyzer.analyze_json(sample_sbom)

        if report.risk_report:
            print(f"\n{report.risk_report.summary}")

            print("\n[+] Package Risks:")
            for pkg in sorted(
                report.risk_report.package_risks,
                key=lambda x: x.risk_score,
                reverse=True,
            ):
                print(
                    f"    {pkg.package_name}@{pkg.package_version}: "
                    f"{pkg.risk_level.value.upper()} ({pkg.risk_score:.1f})"
                )
                for rec in pkg.recommendations:
                    print(f"        -> {rec}")

            print("\n[+] Recommendations:")
            for rec in report.risk_report.recommendations:
                print(f"    {rec}")

            print("\n[+] FDA Compliance Notes:")
            for note in report.risk_report.fda_compliance_notes:
                print(f"    {note}")

        if args.html:
            analyzer.generate_html_report(report, args.html)
            print(f"\n[+] HTML report saved to: {args.html}")

        print("\n[OK] Demo complete!")

    elif args.command == "parse":
        # Parse and display
        print(f"[+] Parsing SBOM: {args.sbom_file}")

        parser_inst = SBOMParser()
        try:
            graph = parser_inst.parse(args.sbom_file)
        except Exception as e:
            print(f"[-] Parse failed: {e}")
            sys.exit(1)

        print(f"\n[+] SBOM Format: {graph.metadata.get('format', 'unknown')}")
        print(f"    Packages: {graph.package_count}")
        print(f"    Dependencies: {graph.dependency_count}")
        print(f"    Vulnerabilities: {graph.vulnerability_count}")

        print("\n[+] Packages:")
        for pkg in graph.packages.values():
            vuln_str = f" [{len(pkg.vulnerabilities)} vulns]" if pkg.vulnerabilities else ""
            print(f"    - {pkg.name}@{pkg.version}{vuln_str}")

        if graph.get_vulnerable_packages():
            print("\n[+] Vulnerabilities:")
            for pkg in graph.get_vulnerable_packages():
                for vuln in pkg.vulnerabilities:
                    print(f"    {pkg.name}: {vuln.cve_id} (CVSS: {vuln.cvss_score})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
