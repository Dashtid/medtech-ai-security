"""
Claude Response Processor - Merge Claude.ai Analysis with CVE Data

This script processes JSON responses from Claude.ai that contain
threat intelligence analysis of CVEs, and merges them back into
the original CVE data file.

Workflow:
1. Run NVD scraper to get CVEs and generate Claude prompt
2. Copy prompt to Claude.ai, get JSON response
3. Save response to a file
4. Run this processor to merge analysis into CVE data

Usage:
    python -m medtech_ai_security.threat_intel.claude_processor \
        --cve-file data/threat_intel/cves/medical_devices.json \
        --response-file data/threat_intel/cves/claude_response.json \
        --output data/threat_intel/cves/medical_devices_enriched.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_claude_response(response_file: Path) -> dict:
    """
    Load and parse Claude.ai response JSON.

    Handles both raw JSON and markdown code blocks.

    Args:
        response_file: Path to the response file.

    Returns:
        Parsed JSON data.
    """
    with open(response_file, encoding="utf-8") as f:
        content = f.read().strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        # Extract JSON from code block
        lines = content.split("\n")
        # Find start and end of code block
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if line.startswith("```json") or line.startswith("```"):
                start_idx = i + 1
                break

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break

        content = "\n".join(lines[start_idx:end_idx])

    return json.loads(content)


def merge_analysis(
    cve_file: Path,
    response_file: Path,
    output_file: Path | None = None,
) -> dict:
    """
    Merge Claude.ai analysis into CVE data.

    Args:
        cve_file: Path to original CVE JSON file.
        response_file: Path to Claude.ai response JSON.
        output_file: Path for output file (defaults to enriched version of input).

    Returns:
        Enriched CVE data.
    """
    # Load original CVE data
    with open(cve_file, encoding="utf-8") as f:
        cve_data = json.load(f)

    # Load Claude response
    response = load_claude_response(response_file)
    analyses = response.get("analyses", [])

    # Create lookup by CVE ID
    analysis_lookup = {a["cve_id"]: a for a in analyses}

    # Merge analysis into CVE data
    enriched_count = 0
    for cve in cve_data.get("cves", []):
        cve_id = cve.get("cve_id")
        if cve_id in analysis_lookup:
            analysis = analysis_lookup[cve_id]
            cve["device_type"] = analysis.get("device_type")
            cve["clinical_impact"] = analysis.get("clinical_impact")
            cve["exploitability"] = analysis.get("exploitability")
            cve["remediation"] = analysis.get("remediation")
            cve["ai_reasoning"] = analysis.get("reasoning")
            enriched_count += 1

    # Update metadata
    cve_data["metadata"]["enriched_at"] = datetime.now(timezone.utc).isoformat()
    cve_data["metadata"]["enriched_count"] = enriched_count
    cve_data["metadata"]["analysis_source"] = "Claude.ai"

    # Save enriched data
    if output_file is None:
        output_file = cve_file.parent / (cve_file.stem + "_enriched.json")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cve_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Enriched {enriched_count}/{len(cve_data.get('cves', []))} CVEs")
    logger.info(f"Saved to: {output_file}")

    return cve_data


def generate_summary_report(cve_data: dict, output_file: Path | None = None) -> str:
    """
    Generate a human-readable summary report from enriched CVE data.

    Args:
        cve_data: Enriched CVE data dictionary.
        output_file: Optional path to save the report.

    Returns:
        Report text.
    """
    cves = cve_data.get("cves", [])
    metadata = cve_data.get("metadata", {})

    # Count statistics
    device_types = {}
    clinical_impacts = {}
    exploitability = {}
    severities = {}

    for cve in cves:
        dt = cve.get("device_type") or "unclassified"
        ci = cve.get("clinical_impact") or "unclassified"
        ex = cve.get("exploitability") or "unclassified"
        sev = cve.get("cvss_v3_severity") or "N/A"

        device_types[dt] = device_types.get(dt, 0) + 1
        clinical_impacts[ci] = clinical_impacts.get(ci, 0) + 1
        exploitability[ex] = exploitability.get(ex, 0) + 1
        severities[sev] = severities.get(sev, 0) + 1

    # Build report
    lines = [
        "=" * 70,
        "MEDICAL DEVICE THREAT INTELLIGENCE REPORT",
        "=" * 70,
        "",
        f"Generated: {metadata.get('generated_at', 'N/A')}",
        f"Enriched: {metadata.get('enriched_at', 'N/A')}",
        f"Total CVEs: {len(cves)}",
        f"Enriched with AI Analysis: {metadata.get('enriched_count', 0)}",
        "",
        "-" * 70,
        "SEVERITY DISTRIBUTION",
        "-" * 70,
    ]

    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "N/A"]:
        if sev in severities:
            lines.append(f"  {sev}: {severities[sev]}")

    lines.extend([
        "",
        "-" * 70,
        "DEVICE TYPE DISTRIBUTION",
        "-" * 70,
    ])

    for dt, count in sorted(device_types.items(), key=lambda x: -x[1]):
        lines.append(f"  {dt}: {count}")

    lines.extend([
        "",
        "-" * 70,
        "CLINICAL IMPACT ASSESSMENT",
        "-" * 70,
    ])

    for ci in ["HIGH", "MEDIUM", "LOW", "unclassified"]:
        if ci in clinical_impacts:
            lines.append(f"  {ci}: {clinical_impacts[ci]}")

    lines.extend([
        "",
        "-" * 70,
        "EXPLOITABILITY ASSESSMENT",
        "-" * 70,
    ])

    for ex in ["EASY", "MODERATE", "HARD", "unclassified"]:
        if ex in exploitability:
            lines.append(f"  {ex}: {exploitability[ex]}")

    # High priority CVEs (CRITICAL + HIGH clinical impact + EASY exploit)
    high_priority = [
        cve for cve in cves
        if cve.get("cvss_v3_severity") == "CRITICAL"
        and cve.get("clinical_impact") == "HIGH"
        and cve.get("exploitability") == "EASY"
    ]

    if high_priority:
        lines.extend([
            "",
            "-" * 70,
            "HIGH PRIORITY CVEs (Critical + High Clinical Impact + Easy Exploit)",
            "-" * 70,
        ])
        for cve in high_priority[:10]:
            lines.append(f"  {cve['cve_id']}: {cve.get('description', '')[:80]}...")

    # Top CVEs by severity
    lines.extend([
        "",
        "-" * 70,
        "TOP 10 CVEs BY CVSS SCORE",
        "-" * 70,
    ])

    sorted_cves = sorted(cves, key=lambda x: -(x.get("cvss_v3_score") or 0))
    for cve in sorted_cves[:10]:
        score = cve.get("cvss_v3_score") or "N/A"
        severity = cve.get("cvss_v3_severity") or "N/A"
        device = cve.get("device_type") or "?"
        lines.append(f"  {cve['cve_id']}: {score} ({severity}) [{device}]")
        lines.append(f"    {cve.get('description', '')[:70]}...")

    lines.extend(["", "=" * 70])

    report = "\n".join(lines)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to: {output_file}")

    return report


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Claude.ai analysis and merge with CVE data"
    )
    parser.add_argument(
        "--cve-file",
        type=str,
        required=True,
        help="Path to original CVE JSON file",
    )
    parser.add_argument(
        "--response-file",
        type=str,
        required=True,
        help="Path to Claude.ai response JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for enriched CVE file",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate summary report",
    )

    args = parser.parse_args()

    cve_file = Path(args.cve_file)
    response_file = Path(args.response_file)
    output_file = Path(args.output) if args.output else None

    # Merge analysis
    enriched_data = merge_analysis(cve_file, response_file, output_file)

    # Generate report if requested
    if args.report:
        report_path = (output_file or cve_file).parent / "threat_intel_report.txt"
        report = generate_summary_report(enriched_data, report_path)
        print(report)


if __name__ == "__main__":
    main()
