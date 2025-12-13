#!/usr/bin/env python3
"""Performance profiling script for MedTech AI Security.

This script profiles the main components of the platform and generates
performance reports with CPU and memory analysis.

Usage:
    python scripts/profile_performance.py --module anomaly
    python scripts/profile_performance.py --module sbom --memory
    python scripts/profile_performance.py --all --output profile_report.html
"""

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def profile_anomaly_detection() -> dict[str, Any]:
    """Profile anomaly detection module."""
    from medtech_ai_security.anomaly.detector import (
        AnomalyDetector,
        preprocess_traffic,
    )

    results = {"module": "anomaly_detection", "operations": {}}

    # Generate test data
    np.random.seed(42)
    small_data = np.random.randn(1000, 20).astype(np.float32)
    large_data = np.random.randn(10000, 20).astype(np.float32)

    # Profile preprocessing
    start = time.perf_counter()
    for _ in range(100):
        preprocess_traffic(small_data)
    elapsed = (time.perf_counter() - start) / 100
    results["operations"]["preprocess_1k"] = {
        "avg_time_ms": elapsed * 1000,
        "throughput_samples_per_sec": 1000 / elapsed,
    }

    # Profile detector training
    detector = AnomalyDetector()
    start = time.perf_counter()
    detector.fit(small_data)
    results["operations"]["train_1k"] = {
        "time_ms": (time.perf_counter() - start) * 1000,
    }

    # Profile inference
    start = time.perf_counter()
    for _ in range(100):
        detector.predict(small_data)
    elapsed = (time.perf_counter() - start) / 100
    results["operations"]["predict_1k"] = {
        "avg_time_ms": elapsed * 1000,
        "throughput_samples_per_sec": 1000 / elapsed,
    }

    # Large dataset inference
    start = time.perf_counter()
    detector.predict(large_data)
    results["operations"]["predict_10k"] = {
        "time_ms": (time.perf_counter() - start) * 1000,
        "throughput_samples_per_sec": 10000 / (time.perf_counter() - start),
    }

    return results


def profile_sbom_analysis() -> dict[str, Any]:
    """Profile SBOM analysis module."""
    from medtech_ai_security.sbom_analysis.analyzer import (
        DependencyGraphBuilder,
        RiskScorer,
        SBOMParser,
    )

    results = {"module": "sbom_analysis", "operations": {}}

    # Generate synthetic SBOM
    def generate_sbom(n: int) -> dict:
        import random

        random.seed(42)
        components = [
            {
                "type": "library",
                "name": f"pkg-{i}",
                "version": f"{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 10)}",
                "purl": f"pkg:pypi/pkg-{i}",
            }
            for i in range(n)
        ]
        return {"bomFormat": "CycloneDX", "specVersion": "1.5", "components": components}

    small_sbom = generate_sbom(50)
    medium_sbom = generate_sbom(200)
    large_sbom = generate_sbom(500)

    # Profile graph building
    builder = DependencyGraphBuilder()

    start = time.perf_counter()
    for _ in range(10):
        builder.build_from_cyclonedx(small_sbom)
    results["operations"]["build_graph_50"] = {
        "avg_time_ms": (time.perf_counter() - start) / 10 * 1000,
    }

    start = time.perf_counter()
    builder.build_from_cyclonedx(medium_sbom)
    results["operations"]["build_graph_200"] = {
        "time_ms": (time.perf_counter() - start) * 1000,
    }

    start = time.perf_counter()
    builder.build_from_cyclonedx(large_sbom)
    results["operations"]["build_graph_500"] = {
        "time_ms": (time.perf_counter() - start) * 1000,
    }

    # Profile risk scoring
    scorer = RiskScorer()

    start = time.perf_counter()
    for _ in range(10):
        scorer.score_components(medium_sbom["components"])
    results["operations"]["score_components_200"] = {
        "avg_time_ms": (time.perf_counter() - start) / 10 * 1000,
    }

    return results


def profile_adversarial_ml() -> dict[str, Any]:
    """Profile adversarial ML module."""
    from medtech_ai_security.adversarial.evaluator import (
        RobustnessEvaluator,
        create_simple_classifier,
        fgsm_attack,
        pgd_attack,
    )

    results = {"module": "adversarial_ml", "operations": {}}

    # Generate test data
    np.random.seed(42)
    images = np.random.rand(32, 28, 28, 1).astype(np.float32)
    labels = np.random.randint(0, 10, size=32)

    # Create model
    model = create_simple_classifier(input_shape=(28, 28, 1), num_classes=10)

    # Profile FGSM
    start = time.perf_counter()
    for _ in range(10):
        fgsm_attack(model, images, labels, epsilon=0.03)
    results["operations"]["fgsm_32_images"] = {
        "avg_time_ms": (time.perf_counter() - start) / 10 * 1000,
        "throughput_images_per_sec": 32 / ((time.perf_counter() - start) / 10),
    }

    # Profile PGD (10 steps)
    start = time.perf_counter()
    for _ in range(5):
        pgd_attack(model, images, labels, epsilon=0.1, steps=10, step_size=0.01)
    results["operations"]["pgd_10steps_32_images"] = {
        "avg_time_ms": (time.perf_counter() - start) / 5 * 1000,
        "throughput_images_per_sec": 32 / ((time.perf_counter() - start) / 5),
    }

    # Profile PGD (40 steps)
    start = time.perf_counter()
    pgd_attack(model, images[:16], labels[:16], epsilon=0.1, steps=40, step_size=0.005)
    results["operations"]["pgd_40steps_16_images"] = {
        "time_ms": (time.perf_counter() - start) * 1000,
    }

    # Profile full evaluation
    evaluator = RobustnessEvaluator(model)
    start = time.perf_counter()
    evaluator.evaluate(images[:16], labels[:16], attacks=["fgsm"])
    results["operations"]["full_eval_fgsm_16"] = {
        "time_ms": (time.perf_counter() - start) * 1000,
    }

    return results


def profile_threat_intel() -> dict[str, Any]:
    """Profile threat intelligence module."""
    from medtech_ai_security.threat_intel.nvd_scraper import NVDScraper

    results = {"module": "threat_intel", "operations": {}}

    # Profile CVE parsing (if sample data available)
    sample_cve = {
        "cve": {
            "id": "CVE-2025-12345",
            "descriptions": [{"lang": "en", "value": "Test vulnerability"}],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "cvssData": {
                            "baseScore": 8.5,
                            "baseSeverity": "HIGH",
                            "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                        }
                    }
                ]
            },
        }
    }

    scraper = NVDScraper()

    start = time.perf_counter()
    for _ in range(1000):
        scraper.parse_cve(sample_cve)
    results["operations"]["parse_cve_single"] = {
        "avg_time_ms": (time.perf_counter() - start) / 1000 * 1000,
        "throughput_cves_per_sec": 1000 / (time.perf_counter() - start),
    }

    # Profile batch parsing
    batch = [sample_cve] * 100
    start = time.perf_counter()
    for _ in range(10):
        [scraper.parse_cve(cve) for cve in batch]
    results["operations"]["parse_cve_batch_100"] = {
        "avg_time_ms": (time.perf_counter() - start) / 10 * 1000,
        "throughput_cves_per_sec": 100 / ((time.perf_counter() - start) / 10),
    }

    return results


def profile_with_memory(func, name: str) -> dict[str, Any]:
    """Profile function with memory tracking."""
    try:
        from memory_profiler import memory_usage
    except ImportError:
        print("[!] memory_profiler not installed. Skipping memory profiling.")
        return func()

    # Measure memory
    mem_usage, result = memory_usage(
        (func, (), {}), interval=0.1, retval=True, include_children=True
    )

    result["memory"] = {
        "peak_mb": max(mem_usage),
        "start_mb": mem_usage[0],
        "increase_mb": max(mem_usage) - mem_usage[0],
    }

    return result


def run_cprofile(func, name: str) -> str:
    """Run cProfile on a function and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    return stream.getvalue()


def generate_html_report(results: list[dict], cprofile_stats: dict[str, str]) -> str:
    """Generate HTML performance report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>MedTech AI Security - Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .module { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background: #007bff; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .metric { font-weight: bold; color: #28a745; }
        .timestamp { color: #888; font-size: 0.9em; }
        pre { background: #282c34; color: #abb2bf; padding: 15px; overflow-x: auto; border-radius: 4px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .summary-card { background: white; padding: 15px; border-radius: 8px; text-align: center; }
        .summary-value { font-size: 2em; color: #007bff; font-weight: bold; }
        .summary-label { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Performance Profiling Report</h1>
    <p class="timestamp">Generated: """ + datetime.now().isoformat() + """</p>
"""

    # Summary cards
    total_ops = sum(len(r.get("operations", {})) for r in results)
    html += f"""
    <div class="summary">
        <div class="summary-card">
            <div class="summary-value">{len(results)}</div>
            <div class="summary-label">Modules Profiled</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{total_ops}</div>
            <div class="summary-label">Operations Measured</div>
        </div>
    </div>
"""

    # Module results
    for result in results:
        html += f"""
    <div class="module">
        <h2>{result['module'].replace('_', ' ').title()}</h2>
        <table>
            <tr>
                <th>Operation</th>
                <th>Time (ms)</th>
                <th>Throughput</th>
            </tr>
"""
        for op, metrics in result.get("operations", {}).items():
            time_val = metrics.get("avg_time_ms", metrics.get("time_ms", "N/A"))
            if isinstance(time_val, float):
                time_val = f"{time_val:.2f}"

            throughput = "N/A"
            for key, val in metrics.items():
                if "throughput" in key:
                    throughput = f"{val:.0f} {key.replace('throughput_', '').replace('_', '/')}"
                    break

            html += f"""
            <tr>
                <td>{op}</td>
                <td class="metric">{time_val}</td>
                <td>{throughput}</td>
            </tr>
"""
        html += """
        </table>
"""

        if "memory" in result:
            html += f"""
        <h3>Memory Usage</h3>
        <p>Peak: <span class="metric">{result['memory']['peak_mb']:.1f} MB</span> |
           Increase: <span class="metric">{result['memory']['increase_mb']:.1f} MB</span></p>
"""
        html += """
    </div>
"""

    # cProfile stats
    if cprofile_stats:
        html += """
    <h2>Detailed CPU Profile</h2>
"""
        for name, stats in cprofile_stats.items():
            html += f"""
    <div class="module">
        <h3>{name}</h3>
        <pre>{stats}</pre>
    </div>
"""

    html += """
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Profile MedTech AI Security performance"
    )
    parser.add_argument(
        "--module",
        choices=["anomaly", "sbom", "adversarial", "threat_intel"],
        help="Module to profile",
    )
    parser.add_argument(
        "--all", action="store_true", help="Profile all modules"
    )
    parser.add_argument(
        "--memory", action="store_true", help="Include memory profiling"
    )
    parser.add_argument(
        "--cprofile", action="store_true", help="Include detailed CPU profile"
    )
    parser.add_argument(
        "--output", type=str, help="Output file (HTML or JSON)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    args = parser.parse_args()

    if not args.module and not args.all:
        parser.print_help()
        sys.exit(1)

    modules = {
        "anomaly": profile_anomaly_detection,
        "sbom": profile_sbom_analysis,
        "adversarial": profile_adversarial_ml,
        "threat_intel": profile_threat_intel,
    }

    results = []
    cprofile_stats = {}

    if args.all:
        to_profile = modules.keys()
    else:
        to_profile = [args.module]

    for name in to_profile:
        print(f"[*] Profiling {name}...")
        func = modules[name]

        try:
            if args.memory:
                result = profile_with_memory(func, name)
            else:
                result = func()

            if args.cprofile:
                cprofile_stats[name] = run_cprofile(func, name)

            results.append(result)
            print(f"[+] {name} complete")
        except Exception as e:
            print(f"[-] Error profiling {name}: {e}")
            results.append({"module": name, "error": str(e)})

    # Output results
    if args.json or (args.output and args.output.endswith(".json")):
        output = json.dumps(results, indent=2)
    else:
        output = generate_html_report(results, cprofile_stats)

    if args.output:
        Path(args.output).write_text(output)
        print(f"[+] Report saved to {args.output}")
    else:
        if args.json:
            print(output)
        else:
            # Print summary to console
            print("\n" + "=" * 60)
            print("PERFORMANCE SUMMARY")
            print("=" * 60)
            for result in results:
                print(f"\n{result['module'].upper()}")
                print("-" * 40)
                for op, metrics in result.get("operations", {}).items():
                    time_val = metrics.get("avg_time_ms", metrics.get("time_ms", "N/A"))
                    if isinstance(time_val, float):
                        print(f"  {op}: {time_val:.2f} ms")


if __name__ == "__main__":
    main()
