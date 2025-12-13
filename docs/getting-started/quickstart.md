# Quick Start

This guide walks you through running your first security analysis.

## Run the Demo

The easiest way to see all modules in action:

```bash
uv run python scripts/demo_security.py
```

This demonstrates:

1. Threat intelligence gathering
2. ML risk scoring
3. Anomaly detection
4. Adversarial ML testing
5. SBOM analysis

## CLI Tools

All modules are accessible via command-line tools:

### Phase 1: Threat Intelligence

```bash
# Scrape NVD for medical device CVEs (last 30 days)
medsec-nvd --days 30 --output data/nvd_cves.json

# Scrape CISA ICS-CERT advisories
medsec-cisa --output data/cisa_advisories.json

# Enrich CVEs with Claude AI analysis
medsec-enrich --input data/nvd_cves.json --output data/enriched.json
```

### Phase 2: ML Risk Scoring

```bash
# Train the risk scoring model
medsec-risk train --data data/enriched.json

# Score new vulnerabilities
medsec-risk predict --input new_cves.json
```

### Phase 3: Anomaly Detection

```bash
# Generate synthetic traffic for training
medsec-traffic-gen --normal 1000 --attack 100

# Train the anomaly detector
medsec-anomaly train

# Detect anomalies in traffic
medsec-anomaly detect --input traffic.csv
```

### Phase 4: Adversarial ML

```bash
# Evaluate model robustness
medsec-adversarial evaluate --model models/classifier.keras

# Run specific attack
medsec-adversarial attack --method fgsm --epsilon 0.1
```

### Phase 5: SBOM Analysis

```bash
# Analyze an SBOM file
medsec-sbom analyze --input sbom.json --output report.html

# Run demo with sample SBOM
medsec-sbom demo --html
```

## Python API

Use the modules programmatically:

```python
from medtech_ai_security.sbom_analyzer import SBOMAnalyzer, RiskScorer
from medtech_ai_security.anomaly import AnomalyDetector
from medtech_ai_security.adversarial import AdversarialAttacker

# Analyze SBOM
analyzer = SBOMAnalyzer()
result = analyzer.analyze("sbom.json")
print(f"Risk Score: {result.risk_score}")

# Detect anomalies
detector = AnomalyDetector()
detector.fit(normal_traffic)
anomalies = detector.predict(new_traffic)

# Test adversarial robustness
attacker = AdversarialAttacker(model)
adversarial_examples = attacker.fgsm_attack(images, labels)
```

## Next Steps

- [Configuration](configuration.md) - Set up API keys
- [Modules Overview](../modules/overview.md) - Deep dive into each module
- [API Reference](../api/overview.md) - Full API documentation
