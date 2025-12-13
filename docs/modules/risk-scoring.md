# ML Risk Scoring Module

Phase 2 of the MedTech AI Security platform. Uses machine learning to prioritize
vulnerabilities based on medical device context.

## Overview

The ML Risk Scoring module predicts vulnerability risk priority using an ensemble
classifier trained on medical device CVE data.

## Model Architecture

```
Input Features (12):
  - CVSS Base Score (normalized)
  - Attack Vector (one-hot)
  - Attack Complexity
  - Privileges Required
  - User Interaction
  - Scope
  - Confidentiality Impact
  - Integrity Impact
  - Availability Impact
  - Device Type (encoded)
  - CWE Category
  - Clinical Impact Score

Ensemble:
  - Naive Bayes (probability estimation)
  - K-Nearest Neighbors (pattern matching)
  - Weighted voting

Output:
  - Risk Priority: Critical / High / Medium / Low
  - Confidence Score: 0-100%
```

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 75% |
| Precision | 0.73 |
| Recall | 0.75 |
| F1 Score | 0.74 |

## CLI Usage

### Train Model

```bash
# Train on enriched CVE data
medsec-risk train --data data/enriched.json

# Specify output path
medsec-risk train --data data/enriched.json --model models/risk_model.pkl
```

### Predict Risk

```bash
# Score a single CVE
medsec-risk predict --cve CVE-2025-12345

# Score from file
medsec-risk predict --input new_cves.json --output scored.json
```

## Python API

```python
from medtech_ai_security.ml import RiskScorer

# Initialize scorer
scorer = RiskScorer()

# Train on data
scorer.fit(training_data)

# Predict risk
prediction = scorer.predict(cve_features)
print(f"Risk: {prediction.priority}, Confidence: {prediction.confidence}%")

# Save/load model
scorer.save("models/risk_model.pkl")
scorer = RiskScorer.load("models/risk_model.pkl")
```

## Feature Engineering

### CVSS Features

Extracted from CVSS v3.1 vector string:

```python
# Example: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
features = {
    "attack_vector": "NETWORK",
    "attack_complexity": "LOW",
    "privileges_required": "NONE",
    "user_interaction": "NONE",
    "scope": "UNCHANGED",
    "confidentiality": "HIGH",
    "integrity": "HIGH",
    "availability": "HIGH"
}
```

### Clinical Impact

Derived from Claude AI enrichment:

| Level | Description | Weight |
|-------|-------------|--------|
| Critical | Direct patient harm possible | 1.0 |
| High | Care disruption likely | 0.75 |
| Medium | Data integrity risk | 0.5 |
| Low | Minimal clinical impact | 0.25 |

### Device Type Encoding

Medical device categories mapped to risk multipliers:

| Device Type | Multiplier |
|-------------|------------|
| Life-sustaining | 1.5 |
| Diagnostic | 1.2 |
| Monitoring | 1.1 |
| Administrative | 1.0 |

## Output Format

```json
{
  "cve_id": "CVE-2025-12345",
  "risk_priority": "critical",
  "confidence": 87.5,
  "factors": {
    "cvss_contribution": 0.35,
    "clinical_contribution": 0.40,
    "device_contribution": 0.25
  },
  "recommendation": "Immediate patching required"
}
```
