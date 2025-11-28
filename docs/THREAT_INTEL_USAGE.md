# Medical Device Threat Intelligence - Usage Guide

This guide covers the threat intelligence and ML risk scoring tools for collecting, analyzing, and prioritizing medical device vulnerabilities.

## Overview

The threat intelligence module provides:

- **NVD Scraper**: Fetches CVEs from NIST National Vulnerability Database
- **CISA Scraper**: Fetches ICS-CERT medical device advisories
- **Claude Processor**: Merges Claude.ai analysis with scraped data
- **ML Risk Scorer**: ML-powered vulnerability prioritization

## Quick Start

### 1. Fetch Medical Device CVEs from NVD

```bash
# Fetch 50 CVEs (takes ~5 minutes due to rate limiting)
UV_LINK_MODE=copy uv run medsec-nvd --max-results 50 --generate-prompt

# Fetch 100 CVEs with prompt generation
UV_LINK_MODE=copy uv run medsec-nvd --max-results 100 --generate-prompt \
    --output data/threat_intel/cves/medical_devices.json

# Filter by severity
UV_LINK_MODE=copy uv run medsec-nvd --max-results 50 --severity CRITICAL
```

Output files:
- `data/threat_intel/cves/medical_devices.json` - CVE data
- `data/threat_intel/cves/claude_prompt.txt` - Prompt for Claude.ai

### 2. Analyze CVEs with Claude.ai

1. Open `data/threat_intel/cves/claude_prompt.txt`
2. Copy the content to claude.ai
3. Get the JSON response
4. Save the response to `data/threat_intel/cves/claude_response.json`

### 3. Merge Analysis into CVE Data

```bash
UV_LINK_MODE=copy uv run medsec-enrich \
    --cve-file data/threat_intel/cves/medical_devices.json \
    --response-file data/threat_intel/cves/claude_response.json \
    --report
```

Output:
- `data/threat_intel/cves/medical_devices_enriched.json` - Enriched data
- `data/threat_intel/cves/threat_intel_report.txt` - Summary report

### 4. Fetch CISA Medical Advisories (Optional)

```bash
UV_LINK_MODE=copy uv run medsec-cisa --max-results 20 --generate-prompt
```

Output:
- `data/threat_intel/advisories/cisa_medical.json`
- `data/threat_intel/advisories/cisa_claude_prompt.txt`

## CLI Commands

| Command | Description |
|---------|-------------|
| `medsec-nvd` | Scrape NVD for medical device CVEs |
| `medsec-cisa` | Scrape CISA ICS-CERT advisories |
| `medsec-enrich` | Merge Claude.ai analysis with CVE data |
| `medsec-risk` | ML-powered vulnerability risk scoring |

## Command Options

### medsec-nvd

```
--max-results    Maximum CVEs to fetch (default: 50)
--days-back      How far back to search (default: 365)
--output         Output file path
--api-key        NVD API key for faster rate limits
--generate-prompt Generate Claude.ai prompt
--severity       Filter by LOW, MEDIUM, HIGH, CRITICAL
```

### medsec-cisa

```
--max-results         Maximum advisories (default: 20)
--output              Output file path
--generate-prompt     Generate Claude.ai prompt
--include-general-ics Also search general ICS advisories
```

### medsec-enrich

```
--cve-file       Path to original CVE JSON
--response-file  Path to Claude.ai response JSON
--output         Output path for enriched file
--report         Generate summary report
```

## Python API

```python
from medtech_ai_security.threat_intel import NVDScraper, CISAScraper
from medtech_ai_security.ml import VulnerabilityRiskScorer

# NVD Scraper
scraper = NVDScraper(api_key="your-api-key")  # optional
cves = scraper.search_medical_device_cves(max_results=100)
scraper.save_results(cves, "data/cves.json")

# Generate prompt for Claude.ai
prompt = scraper.generate_claude_prompt(cves)

# CISA Scraper
cisa = CISAScraper()
advisories = cisa.scrape_medical_advisories(max_results=50)
cisa.save_results(advisories, "data/advisories.json")

# ML Risk Scorer
scorer = VulnerabilityRiskScorer()
scorer.load_training_data("data/cves_enriched.json")
scorer.train()

# Predict risk for vulnerabilities
predictions = scorer.predict_batch(cves)
for pred in predictions[:5]:
    print(f"{pred.cve_id}: {pred.risk_score:.1f} ({pred.priority})")
```

## ML Risk Scoring

### 5. Train and Run Risk Predictions

```bash
# Train model and predict on enriched data
UV_LINK_MODE=copy uv run medsec-risk \
    --data data/threat_intel/cves/medical_devices_enriched.json \
    --predict \
    --output data/threat_intel/cves/risk_predictions.json
```

### medsec-risk Options

```text
--data           Path to enriched CVE JSON file (required)
--save-model     Path to save trained model
--load-model     Path to load pre-trained model
--predict        Run predictions on all CVEs
--output         Output path for predictions JSON
```

### Risk Prediction Output

```json
{
  "predictions": [
    {
      "cve_id": "CVE-2021-27410",
      "risk_score": 100.0,
      "priority": "CRITICAL",
      "confidence": 1.0,
      "contributing_factors": {
        "cvss_score": 0.15,
        "clinical_impact": 0.12,
        "device_imaging": 0.10
      },
      "recommendation": "IMMEDIATE ACTION REQUIRED..."
    }
  ],
  "summary": {
    "total": 100,
    "critical": 10,
    "high": 51,
    "medium": 10,
    "low": 29
  }
}
```

### Feature Engineering

The ML model uses these features:

- **CVSS Components**: Score, attack vector, complexity, privileges, user interaction
- **CWE Domains**: Memory safety, authentication, injection, cryptography, etc.
- **Device Type**: Imaging, monitoring, infusion, implantable, etc.
- **Clinical Impact**: HIGH, MEDIUM, LOW (from Claude.ai enrichment)
- **Exploitability**: EASY, MODERATE, HARD (from Claude.ai enrichment)
- **Vulnerability Age**: Years since publication
- **Exploit Availability**: References to exploit-db, metasploit, etc.

## Medical Device Keywords

The NVD scraper searches for these keywords:

**Device Types:**
- DICOM, PACS, MRI, CT scanner, X-ray, ultrasound
- infusion pump, insulin pump, pacemaker, defibrillator
- patient monitor, vital signs, ECG, pulse oximeter
- ventilator, dialysis, radiation therapy

**Healthcare IT:**
- HL7, FHIR, EHR, EMR, HIPAA

**Vendors:**
- Philips Healthcare, GE Healthcare, Siemens Healthineers
- Medtronic, Baxter, Abbott, BD, Boston Scientific

## Data Schema

### CVE Entry

```json
{
  "cve_id": "CVE-2021-27410",
  "description": "...",
  "published_date": "2021-06-11T17:15:10.770",
  "cvss_v3_score": 9.8,
  "cvss_v3_severity": "CRITICAL",
  "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
  "cwe_ids": ["CWE-787"],
  "references": ["https://..."],
  "affected_products": ["cpe:2.3:..."],
  "matched_keywords": ["medical device"],
  "device_type": "monitoring",
  "clinical_impact": "HIGH",
  "exploitability": "EASY",
  "remediation": "patch_available"
}
```

### Claude.ai Response Format

```json
{
  "analyses": [
    {
      "cve_id": "CVE-2021-27410",
      "device_type": "monitoring",
      "clinical_impact": "HIGH",
      "exploitability": "EASY",
      "remediation": "patch_available",
      "reasoning": "Welch Allyn patient monitors with remote code execution..."
    }
  ]
}
```

## Rate Limits

**NVD API:**
- Without API key: 5 requests per 30 seconds (6s delay)
- With API key: 50 requests per 30 seconds (0.6s delay)
- Get API key: https://nvd.nist.gov/developers/request-an-api-key

**CISA Website:**
- No official API, web scraping only
- 2 second delay between requests (be respectful)

## Troubleshooting

### NVD API Returns 404

The NVD API has strict date format requirements. Date filtering is currently disabled. CVEs are fetched by keyword search instead.

### CISA Website Timeout

CISA may block or rate-limit requests. Try:
- Increase timeout in scraper
- Use a different network/VPN
- Wait and retry later

### UV Sync Fails with "Access Denied"

Close any Python processes using the venv, then retry:
```bash
# Kill Python processes
taskkill /F /IM python.exe

# Retry sync
UV_LINK_MODE=copy uv sync
```

## Sources

- [NVD API Documentation](https://nvd.nist.gov/developers/vulnerabilities)
- [CISA ICS Medical Advisories](https://www.cisa.gov/news-events/ics-medical-advisories)
- [ICS Advisory Project](https://www.icsadvisoryproject.com/) - Community-maintained advisory database
