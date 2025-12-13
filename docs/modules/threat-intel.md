# Threat Intelligence Module

Phase 1 of the MedTech AI Security platform. Collects, enriches, and analyzes
vulnerability data specific to medical devices.

## Overview

The Threat Intelligence module automates the collection of CVE data from authoritative
sources and enriches it with AI-powered clinical context analysis.

## Components

| Component | Description |
|-----------|-------------|
| `nvd_scraper.py` | NVD API client for CVE collection |
| `cisa_scraper.py` | CISA ICS-CERT advisory parser |
| `claude_processor.py` | AI-powered vulnerability enrichment |

## Data Flow

```mermaid
flowchart LR
    A["NVD/CISA APIs"] --> B["Scrapers"]
    B --> C["Raw CVE Data"]
    C --> D["Claude Enrichment"]
    D --> E["Enriched CVEs"]
    E --> F["DefectDojo"]
```

## CLI Usage

### Scrape NVD

```bash
# Get CVEs from last 30 days
medsec-nvd --days 30 --output data/nvd_cves.json

# Filter by keyword
medsec-nvd --keyword "infusion pump" --output data/pumps.json

# With API key (higher rate limit)
NVD_API_KEY=xxx medsec-nvd --days 90 --output data/cves.json
```

### Scrape CISA

```bash
# Get all ICS-CERT advisories
medsec-cisa --output data/cisa_advisories.json

# Filter by year
medsec-cisa --year 2025 --output data/cisa_2025.json
```

### Enrich CVEs

```bash
# Enrich with Claude AI
medsec-enrich --input data/nvd_cves.json --output data/enriched.json

# Specify model
medsec-enrich --input data/cves.json --model claude-3-sonnet --output enriched.json
```

## Python API

```python
from medtech_ai_security.threat_intel import NVDScraper, CISAScraper, ClaudeProcessor

# Scrape NVD
scraper = NVDScraper(api_key="your_key")
cves = scraper.fetch_medical_device_cves(days=30)

# Scrape CISA
cisa = CISAScraper()
advisories = cisa.fetch_advisories()

# Enrich with Claude
processor = ClaudeProcessor(api_key="your_anthropic_key")
enriched = processor.enrich_cve(cve_data)
```

## Enrichment Fields

Claude AI adds the following fields to each CVE:

| Field | Description |
|-------|-------------|
| `clinical_impact` | Impact on patient safety (critical/high/medium/low) |
| `device_type` | Type of medical device affected |
| `attack_scenario` | Realistic attack scenario description |
| `mitigation_steps` | Recommended remediation steps |
| `regulatory_impact` | FDA/EU MDR compliance implications |

## Medical Device Keywords

The scraper filters CVEs using these keywords:

- Medical device categories: infusion pump, pacemaker, ventilator, etc.
- Protocols: DICOM, HL7, FHIR
- Vendors: Philips, GE Healthcare, Siemens Healthineers, etc.

## Rate Limiting

| Source | Without API Key | With API Key |
|--------|-----------------|--------------|
| NVD | 5 req/30s | 50 req/30s |
| CISA | 10 req/min | 10 req/min |
| Claude | Based on tier | Based on tier |

## Output Format

```json
{
  "cve_id": "CVE-2025-12345",
  "description": "Buffer overflow in...",
  "cvss_score": 8.5,
  "clinical_impact": "high",
  "device_type": "infusion_pump",
  "attack_scenario": "An attacker on the hospital network...",
  "mitigation_steps": ["Update firmware to v2.1", "Enable network segmentation"],
  "regulatory_impact": "May require FDA notification under 21 CFR Part 806"
}
```
