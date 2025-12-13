# API Endpoints

Complete reference for all REST API endpoints.

## Threat Intelligence

### Fetch CVEs from NVD

```http
GET /api/v1/threat-intel/nvd
```

Query parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| days | integer | No | Number of days to look back (default: 30) |
| keyword | string | No | Filter by keyword |
| severity | string | No | Filter by severity (critical, high, medium, low) |
| limit | integer | No | Maximum results (default: 100, max: 1000) |
| offset | integer | No | Pagination offset |

**Example:**

```bash
curl -X GET "http://localhost:8000/api/v1/threat-intel/nvd?days=7&keyword=medical" \
  -H "Authorization: Bearer your-api-key"
```

### Analyze CVE

```http
POST /api/v1/threat-intel/analyze
```

Request body:

```json
{
  "cve_id": "CVE-2025-12345",
  "enrich": true
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "cve_id": "CVE-2025-12345",
    "description": "Buffer overflow in medical device firmware...",
    "cvss_score": 8.5,
    "clinical_impact": "high",
    "device_type": "infusion_pump",
    "attack_scenario": "An attacker on the hospital network...",
    "mitigation_steps": ["Update firmware", "Enable segmentation"],
    "regulatory_impact": "May require FDA notification"
  }
}
```

### Fetch CISA Advisories

```http
GET /api/v1/threat-intel/cisa
```

Query parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| year | integer | No | Filter by year |
| sector | string | No | Filter by sector (healthcare, manufacturing, etc.) |

---

## Risk Scoring

### Score Vulnerability

```http
POST /api/v1/risk/score
```

Request body:

```json
{
  "cve_id": "CVE-2025-12345",
  "device_context": {
    "type": "infusion_pump",
    "network_accessible": true,
    "patient_facing": true
  }
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "cve_id": "CVE-2025-12345",
    "risk_score": 8.7,
    "priority": "critical",
    "factors": {
      "cvss_base": 8.5,
      "exploitability": 0.9,
      "clinical_impact": 0.95,
      "network_exposure": 0.8
    },
    "recommendation": "Immediate patching required"
  }
}
```

### Batch Score

```http
POST /api/v1/risk/batch
```

Request body:

```json
{
  "cve_ids": ["CVE-2025-12345", "CVE-2025-12346", "CVE-2025-12347"],
  "device_context": {
    "type": "medical_imaging"
  }
}
```

---

## Anomaly Detection

### Analyze Traffic

```http
POST /api/v1/anomaly/analyze
```

Request body (form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file | Yes | PCAP file to analyze |
| protocol | string | No | Focus on specific protocol (dicom, hl7, fhir) |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/anomaly/analyze" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@traffic.pcap" \
  -F "protocol=dicom"
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "total_packets": 15420,
    "anomalies_detected": 23,
    "anomaly_rate": 0.0015,
    "findings": [
      {
        "timestamp": "2025-12-13T10:15:32Z",
        "type": "unusual_port_scan",
        "severity": "medium",
        "source_ip": "192.168.1.105",
        "description": "Port scanning activity detected"
      }
    ],
    "protocol_stats": {
      "dicom": {"packets": 8500, "anomalies": 12},
      "hl7": {"packets": 4200, "anomalies": 8},
      "other": {"packets": 2720, "anomalies": 3}
    }
  }
}
```

### Real-time Monitoring

```http
WebSocket /api/v1/anomaly/stream
```

Connect via WebSocket for real-time anomaly notifications:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/anomaly/stream');

ws.onmessage = (event) => {
  const anomaly = JSON.parse(event.data);
  console.log('Anomaly detected:', anomaly);
};
```

---

## Adversarial ML

### Evaluate Model Robustness

```http
POST /api/v1/adversarial/evaluate
```

Request body (form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model | file | Yes | Keras model file (.keras or .h5) |
| test_images | file | Yes | NumPy array of test images (.npy) |
| test_labels | file | Yes | NumPy array of labels (.npy) |
| attacks | string | No | Comma-separated attack methods (default: fgsm,pgd) |

**Response:**

```json
{
  "status": "success",
  "data": {
    "clean_accuracy": 0.95,
    "attacks": {
      "fgsm_0.03": {"accuracy": 0.42, "success_rate": 0.58},
      "pgd_0.1": {"accuracy": 0.15, "success_rate": 0.85}
    },
    "robustness_score": 0.35,
    "clinical_impact": "high",
    "recommendations": [
      "Implement adversarial training",
      "Add input validation layer"
    ]
  }
}
```

### Generate Adversarial Examples

```http
POST /api/v1/adversarial/attack
```

Request body:

```json
{
  "method": "fgsm",
  "epsilon": 0.03,
  "images": "base64-encoded-numpy-array",
  "labels": [0, 1, 0, 1]
}
```

### Apply Defense

```http
POST /api/v1/adversarial/defend
```

Request body:

```json
{
  "method": "jpeg_compression",
  "quality": 75,
  "images": "base64-encoded-numpy-array"
}
```

---

## SBOM Analysis

### Analyze SBOM

```http
POST /api/v1/sbom/analyze
```

Request body (form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file | Yes | SBOM file (CycloneDX or SPDX JSON) |
| format | string | No | Force format (cyclonedx, spdx) |

**Response:**

```json
{
  "status": "success",
  "data": {
    "sbom_format": "cyclonedx",
    "total_components": 156,
    "risk_summary": {
      "overall_score": 7.2,
      "risk_level": "high",
      "critical_vulns": 3,
      "high_vulns": 12,
      "medium_vulns": 45
    },
    "top_risks": [
      {
        "component": "openssl",
        "version": "1.1.1",
        "risk_score": 9.1,
        "cves": ["CVE-2025-1234"],
        "recommendation": "Upgrade to 3.0.x"
      }
    ],
    "license_issues": [
      {
        "component": "gpl-library",
        "license": "GPL-3.0",
        "issue": "Copyleft may require source disclosure"
      }
    ],
    "dependency_graph": {
      "nodes": 156,
      "edges": 423,
      "max_depth": 8
    }
  }
}
```

### Generate SBOM

```http
POST /api/v1/sbom/generate
```

Request body:

```json
{
  "source": "requirements.txt",
  "format": "cyclonedx",
  "include_transitive": true
}
```

### Export to DefectDojo

```http
POST /api/v1/sbom/export/defectdojo
```

Request body:

```json
{
  "sbom_analysis_id": "uuid-from-analyze",
  "product_name": "My Medical Device",
  "engagement_name": "SBOM Analysis Q4 2025"
}
```

---

## Health and Status

### Health Check

```http
GET /api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "0.2.0",
  "components": {
    "database": "healthy",
    "ml_models": "healthy",
    "external_apis": "healthy"
  }
}
```

### Metrics

```http
GET /api/v1/metrics
```

Returns Prometheus-formatted metrics for monitoring.
