# API Reference

REST API documentation for MedTech AI Security platform.

## Overview

The MedTech AI Security platform exposes a REST API for programmatic access to all
security analysis capabilities. The API follows OpenAPI 3.0 specification.

## Base URL

```
http://localhost:8000/api/v1
```

For production deployments:

```
https://your-domain.com/api/v1
```

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Quick Start

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())
# {"status": "healthy", "version": "0.2.0"}

# Analyze CVE
response = requests.post(
    f"{BASE_URL}/threat-intel/analyze",
    json={"cve_id": "CVE-2025-12345"}
)
print(response.json())
```

## Response Format

All API responses follow a consistent JSON structure:

### Success Response

```json
{
  "status": "success",
  "data": {
    // Response payload
  },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2025-12-13T10:30:00Z",
    "processing_time_ms": 150
  }
}
```

### Error Response

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input provided",
    "details": [
      {"field": "cve_id", "message": "Must match CVE-YYYY-NNNNN format"}
    ]
  },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2025-12-13T10:30:00Z"
  }
}
```

## HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Missing or invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Input validation failed |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

## Rate Limiting

API requests are rate-limited per API key:

| Tier | Requests/minute | Requests/day |
|------|-----------------|--------------|
| Free | 10 | 100 |
| Standard | 60 | 10,000 |
| Enterprise | 600 | Unlimited |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1702468200
```

## Modules

The API is organized into modules corresponding to the platform phases:

| Module | Endpoint Prefix | Description |
|--------|-----------------|-------------|
| [Threat Intelligence](endpoints.md#threat-intelligence) | `/threat-intel` | CVE collection and enrichment |
| [Risk Scoring](endpoints.md#risk-scoring) | `/risk` | ML-based vulnerability prioritization |
| [Anomaly Detection](endpoints.md#anomaly-detection) | `/anomaly` | Network traffic analysis |
| [Adversarial ML](endpoints.md#adversarial-ml) | `/adversarial` | Model robustness testing |
| [SBOM Analysis](endpoints.md#sbom-analysis) | `/sbom` | Supply chain risk assessment |

## SDKs

Official client libraries:

- **Python**: `pip install medtech-ai-security[client]`
- **JavaScript/TypeScript**: Coming soon
- **Go**: Coming soon

## Webhooks

Configure webhooks to receive real-time notifications:

```python
# Register webhook
response = requests.post(
    f"{BASE_URL}/webhooks",
    json={
        "url": "https://your-server.com/webhook",
        "events": ["vulnerability.critical", "anomaly.detected"],
        "secret": "your-webhook-secret"
    },
    headers={"Authorization": "Bearer your-api-key"}
)
```

Webhook payloads are signed using HMAC-SHA256. Verify signatures to ensure
authenticity:

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```
