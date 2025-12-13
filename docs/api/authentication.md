# Authentication

API authentication and authorization guide.

## Overview

The MedTech AI Security API supports multiple authentication methods:

| Method | Use Case | Security Level |
|--------|----------|----------------|
| API Key | Server-to-server integration | Standard |
| Bearer Token (JWT) | User sessions, short-lived access | High |
| OAuth 2.0 | Third-party integrations | Enterprise |

## API Key Authentication

### Obtaining an API Key

API keys are managed through the web dashboard or CLI:

```bash
# Generate new API key
medsec-api keys create --name "production-server"

# List existing keys
medsec-api keys list

# Revoke a key
medsec-api keys revoke <key-id>
```

### Using API Keys

Include the API key in the `Authorization` header:

```bash
curl -X GET "http://localhost:8000/api/v1/health" \
  -H "Authorization: Bearer your-api-key"
```

Or as a query parameter (not recommended for production):

```bash
curl -X GET "http://localhost:8000/api/v1/health?api_key=your-api-key"
```

### Python Example

```python
import requests

API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000/api/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(f"{BASE_URL}/health", headers=headers)
print(response.json())
```

## JWT Authentication

For user-facing applications, use JWT tokens for session management.

### Obtaining a Token

```http
POST /api/v1/auth/login
```

Request body:

```json
{
  "username": "user@example.com",
  "password": "secure-password"
}
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh-token-here"
}
```

### Refreshing Tokens

```http
POST /api/v1/auth/refresh
```

Request body:

```json
{
  "refresh_token": "refresh-token-here"
}
```

### Token Expiration

| Token Type | Default Expiration | Configurable |
|------------|-------------------|--------------|
| Access Token | 1 hour | Yes |
| Refresh Token | 7 days | Yes |
| API Key | Never (until revoked) | N/A |

## OAuth 2.0

For enterprise integrations, OAuth 2.0 is supported.

### Supported Flows

- Authorization Code (with PKCE)
- Client Credentials

### Configuration

Register your OAuth application:

```bash
medsec-api oauth register \
  --name "My Integration" \
  --redirect-uri "https://myapp.com/callback" \
  --scopes "read:vulnerabilities,write:reports"
```

### Authorization URL

```
https://your-domain.com/oauth/authorize?
  response_type=code&
  client_id=your-client-id&
  redirect_uri=https://myapp.com/callback&
  scope=read:vulnerabilities&
  state=random-state-value&
  code_challenge=code-challenge&
  code_challenge_method=S256
```

### Token Exchange

```http
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=authorization-code&
redirect_uri=https://myapp.com/callback&
client_id=your-client-id&
code_verifier=code-verifier
```

## Scopes and Permissions

API access is controlled through scopes:

| Scope | Description |
|-------|-------------|
| `read:vulnerabilities` | Read CVE and vulnerability data |
| `write:vulnerabilities` | Submit new vulnerability reports |
| `read:sbom` | Read SBOM analysis results |
| `write:sbom` | Submit SBOMs for analysis |
| `read:anomalies` | Read anomaly detection results |
| `write:anomalies` | Submit traffic for analysis |
| `read:models` | Read adversarial ML results |
| `write:models` | Submit models for evaluation |
| `admin` | Full administrative access |

### Checking Permissions

```http
GET /api/v1/auth/permissions
Authorization: Bearer your-token
```

Response:

```json
{
  "scopes": ["read:vulnerabilities", "read:sbom"],
  "expires_at": "2025-12-13T11:30:00Z"
}
```

## Security Best Practices

### API Key Security

1. **Never commit API keys** to version control
2. **Use environment variables** for key storage
3. **Rotate keys regularly** (recommended: every 90 days)
4. **Use separate keys** for different environments
5. **Monitor key usage** for anomalies

### Environment Variables

```bash
# .env file (never commit this)
MEDSEC_API_KEY=your-api-key
MEDSEC_API_URL=https://api.medsec.example.com
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MEDSEC_API_KEY")
```

### IP Allowlisting

Restrict API key usage to specific IP addresses:

```bash
medsec-api keys update <key-id> \
  --allowed-ips "192.168.1.0/24,10.0.0.0/8"
```

### Request Signing

For high-security environments, enable request signing:

```python
import hashlib
import hmac
import time

def sign_request(method: str, path: str, body: str, secret: str) -> dict:
    timestamp = str(int(time.time()))
    message = f"{method}\n{path}\n{timestamp}\n{body}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()

    return {
        "X-Signature": signature,
        "X-Timestamp": timestamp
    }
```

## Error Handling

### Authentication Errors

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `INVALID_API_KEY` | API key not found or revoked | Check key validity |
| `EXPIRED_TOKEN` | JWT token has expired | Refresh the token |
| `INSUFFICIENT_SCOPE` | Missing required scope | Request additional scopes |
| `IP_NOT_ALLOWED` | Request from non-allowed IP | Update IP allowlist |

### Example Error Response

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_API_KEY",
    "message": "The provided API key is invalid or has been revoked",
    "details": {
      "key_prefix": "msk_live_abc..."
    }
  }
}
```

## Audit Logging

All authentication events are logged:

```json
{
  "event": "api_key_used",
  "timestamp": "2025-12-13T10:30:00Z",
  "key_id": "key-uuid",
  "ip_address": "192.168.1.100",
  "endpoint": "/api/v1/threat-intel/nvd",
  "status": "success"
}
```

Access audit logs via:

```http
GET /api/v1/admin/audit-logs?start=2025-12-01&end=2025-12-13
Authorization: Bearer admin-token
```
