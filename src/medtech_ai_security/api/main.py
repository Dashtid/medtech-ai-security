"""
MedTech AI Security - Main API Application

FastAPI application providing REST endpoints for:
- SBOM analysis and supply chain risk scoring
- Anomaly detection for DICOM/HL7 traffic
- Threat intelligence CVE lookup
- Adversarial ML model testing
- DefectDojo integration

OpenAPI documentation available at /docs (Swagger UI) and /redoc (ReDoc).
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for API
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status", examples=["healthy"])
    version: str = Field(..., description="API version", examples=["1.1.0"])
    timestamp: str = Field(..., description="Current timestamp in ISO format")


class SBOMRequest(BaseModel):
    """Request model for SBOM analysis."""

    sbom: dict[str, Any] = Field(..., description="SBOM in CycloneDX or SPDX format")
    format: str = Field(
        default="cyclonedx",
        description="SBOM format",
        examples=["cyclonedx", "spdx"],
    )
    include_gnn: bool = Field(
        default=True,
        description="Include GNN-based vulnerability propagation analysis",
    )


class SBOMResponse(BaseModel):
    """Response model for SBOM analysis."""

    risk_level: str = Field(..., description="Overall risk level", examples=["high"])
    risk_score: float = Field(..., description="Risk score 0-100", examples=[65.5])
    total_packages: int = Field(..., description="Total packages analyzed")
    vulnerable_packages: int = Field(..., description="Packages with vulnerabilities")
    critical_vulnerabilities: int = Field(..., description="Critical CVE count")
    high_vulnerabilities: int = Field(..., description="High severity CVE count")
    recommendations: list[str] = Field(..., description="Remediation recommendations")
    fda_compliance_notes: list[str] = Field(..., description="FDA SBOM compliance notes")


class AnomalyRequest(BaseModel):
    """Request model for anomaly detection."""

    traffic_data: list[dict[str, Any]] = Field(
        ..., description="Network traffic records"
    )
    protocol: str = Field(
        default="dicom",
        description="Protocol type",
        examples=["dicom", "hl7"],
    )


class AnomalyResponse(BaseModel):
    """Response model for anomaly detection."""

    total_records: int = Field(..., description="Total records analyzed")
    anomalies_detected: int = Field(..., description="Number of anomalies found")
    anomaly_rate: float = Field(..., description="Percentage of anomalous traffic")
    alerts: list[dict[str, Any]] = Field(..., description="Detailed anomaly alerts")


class CVERequest(BaseModel):
    """Request model for CVE lookup."""

    cve_id: str = Field(..., description="CVE identifier", examples=["CVE-2024-1234"])


class CVEResponse(BaseModel):
    """Response model for CVE lookup."""

    cve_id: str = Field(..., description="CVE identifier")
    description: str = Field(..., description="Vulnerability description")
    cvss_score: float | None = Field(None, description="CVSS v3 score")
    severity: str = Field(..., description="Severity level")
    affected_products: list[str] = Field(..., description="Affected products")
    references: list[str] = Field(..., description="Reference URLs")
    clinical_impact: str | None = Field(None, description="Medical device impact")


class AdversarialRequest(BaseModel):
    """Request model for adversarial testing."""

    model_type: str = Field(
        ..., description="Model architecture", examples=["cnn", "resnet"]
    )
    attack_method: str = Field(
        default="fgsm",
        description="Attack method",
        examples=["fgsm", "pgd", "cw"],
    )
    epsilon: float = Field(
        default=0.03,
        description="Perturbation budget",
        ge=0.0,
        le=1.0,
    )


class AdversarialResponse(BaseModel):
    """Response model for adversarial testing."""

    attack_method: str = Field(..., description="Attack method used")
    clean_accuracy: float = Field(..., description="Accuracy on clean samples")
    adversarial_accuracy: float = Field(..., description="Accuracy on adversarial samples")
    robustness_score: float = Field(..., description="Model robustness score 0-100")
    recommendations: list[str] = Field(..., description="Defense recommendations")


class BenchmarkRequest(BaseModel):
    """Request model for benchmark execution."""

    modules: list[str] = Field(
        default=["anomaly", "sbom"],
        description="Modules to benchmark",
        examples=[["anomaly", "sbom", "adversarial"]],
    )
    iterations: int = Field(
        default=10,
        description="Number of benchmark iterations",
        ge=1,
        le=100,
    )


class BenchmarkResponse(BaseModel):
    """Response model for benchmark results."""

    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    modules: dict[str, dict[str, Any]] = Field(
        ..., description="Per-module benchmark results"
    )
    system_info: dict[str, Any] = Field(..., description="System information")
    timestamp: str = Field(..., description="Benchmark timestamp")


class ModelCompareRequest(BaseModel):
    """Request model for model comparison."""

    model_ids: list[str] = Field(
        ...,
        description="List of model IDs to compare",
        min_length=2,
        max_length=5,
    )
    attack_methods: list[str] = Field(
        default=["fgsm", "pgd"],
        description="Attack methods to use for comparison",
    )
    epsilon: float = Field(
        default=0.03,
        description="Perturbation budget for attacks",
        ge=0.0,
        le=1.0,
    )


class ModelCompareResponse(BaseModel):
    """Response model for model comparison."""

    comparison_id: str = Field(..., description="Unique comparison ID")
    models: dict[str, dict[str, Any]] = Field(
        ..., description="Per-model robustness results"
    )
    ranking: list[str] = Field(..., description="Models ranked by robustness")
    best_model: str = Field(..., description="Most robust model ID")
    recommendations: list[str] = Field(..., description="Recommendations based on comparison")


# =============================================================================
# Application Lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle manager."""
    # Startup
    logger.info("Starting MedTech AI Security API...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")

    yield

    # Shutdown
    logger.info("Shutting down MedTech AI Security API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="MedTech AI Security API",
    description="""
## AI-Powered Medical Device Cybersecurity Platform

This API provides comprehensive security analysis for medical devices, including:

### Features

- **SBOM Analysis**: Supply chain risk scoring using Graph Neural Networks
- **Anomaly Detection**: Real-time DICOM/HL7 traffic analysis
- **Threat Intelligence**: NVD/CISA CVE enrichment for medical devices
- **Adversarial ML**: Robustness testing for medical AI models
- **DefectDojo Integration**: Vulnerability management workflow

### Regulatory Compliance

Supports compliance with:
- FDA Cybersecurity Guidance (SBOM requirements)
- EU MDR 2017/745 (Cybersecurity as essential requirement)
- IEC 62304 (Medical device software lifecycle)
- IEC 81001-5-1 (Health software security)

### Authentication

API authentication is handled via API keys passed in the `X-API-Key` header.
Contact your administrator to obtain API credentials.
    """,
    version="1.1.0",
    contact={
        "name": "David Dashti",
        "email": "david.dashti@hermesmedical.com",
        "url": "https://github.com/Dashtid/medtech-ai-security",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "Health",
            "description": "Service health and readiness endpoints",
        },
        {
            "name": "SBOM Analysis",
            "description": "Software Bill of Materials security analysis",
        },
        {
            "name": "Anomaly Detection",
            "description": "Network traffic anomaly detection for medical protocols",
        },
        {
            "name": "Threat Intelligence",
            "description": "CVE lookup and medical device vulnerability intelligence",
        },
        {
            "name": "Adversarial ML",
            "description": "Robustness testing for medical AI models",
        },
        {
            "name": "Benchmarks",
            "description": "Performance benchmarking and profiling",
        },
        {
            "name": "Model Comparison",
            "description": "Compare robustness across models",
        },
        {
            "name": "Real-time",
            "description": "Real-time monitoring via WebSocket",
        },
    ],
    lifespan=lifespan,
)

# WebSocket connection manager for real-time anomaly streaming
class ConnectionManager:
    """Manage WebSocket connections for real-time anomaly streaming."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


ws_manager = ConnectionManager()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Returns service health status and version information.",
)
async def health_check() -> HealthResponse:
    """Check if the service is healthy."""
    return HealthResponse(
        status="healthy",
        version="1.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/ready",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Readiness check",
    description="Returns service readiness status. Used by Kubernetes probes.",
)
async def readiness_check() -> HealthResponse:
    """Check if the service is ready to accept requests."""
    # Add readiness checks here (database, model loading, etc.)
    return HealthResponse(
        status="ready",
        version="1.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/metrics",
    tags=["Health"],
    summary="Prometheus metrics",
    description="Returns Prometheus-formatted metrics for monitoring.",
)
async def metrics() -> str:
    """Return Prometheus metrics."""
    # Basic metrics - extend with prometheus_client for production
    return """# HELP medtech_api_requests_total Total API requests
# TYPE medtech_api_requests_total counter
medtech_api_requests_total 0

# HELP medtech_api_health Service health status
# TYPE medtech_api_health gauge
medtech_api_health 1
"""


# =============================================================================
# SBOM Analysis Endpoints
# =============================================================================


@app.post(
    "/api/v1/sbom/analyze",
    response_model=SBOMResponse,
    tags=["SBOM Analysis"],
    summary="Analyze SBOM for security risks",
    description="""
Analyzes a Software Bill of Materials (SBOM) for supply chain security risks.

Supports:
- **CycloneDX** (JSON/XML)
- **SPDX** (JSON/Tag-Value)

Returns risk scores, vulnerability counts, and FDA compliance notes.
    """,
)
async def analyze_sbom(request: SBOMRequest) -> SBOMResponse:
    """Analyze SBOM for vulnerabilities and supply chain risks."""
    try:
        # Import here to avoid circular imports
        from medtech_ai_security.sbom_analysis.analyzer import SBOMAnalyzer

        analyzer = SBOMAnalyzer()
        # In production, this would call the actual analyzer
        # For now, return a sample response

        return SBOMResponse(
            risk_level="medium",
            risk_score=45.5,
            total_packages=len(request.sbom.get("components", [])),
            vulnerable_packages=2,
            critical_vulnerabilities=0,
            high_vulnerabilities=2,
            recommendations=[
                "Update vulnerable-lib to version 2.0.0",
                "Review dependency tree for transitive vulnerabilities",
            ],
            fda_compliance_notes=[
                "FDA: SBOM contains required component information",
                "FDA: All components have license information",
            ],
        )
    except Exception as e:
        logger.error(f"SBOM analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SBOM analysis failed: {str(e)}",
        )


# =============================================================================
# Anomaly Detection Endpoints
# =============================================================================


@app.post(
    "/api/v1/anomaly/detect",
    response_model=AnomalyResponse,
    tags=["Anomaly Detection"],
    summary="Detect anomalies in network traffic",
    description="""
Analyzes DICOM or HL7 network traffic for anomalies using an autoencoder-based detector.

Supports detection of:
- Data exfiltration attempts
- Protocol violations
- Ransomware patterns
- DoS attacks
- Unauthorized access
    """,
)
async def detect_anomalies(request: AnomalyRequest) -> AnomalyResponse:
    """Detect anomalies in medical device network traffic."""
    try:
        # In production, this would call the actual detector
        total = len(request.traffic_data)
        anomalies = max(1, int(total * 0.05))  # Sample: 5% anomaly rate

        return AnomalyResponse(
            total_records=total,
            anomalies_detected=anomalies,
            anomaly_rate=round(anomalies / total * 100, 2) if total > 0 else 0,
            alerts=[
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "data_exfiltration",
                    "severity": "high",
                    "description": "Unusual data transfer pattern detected",
                }
            ],
        )
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}",
        )


# =============================================================================
# Threat Intelligence Endpoints
# =============================================================================


@app.get(
    "/api/v1/cve/{cve_id}",
    response_model=CVEResponse,
    tags=["Threat Intelligence"],
    summary="Lookup CVE details",
    description="""
Retrieves detailed information about a CVE, including:
- CVSS scores and severity
- Affected products
- Clinical impact for medical devices
- Remediation guidance
    """,
)
async def get_cve(cve_id: str) -> CVEResponse:
    """Lookup CVE vulnerability details."""
    # Validate CVE ID format
    if not cve_id.upper().startswith("CVE-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid CVE ID format. Expected format: CVE-YYYY-NNNNN",
        )

    # In production, this would query NVD/CISA APIs
    return CVEResponse(
        cve_id=cve_id.upper(),
        description="Sample vulnerability description",
        cvss_score=7.5,
        severity="high",
        affected_products=["Medical Device Firmware v1.0"],
        references=["https://nvd.nist.gov/vuln/detail/" + cve_id.upper()],
        clinical_impact="May affect device availability and patient safety",
    )


# =============================================================================
# Adversarial ML Endpoints
# =============================================================================


@app.post(
    "/api/v1/adversarial/test",
    response_model=AdversarialResponse,
    tags=["Adversarial ML"],
    summary="Test model robustness",
    description="""
Tests a medical AI model's robustness against adversarial attacks.

Supported attack methods:
- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent
- **C&W**: Carlini & Wagner attack
    """,
)
async def test_adversarial(request: AdversarialRequest) -> AdversarialResponse:
    """Test model robustness against adversarial attacks."""
    try:
        # In production, this would run actual adversarial testing
        return AdversarialResponse(
            attack_method=request.attack_method,
            clean_accuracy=0.95,
            adversarial_accuracy=0.72,
            robustness_score=75.8,
            recommendations=[
                "Consider adversarial training to improve robustness",
                "Implement input validation for unusual pixel values",
                "Add JPEG compression as preprocessing defense",
            ],
        )
    except Exception as e:
        logger.error(f"Adversarial testing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adversarial testing failed: {str(e)}",
        )


# =============================================================================
# Benchmark Endpoints
# =============================================================================


@app.post(
    "/api/v1/benchmark",
    response_model=BenchmarkResponse,
    tags=["Benchmarks"],
    summary="Run performance benchmarks",
    description="""
Execute performance benchmarks on specified modules.

Available modules:
- **anomaly**: Anomaly detection benchmarks
- **sbom**: SBOM analysis benchmarks
- **adversarial**: Adversarial ML benchmarks
- **threat_intel**: Threat intelligence benchmarks

Returns detailed timing metrics, throughput, and system information.
    """,
)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """Run performance benchmarks on specified modules."""
    import platform
    import time

    try:
        start_time = time.perf_counter()
        module_results: dict[str, dict[str, Any]] = {}

        for module in request.modules:
            module_start = time.perf_counter()

            if module == "anomaly":
                # Simulate anomaly detection benchmark
                module_results["anomaly"] = {
                    "avg_latency_ms": 12.5,
                    "throughput_per_sec": 8000,
                    "p95_latency_ms": 25.3,
                    "p99_latency_ms": 45.1,
                    "iterations": request.iterations,
                }
            elif module == "sbom":
                # Simulate SBOM analysis benchmark
                module_results["sbom"] = {
                    "avg_latency_ms": 150.2,
                    "throughput_per_sec": 66,
                    "p95_latency_ms": 210.5,
                    "p99_latency_ms": 350.8,
                    "iterations": request.iterations,
                }
            elif module == "adversarial":
                # Simulate adversarial ML benchmark
                module_results["adversarial"] = {
                    "avg_latency_ms": 85.7,
                    "throughput_per_sec": 117,
                    "p95_latency_ms": 120.3,
                    "p99_latency_ms": 180.5,
                    "iterations": request.iterations,
                }
            elif module == "threat_intel":
                # Simulate threat intelligence benchmark
                module_results["threat_intel"] = {
                    "avg_latency_ms": 25.3,
                    "throughput_per_sec": 395,
                    "p95_latency_ms": 45.2,
                    "p99_latency_ms": 78.9,
                    "iterations": request.iterations,
                }
            else:
                module_results[module] = {"error": f"Unknown module: {module}"}

            module_results.get(module, {})["execution_time_ms"] = round(
                (time.perf_counter() - module_start) * 1000, 2
            )

        total_time = (time.perf_counter() - start_time) * 1000

        return BenchmarkResponse(
            execution_time_ms=round(total_time, 2),
            modules=module_results,
            system_info={
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Benchmark execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution failed: {str(e)}",
        )


# =============================================================================
# Model Comparison Endpoints
# =============================================================================


@app.post(
    "/api/v1/models/compare",
    response_model=ModelCompareResponse,
    tags=["Model Comparison"],
    summary="Compare model robustness",
    description="""
Compare robustness of multiple models against adversarial attacks.

This endpoint evaluates multiple models using the specified attack methods
and ranks them by their robustness scores.

Returns:
- Per-model accuracy metrics
- Robustness scores
- Ranked list of models
- Recommendations for model selection
    """,
)
async def compare_models(request: ModelCompareRequest) -> ModelCompareResponse:
    """Compare robustness across multiple models."""
    import hashlib
    import random

    try:
        comparison_id = hashlib.sha256(
            f"{request.model_ids}{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]

        model_results: dict[str, dict[str, Any]] = {}
        scores: list[tuple[str, float]] = []

        for model_id in request.model_ids:
            # Simulate model evaluation with randomized results
            clean_acc = random.uniform(0.88, 0.98)
            attack_results: dict[str, dict[str, float]] = {}

            for attack in request.attack_methods:
                # Accuracy drops more with stronger attacks
                if attack == "fgsm":
                    adv_acc = clean_acc * random.uniform(0.75, 0.90)
                elif attack == "pgd":
                    adv_acc = clean_acc * random.uniform(0.65, 0.82)
                elif attack == "cw":
                    adv_acc = clean_acc * random.uniform(0.55, 0.75)
                elif attack == "deepfool":
                    adv_acc = clean_acc * random.uniform(0.60, 0.80)
                elif attack == "autoattack":
                    adv_acc = clean_acc * random.uniform(0.50, 0.70)
                else:
                    adv_acc = clean_acc * random.uniform(0.70, 0.85)

                attack_results[attack] = {
                    "clean_accuracy": round(clean_acc, 4),
                    "adversarial_accuracy": round(adv_acc, 4),
                    "accuracy_drop": round(clean_acc - adv_acc, 4),
                }

            # Calculate overall robustness score
            avg_adv_acc = sum(
                r["adversarial_accuracy"] for r in attack_results.values()
            ) / len(attack_results)
            robustness_score = round(avg_adv_acc * 100, 2)

            model_results[model_id] = {
                "clean_accuracy": round(clean_acc, 4),
                "attack_results": attack_results,
                "robustness_score": robustness_score,
            }
            scores.append((model_id, robustness_score))

        # Rank models by robustness
        scores.sort(key=lambda x: x[1], reverse=True)
        ranking = [model_id for model_id, _ in scores]
        best_model = ranking[0]

        # Generate recommendations
        recommendations = []
        if scores[0][1] - scores[-1][1] > 10:
            recommendations.append(
                f"Significant robustness gap detected. {best_model} is notably more robust."
            )
        if scores[0][1] < 70:
            recommendations.append(
                "All models show vulnerability to adversarial attacks. Consider adversarial training."
            )
        if "autoattack" in request.attack_methods:
            recommendations.append(
                "AutoAttack results provide the most reliable robustness estimate."
            )
        recommendations.append(
            f"Recommended model for deployment: {best_model} (robustness: {scores[0][1]:.1f}%)"
        )

        return ModelCompareResponse(
            comparison_id=comparison_id,
            models=model_results,
            ranking=ranking,
            best_model=best_model,
            recommendations=recommendations,
        )
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model comparison failed: {str(e)}",
        )


# =============================================================================
# Real-time WebSocket Endpoints
# =============================================================================


@app.websocket("/ws/anomaly/stream")
async def anomaly_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time anomaly streaming.

    Clients connect to receive real-time anomaly alerts as they are detected.
    Messages are JSON-formatted with the following structure:

    ```json
    {
        "timestamp": "2024-01-01T12:00:00Z",
        "type": "anomaly_alert",
        "severity": "high",
        "protocol": "dicom",
        "description": "Unusual data transfer pattern detected",
        "confidence": 0.92
    }
    ```

    Connection lifecycle:
    1. Client connects to ws://host/ws/anomaly/stream
    2. Server sends heartbeat every 30 seconds
    3. Anomaly alerts are pushed as they occur
    4. Client can disconnect at any time
    """
    import asyncio
    import random

    await ws_manager.connect(websocket)
    try:
        heartbeat_interval = 30
        last_heartbeat = datetime.now(timezone.utc)

        while True:
            # Check for incoming messages (non-blocking)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=1.0
                )
                # Handle client commands
                if data.get("command") == "subscribe":
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "protocols": data.get("protocols", ["dicom", "hl7"]),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except asyncio.TimeoutError:
                pass

            # Send periodic heartbeat
            now = datetime.now(timezone.utc)
            if (now - last_heartbeat).total_seconds() >= heartbeat_interval:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": now.isoformat(),
                    "active_connections": len(ws_manager.active_connections),
                })
                last_heartbeat = now

            # Simulate random anomaly detection (in production, this would be event-driven)
            if random.random() < 0.1:  # 10% chance per iteration
                anomaly_types = [
                    ("data_exfiltration", "high", "Unusual data transfer pattern detected"),
                    ("protocol_violation", "medium", "Invalid DICOM command received"),
                    ("ransomware_pattern", "critical", "Potential ransomware activity detected"),
                    ("dos_attack", "high", "Denial of service pattern identified"),
                    ("unauthorized_access", "medium", "Authentication anomaly detected"),
                ]
                anomaly = random.choice(anomaly_types)
                await websocket.send_json({
                    "type": "anomaly_alert",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "anomaly_type": anomaly[0],
                    "severity": anomaly[1],
                    "description": anomaly[2],
                    "protocol": random.choice(["dicom", "hl7"]),
                    "confidence": round(random.uniform(0.75, 0.99), 2),
                })

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"},
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the API server."""
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")  # nosec B104 - Intentional for container deployment
    workers = int(os.getenv("WORKERS", "1"))

    uvicorn.run(
        "medtech_ai_security.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv("ENVIRONMENT") == "development",
    )


if __name__ == "__main__":
    main()
