"""
MedTech AI Security - Real-Time Monitoring Dashboard

FastAPI-based dashboard with WebSocket support for live updates.

Features:
- Real-time security metrics visualization
- Live anomaly detection alerts
- Vulnerability trend analysis
- SBOM risk overview
- Threat intelligence feed

Usage:
    medsec-dashboard --port 3000
    # Access at http://localhost:3000
"""

import asyncio
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class SecurityMetrics(BaseModel):
    """Current security metrics snapshot."""

    timestamp: str
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    anomalies_24h: int
    sbom_risk_score: float
    threat_intel_updates: int


class Alert(BaseModel):
    """Security alert."""

    id: str
    timestamp: str
    severity: str
    type: str
    message: str
    source: str


# =============================================================================
# WebSocket Connection Manager
# =============================================================================


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# =============================================================================
# Background Tasks
# =============================================================================


async def generate_metrics() -> SecurityMetrics:
    """Generate current security metrics (simulated for demo)."""
    return SecurityMetrics(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_vulnerabilities=random.randint(50, 100),
        critical_count=random.randint(0, 5),
        high_count=random.randint(5, 15),
        medium_count=random.randint(15, 30),
        low_count=random.randint(30, 50),
        anomalies_24h=random.randint(0, 20),
        sbom_risk_score=round(random.uniform(30, 70), 1),
        threat_intel_updates=random.randint(0, 10),
    )


async def generate_alert() -> Alert | None:
    """Generate random alert (simulated for demo)."""
    if random.random() < 0.3:  # 30% chance of alert
        alert_types = [
            ("anomaly", "Unusual network traffic detected", "high"),
            ("vulnerability", "New critical CVE published", "critical"),
            ("sbom", "Supply chain risk increased", "medium"),
            ("adversarial", "Model robustness degradation", "high"),
            ("threat_intel", "CISA advisory for medical devices", "high"),
        ]
        alert_type, message, severity = random.choice(alert_types)
        return Alert(
            id=f"ALERT-{random.randint(1000, 9999)}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            type=alert_type,
            message=message,
            source="MedTech AI Security",
        )
    return None


async def metrics_broadcaster() -> None:
    """Background task to broadcast metrics updates."""
    while True:
        if manager.active_connections:
            metrics = await generate_metrics()
            await manager.broadcast({"type": "metrics", "data": metrics.model_dump()})

            alert = await generate_alert()
            if alert:
                await manager.broadcast({"type": "alert", "data": alert.model_dump()})

        await asyncio.sleep(5)  # Update every 5 seconds


# =============================================================================
# Application Lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("Starting MedTech AI Security Dashboard...")

    # Start background metrics broadcaster
    task = asyncio.create_task(metrics_broadcaster())

    yield

    # Cleanup
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Dashboard shutdown complete.")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="MedTech AI Security Dashboard",
    description="Real-time monitoring dashboard for medical device security",
    version="1.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Dashboard HTML
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedTech AI Security Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .metric-card {
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .alert-critical { border-left: 4px solid #ef4444; }
        .alert-high { border-left: 4px solid #f97316; }
        .alert-medium { border-left: 4px solid #eab308; }
        .alert-low { border-left: 4px solid #22c55e; }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <!-- Header -->
    <header class="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div class="flex items-center justify-between">
            <div class="flex items-center space-x-4">
                <div class="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
                    </svg>
                </div>
                <div>
                    <h1 class="text-xl font-bold">MedTech AI Security</h1>
                    <p class="text-gray-400 text-sm">Real-Time Monitoring Dashboard</p>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <div id="connection-status" class="flex items-center space-x-2">
                    <span class="w-3 h-3 bg-gray-500 rounded-full"></span>
                    <span class="text-gray-400 text-sm">Connecting...</span>
                </div>
                <span id="last-update" class="text-gray-400 text-sm">--</span>
            </div>
        </div>
    </header>

    <main class="p-6">
        <!-- Metrics Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <!-- Total Vulnerabilities -->
            <div class="metric-card bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-gray-400 text-sm font-medium">Total Vulnerabilities</span>
                    <span class="text-blue-400">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                        </svg>
                    </span>
                </div>
                <div class="text-3xl font-bold" id="total-vulns">--</div>
                <div class="mt-2 flex space-x-2 text-xs">
                    <span class="px-2 py-1 bg-red-900/50 text-red-400 rounded">C: <span id="critical-count">-</span></span>
                    <span class="px-2 py-1 bg-orange-900/50 text-orange-400 rounded">H: <span id="high-count">-</span></span>
                    <span class="px-2 py-1 bg-yellow-900/50 text-yellow-400 rounded">M: <span id="medium-count">-</span></span>
                    <span class="px-2 py-1 bg-green-900/50 text-green-400 rounded">L: <span id="low-count">-</span></span>
                </div>
            </div>

            <!-- Anomalies (24h) -->
            <div class="metric-card bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-gray-400 text-sm font-medium">Anomalies (24h)</span>
                    <span class="text-purple-400">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M13 10V3L4 14h7v7l9-11h-7z"/>
                        </svg>
                    </span>
                </div>
                <div class="text-3xl font-bold" id="anomalies-24h">--</div>
                <div class="mt-2 text-gray-400 text-sm">Network traffic anomalies</div>
            </div>

            <!-- SBOM Risk Score -->
            <div class="metric-card bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-gray-400 text-sm font-medium">SBOM Risk Score</span>
                    <span class="text-yellow-400">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                        </svg>
                    </span>
                </div>
                <div class="text-3xl font-bold" id="sbom-risk">--</div>
                <div class="w-full bg-gray-700 rounded-full h-2 mt-3">
                    <div id="sbom-risk-bar" class="bg-yellow-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
            </div>

            <!-- Threat Intel Updates -->
            <div class="metric-card bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-gray-400 text-sm font-medium">Threat Intel Updates</span>
                    <span class="text-green-400">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                        </svg>
                    </span>
                </div>
                <div class="text-3xl font-bold" id="threat-updates">--</div>
                <div class="mt-2 text-gray-400 text-sm">New CVEs today</div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Vulnerability Trend Chart -->
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 class="text-lg font-semibold mb-4">Vulnerability Trend (7 Days)</h3>
                <canvas id="vuln-chart" height="200"></canvas>
            </div>

            <!-- Severity Distribution -->
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 class="text-lg font-semibold mb-4">Severity Distribution</h3>
                <canvas id="severity-chart" height="200"></canvas>
            </div>
        </div>

        <!-- Alerts Section -->
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold">Live Alerts</h3>
                <span class="flex items-center space-x-2 text-sm text-gray-400">
                    <span class="w-2 h-2 bg-green-500 rounded-full pulse"></span>
                    <span>Live</span>
                </span>
            </div>
            <div id="alerts-container" class="space-y-3 max-h-80 overflow-y-auto">
                <div class="text-gray-500 text-center py-8">Waiting for alerts...</div>
            </div>
        </div>
    </main>

    <script>
        // WebSocket connection
        let ws;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10;
        const alerts = [];
        const maxAlerts = 50;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                reconnectAttempts = 0;
                updateConnectionStatus(true);
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateConnectionStatus(false);

                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connect, 2000 * reconnectAttempts);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }

        function updateConnectionStatus(connected) {
            const status = document.getElementById('connection-status');
            if (connected) {
                status.innerHTML = `
                    <span class="w-3 h-3 bg-green-500 rounded-full"></span>
                    <span class="text-green-400 text-sm">Connected</span>
                `;
            } else {
                status.innerHTML = `
                    <span class="w-3 h-3 bg-red-500 rounded-full pulse"></span>
                    <span class="text-red-400 text-sm">Reconnecting...</span>
                `;
            }
        }

        function handleMessage(data) {
            if (data.type === 'metrics') {
                updateMetrics(data.data);
            } else if (data.type === 'alert') {
                addAlert(data.data);
            }
        }

        function updateMetrics(metrics) {
            document.getElementById('total-vulns').textContent = metrics.total_vulnerabilities;
            document.getElementById('critical-count').textContent = metrics.critical_count;
            document.getElementById('high-count').textContent = metrics.high_count;
            document.getElementById('medium-count').textContent = metrics.medium_count;
            document.getElementById('low-count').textContent = metrics.low_count;
            document.getElementById('anomalies-24h').textContent = metrics.anomalies_24h;
            document.getElementById('sbom-risk').textContent = metrics.sbom_risk_score.toFixed(1);
            document.getElementById('sbom-risk-bar').style.width = metrics.sbom_risk_score + '%';
            document.getElementById('threat-updates').textContent = metrics.threat_intel_updates;

            // Update severity chart
            updateSeverityChart(metrics);

            // Update timestamp
            const now = new Date();
            document.getElementById('last-update').textContent =
                `Last update: ${now.toLocaleTimeString()}`;
        }

        function addAlert(alert) {
            alerts.unshift(alert);
            if (alerts.length > maxAlerts) {
                alerts.pop();
            }
            renderAlerts();
        }

        function renderAlerts() {
            const container = document.getElementById('alerts-container');
            if (alerts.length === 0) {
                container.innerHTML = '<div class="text-gray-500 text-center py-8">Waiting for alerts...</div>';
                return;
            }

            container.innerHTML = alerts.map(alert => `
                <div class="alert-${alert.severity} bg-gray-700/50 rounded-lg p-4 flex items-start space-x-4">
                    <div class="flex-shrink-0">
                        ${getAlertIcon(alert.severity)}
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center justify-between">
                            <span class="font-medium">${alert.id}</span>
                            <span class="text-xs text-gray-400">${new Date(alert.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <p class="text-gray-300 text-sm mt-1">${alert.message}</p>
                        <div class="flex items-center space-x-2 mt-2">
                            <span class="px-2 py-0.5 text-xs rounded bg-gray-600">${alert.type}</span>
                            <span class="px-2 py-0.5 text-xs rounded ${getSeverityClass(alert.severity)}">${alert.severity}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function getAlertIcon(severity) {
            const colors = {
                critical: 'text-red-400',
                high: 'text-orange-400',
                medium: 'text-yellow-400',
                low: 'text-green-400'
            };
            return `<svg class="w-5 h-5 ${colors[severity] || 'text-gray-400'}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
            </svg>`;
        }

        function getSeverityClass(severity) {
            const classes = {
                critical: 'bg-red-900/50 text-red-400',
                high: 'bg-orange-900/50 text-orange-400',
                medium: 'bg-yellow-900/50 text-yellow-400',
                low: 'bg-green-900/50 text-green-400'
            };
            return classes[severity] || 'bg-gray-600 text-gray-300';
        }

        // Initialize charts
        let vulnChart, severityChart;

        function initCharts() {
            // Vulnerability trend chart
            const vulnCtx = document.getElementById('vuln-chart').getContext('2d');
            vulnChart = new Chart(vulnCtx, {
                type: 'line',
                data: {
                    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Today'],
                    datasets: [{
                        label: 'Vulnerabilities',
                        data: [65, 70, 68, 72, 75, 78, 80],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#9ca3af' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#9ca3af' }
                        }
                    }
                }
            });

            // Severity distribution chart
            const severityCtx = document.getElementById('severity-chart').getContext('2d');
            severityChart = new Chart(severityCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Critical', 'High', 'Medium', 'Low'],
                    datasets: [{
                        data: [5, 15, 30, 50],
                        backgroundColor: ['#ef4444', '#f97316', '#eab308', '#22c55e'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: { color: '#9ca3af' }
                        }
                    }
                }
            });
        }

        function updateSeverityChart(metrics) {
            if (severityChart) {
                severityChart.data.datasets[0].data = [
                    metrics.critical_count,
                    metrics.high_count,
                    metrics.medium_count,
                    metrics.low_count
                ];
                severityChart.update();
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            connect();
        });
    </script>
</body>
</html>
"""


# =============================================================================
# Routes
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Serve the dashboard HTML."""
    return DASHBOARD_HTML


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "dashboard"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the dashboard server."""
    import uvicorn

    port = int(os.getenv("PORT", "3000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting dashboard on http://{host}:{port}")
    uvicorn.run(
        "medtech_ai_security.dashboard.server:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
    )


if __name__ == "__main__":
    main()
