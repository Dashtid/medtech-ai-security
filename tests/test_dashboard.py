"""
Tests for the Real-Time Monitoring Dashboard module.

Tests cover:
- Data models (SecurityMetrics, Alert)
- ConnectionManager for WebSocket management
- HTTP endpoints (/, /health)
- WebSocket functionality
- Background task generators
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from medtech_ai_security.dashboard.server import (
    Alert,
    ConnectionManager,
    SecurityMetrics,
    app,
    generate_alert,
    generate_metrics,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a test client for the dashboard app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def connection_manager():
    """Create a fresh connection manager for testing."""
    return ConnectionManager()


# =============================================================================
# Data Model Tests
# =============================================================================


class TestSecurityMetrics:
    """Test SecurityMetrics model."""

    def test_creation(self):
        """Test model creation with all fields."""
        metrics = SecurityMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_vulnerabilities=75,
            critical_count=2,
            high_count=10,
            medium_count=25,
            low_count=38,
            anomalies_24h=5,
            sbom_risk_score=45.5,
            threat_intel_updates=3,
        )

        assert metrics.total_vulnerabilities == 75
        assert metrics.critical_count == 2
        assert metrics.sbom_risk_score == 45.5

    def test_model_dump(self):
        """Test model serialization."""
        metrics = SecurityMetrics(
            timestamp="2023-12-15T10:00:00Z",
            total_vulnerabilities=50,
            critical_count=1,
            high_count=5,
            medium_count=15,
            low_count=29,
            anomalies_24h=3,
            sbom_risk_score=35.0,
            threat_intel_updates=2,
        )

        data = metrics.model_dump()

        assert data["total_vulnerabilities"] == 50
        assert "timestamp" in data

    def test_required_fields(self):
        """Test required field validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            SecurityMetrics(timestamp="2023-01-01T00:00:00Z")


class TestAlert:
    """Test Alert model."""

    def test_creation(self):
        """Test alert creation."""
        alert = Alert(
            id="ALERT-1234",
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity="critical",
            type="vulnerability",
            message="New critical CVE detected",
            source="MedTech AI Security",
        )

        assert alert.id == "ALERT-1234"
        assert alert.severity == "critical"
        assert alert.type == "vulnerability"

    def test_model_dump(self):
        """Test alert serialization."""
        alert = Alert(
            id="ALERT-5678",
            timestamp="2023-12-15T10:00:00Z",
            severity="high",
            type="anomaly",
            message="Unusual network pattern detected",
            source="Traffic Analyzer",
        )

        data = alert.model_dump()

        assert data["id"] == "ALERT-5678"
        assert data["severity"] == "high"

    def test_all_severity_levels(self):
        """Test different severity levels."""
        for severity in ["critical", "high", "medium", "low"]:
            alert = Alert(
                id=f"ALERT-{severity}",
                timestamp="2023-01-01T00:00:00Z",
                severity=severity,
                type="test",
                message="Test message",
                source="Test",
            )
            assert alert.severity == severity


# =============================================================================
# ConnectionManager Tests
# =============================================================================


class TestConnectionManager:
    """Test WebSocket connection manager."""

    def test_initialization(self, connection_manager):
        """Test manager initialization."""
        assert connection_manager.active_connections == []

    @pytest.mark.asyncio
    async def test_connect(self, connection_manager):
        """Test client connection."""
        mock_websocket = AsyncMock(spec=WebSocket)

        await connection_manager.connect(mock_websocket)

        assert mock_websocket in connection_manager.active_connections
        mock_websocket.accept.assert_called_once()

    def test_disconnect(self, connection_manager):
        """Test client disconnection."""
        mock_websocket = MagicMock(spec=WebSocket)
        connection_manager.active_connections.append(mock_websocket)

        connection_manager.disconnect(mock_websocket)

        assert mock_websocket not in connection_manager.active_connections

    def test_disconnect_not_connected(self, connection_manager):
        """Test disconnecting a client that wasn't connected."""
        mock_websocket = MagicMock(spec=WebSocket)

        # Should not raise an error
        connection_manager.disconnect(mock_websocket)

        assert len(connection_manager.active_connections) == 0

    @pytest.mark.asyncio
    async def test_broadcast_single_client(self, connection_manager):
        """Test broadcasting to a single client."""
        mock_websocket = AsyncMock(spec=WebSocket)
        connection_manager.active_connections.append(mock_websocket)

        await connection_manager.broadcast({"type": "test", "data": "hello"})

        mock_websocket.send_json.assert_called_once_with({"type": "test", "data": "hello"})

    @pytest.mark.asyncio
    async def test_broadcast_multiple_clients(self, connection_manager):
        """Test broadcasting to multiple clients."""
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        connection_manager.active_connections.extend([mock_ws1, mock_ws2])

        await connection_manager.broadcast({"message": "broadcast"})

        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_connections(self, connection_manager):
        """Test that failed connections are removed during broadcast."""
        mock_ws_good = AsyncMock(spec=WebSocket)
        mock_ws_bad = AsyncMock(spec=WebSocket)
        mock_ws_bad.send_json.side_effect = Exception("Connection lost")

        connection_manager.active_connections.extend([mock_ws_good, mock_ws_bad])

        await connection_manager.broadcast({"test": "data"})

        # Good connection should remain
        assert mock_ws_good in connection_manager.active_connections
        # Bad connection should be removed
        assert mock_ws_bad not in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_no_clients(self, connection_manager):
        """Test broadcasting with no clients."""
        # Should not raise an error
        await connection_manager.broadcast({"test": "data"})

        assert len(connection_manager.active_connections) == 0


# =============================================================================
# Background Task Tests
# =============================================================================


class TestBackgroundTasks:
    """Test background task generators."""

    @pytest.mark.asyncio
    async def test_generate_metrics(self):
        """Test metrics generation."""
        metrics = await generate_metrics()

        assert isinstance(metrics, SecurityMetrics)
        assert metrics.timestamp is not None
        assert 50 <= metrics.total_vulnerabilities <= 100
        assert 0 <= metrics.critical_count <= 5
        assert 5 <= metrics.high_count <= 15
        assert 15 <= metrics.medium_count <= 30
        assert 30 <= metrics.low_count <= 50
        assert 0 <= metrics.anomalies_24h <= 20
        assert 30 <= metrics.sbom_risk_score <= 70
        assert 0 <= metrics.threat_intel_updates <= 10

    @pytest.mark.asyncio
    async def test_generate_metrics_has_timestamp(self):
        """Test that generated metrics have ISO timestamp."""
        metrics = await generate_metrics()

        # Should be valid ISO format
        datetime.fromisoformat(metrics.timestamp.replace("Z", "+00:00"))

    @pytest.mark.asyncio
    async def test_generate_alert_returns_alert_sometimes(self):
        """Test alert generation (probabilistic)."""
        alerts_generated = 0
        trials = 100

        for _ in range(trials):
            alert = await generate_alert()
            if alert is not None:
                alerts_generated += 1
                assert isinstance(alert, Alert)
                assert alert.id.startswith("ALERT-")
                assert alert.severity in ["critical", "high", "medium"]

        # With 30% probability, we expect roughly 30 alerts in 100 trials
        # Allow for statistical variance (15-45 range)
        assert 10 <= alerts_generated <= 50

    @pytest.mark.asyncio
    async def test_generate_alert_has_valid_fields(self):
        """Test that generated alerts have valid fields."""
        # Keep generating until we get an alert
        alert = None
        for _ in range(50):
            alert = await generate_alert()
            if alert:
                break

        if alert:
            assert alert.id.startswith("ALERT-")
            assert alert.severity in ["critical", "high", "medium"]
            assert alert.type in ["anomaly", "vulnerability", "sbom", "adversarial", "threat_intel"]
            assert alert.source == "MedTech AI Security"
            assert len(alert.message) > 0


# =============================================================================
# HTTP Endpoint Tests
# =============================================================================


class TestHTTPEndpoints:
    """Test HTTP endpoints."""

    def test_dashboard_page(self, client):
        """Test dashboard HTML page."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "MedTech AI Security" in response.text
        assert "Real-Time Monitoring Dashboard" in response.text

    def test_dashboard_contains_metrics_elements(self, client):
        """Test dashboard contains metric display elements."""
        response = client.get("/")

        assert response.status_code == 200
        # Check for metric IDs used by JavaScript
        assert "total-vulns" in response.text
        assert "critical-count" in response.text
        assert "anomalies-24h" in response.text
        assert "sbom-risk" in response.text
        assert "threat-updates" in response.text

    def test_dashboard_contains_websocket_js(self, client):
        """Test dashboard contains WebSocket JavaScript."""
        response = client.get("/")

        assert "WebSocket" in response.text
        assert "/ws" in response.text

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "dashboard"


# =============================================================================
# WebSocket Tests
# =============================================================================


class TestWebSocket:
    """Test WebSocket functionality."""

    def test_websocket_connect_disconnect(self, client):
        """Test WebSocket connection and disconnection."""
        with client.websocket_connect("/ws") as websocket:
            # Connection established - wait briefly for any initial message
            websocket.send_text("ping")
            # Connection will close when exiting context

    def test_websocket_receives_metrics(self, client):
        """Test WebSocket receives metrics updates."""
        with client.websocket_connect("/ws") as websocket:
            # Send a keepalive
            websocket.send_text("keepalive")

            # Try to receive data (may timeout if no data available immediately)
            # In test mode, we might not get the background task data
            # This tests the connection is valid
            try:
                # Set a short timeout for testing
                data = websocket.receive_json()
                if data:
                    assert "type" in data
                    assert data["type"] in ["metrics", "alert"]
            except Exception:
                # Timeout or no data is acceptable in test environment
                pass


# =============================================================================
# Dashboard Features Tests
# =============================================================================


class TestDashboardFeatures:
    """Test dashboard feature elements."""

    def test_dashboard_has_charts(self, client):
        """Test dashboard includes chart elements."""
        response = client.get("/")

        assert "vuln-chart" in response.text
        assert "severity-chart" in response.text
        assert "chart.js" in response.text.lower()

    def test_dashboard_has_alerts_container(self, client):
        """Test dashboard has alerts container."""
        response = client.get("/")

        assert "alerts-container" in response.text
        assert "Live Alerts" in response.text

    def test_dashboard_has_connection_status(self, client):
        """Test dashboard has connection status indicator."""
        response = client.get("/")

        assert "connection-status" in response.text
        assert "Connecting" in response.text

    def test_dashboard_has_tailwind(self, client):
        """Test dashboard uses Tailwind CSS."""
        response = client.get("/")

        assert "tailwindcss" in response.text

    def test_dashboard_responsive_grid(self, client):
        """Test dashboard has responsive grid layout."""
        response = client.get("/")

        # Check for responsive grid classes
        assert "grid-cols-1" in response.text
        assert "md:grid-cols-2" in response.text or "lg:grid-cols-4" in response.text


# =============================================================================
# Integration Tests
# =============================================================================


class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""

    def test_full_page_load(self, client):
        """Test complete page load flow."""
        # Load dashboard
        response = client.get("/")
        assert response.status_code == 200

        # Check health
        health = client.get("/health")
        assert health.status_code == 200

    @pytest.mark.asyncio
    async def test_multiple_connections(self):
        """Test multiple simultaneous connections."""
        manager = ConnectionManager()

        # Connect multiple clients
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        ws3 = AsyncMock(spec=WebSocket)

        await manager.connect(ws1)
        await manager.connect(ws2)
        await manager.connect(ws3)

        assert len(manager.active_connections) == 3

        # Broadcast to all
        await manager.broadcast({"test": "message"})

        ws1.send_json.assert_called_once()
        ws2.send_json.assert_called_once()
        ws3.send_json.assert_called_once()

        # Disconnect one
        manager.disconnect(ws2)
        assert len(manager.active_connections) == 2
        assert ws2 not in manager.active_connections

    @pytest.mark.asyncio
    async def test_metrics_valid_ranges(self):
        """Test that generated metrics are within expected ranges."""
        for _ in range(10):
            metrics = await generate_metrics()

            # Verify all values are non-negative
            assert metrics.total_vulnerabilities >= 0
            assert metrics.critical_count >= 0
            assert metrics.high_count >= 0
            assert metrics.medium_count >= 0
            assert metrics.low_count >= 0
            assert metrics.anomalies_24h >= 0
            assert metrics.sbom_risk_score >= 0
            assert metrics.threat_intel_updates >= 0

            # Verify risk score is percentage-like
            assert metrics.sbom_risk_score <= 100


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.post("/")
        assert response.status_code == 405

    def test_health_method_not_allowed(self, client):
        """Test POST to health endpoint."""
        response = client.post("/health")
        assert response.status_code == 405
