"""
Live Traffic Analyzer

Integrates DICOM/HL7 capture with anomaly detection for real-time analysis.

Usage:
    from medtech_ai_security.capture import TrafficAnalyzer

    analyzer = TrafficAnalyzer(
        interface="eth0",
        model_path="models/anomaly_detector.keras"
    )
    analyzer.start()

    # Get detected anomalies
    anomalies = analyzer.get_anomalies()

    analyzer.stop()
"""

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from medtech_ai_security.capture.dicom_capture import DICOMCapture, DICOMRecord
from medtech_ai_security.capture.hl7_capture import HL7Capture, HL7Record

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class TrafficAnomaly:
    """Detected traffic anomaly."""

    timestamp: datetime
    protocol: str  # "dicom" or "hl7"
    src_ip: str
    dst_ip: str
    anomaly_type: str
    severity: str
    confidence: float
    description: str
    raw_record: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "protocol": self.protocol,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
        }


# =============================================================================
# Traffic Analyzer
# =============================================================================


class TrafficAnalyzer:
    """
    Real-time traffic analyzer for medical device security.

    Combines DICOM and HL7 capture with ML-based anomaly detection.

    Features:
    - Unified capture interface for both protocols
    - Real-time anomaly detection using trained models
    - Rule-based detection for known attack patterns
    - Alert generation with severity levels
    - Statistics and reporting

    Example:
        analyzer = TrafficAnalyzer(interface="eth0")
        analyzer.add_alert_callback(my_alert_handler)
        analyzer.start()

        time.sleep(3600)  # Run for 1 hour

        print(f"Anomalies: {len(analyzer.get_anomalies())}")
        analyzer.stop()
    """

    # Known attack signatures
    ATTACK_SIGNATURES = {
        "large_transfer": {
            "description": "Unusually large data transfer",
            "severity": "high",
            "threshold_bytes": 100_000_000,  # 100MB
        },
        "rapid_connections": {
            "description": "Rapid connection attempts (potential DoS)",
            "severity": "critical",
            "threshold_count": 100,
            "window_seconds": 60,
        },
        "unusual_port": {
            "description": "Connection on non-standard port",
            "severity": "medium",
        },
        "after_hours": {
            "description": "Activity outside business hours",
            "severity": "low",
            "business_hours": (8, 18),  # 8 AM - 6 PM
        },
        "unknown_ae": {
            "description": "Unknown DICOM Application Entity",
            "severity": "medium",
        },
        "phi_exfiltration": {
            "description": "Potential PHI data exfiltration",
            "severity": "critical",
        },
    }

    def __init__(
        self,
        interface: str | None = None,
        dicom_ports: list[int] | None = None,
        hl7_ports: list[int] | None = None,
        model_path: str | Path | None = None,
        known_ae_titles: list[str] | None = None,
        buffer_size: int = 1000,
    ):
        """
        Initialize traffic analyzer.

        Args:
            interface: Network interface for capture
            dicom_ports: DICOM ports to monitor
            hl7_ports: HL7 ports to monitor
            model_path: Path to trained anomaly detection model
            known_ae_titles: List of known DICOM AE titles (whitelist)
            buffer_size: Maximum anomalies to buffer
        """
        self.interface = interface
        self.model_path = Path(model_path) if model_path else None
        self.known_ae_titles = set(known_ae_titles or [])
        self.buffer_size = buffer_size

        # Initialize captures
        self.dicom_capture = DICOMCapture(
            interface=interface,
            ports=dicom_ports,
        )
        self.hl7_capture = HL7Capture(
            interface=interface,
            ports=hl7_ports,
        )

        # Anomaly storage
        self._anomalies: queue.Queue[TrafficAnomaly] = queue.Queue(maxsize=buffer_size)
        self._alert_callbacks: list[Callable[[TrafficAnomaly], None]] = []

        # Statistics tracking
        self._connection_tracker: dict[str, list[datetime]] = {}
        self._data_tracker: dict[str, int] = {}

        self._running = False
        self._model = None

        # Register callbacks
        self.dicom_capture.add_callback(self._analyze_dicom)
        self.hl7_capture.add_callback(self._analyze_hl7)

    def add_alert_callback(self, callback: Callable[[TrafficAnomaly], None]) -> None:
        """Add callback for real-time alerts."""
        self._alert_callbacks.append(callback)

    def start(self) -> None:
        """Start the analyzer."""
        if self._running:
            logger.warning("Analyzer already running")
            return

        self._running = True

        # Load ML model if available
        if self.model_path and self.model_path.exists():
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

        # Start captures
        self.dicom_capture.start()
        self.hl7_capture.start()

        logger.info("Traffic analyzer started")

    def stop(self) -> None:
        """Stop the analyzer."""
        self._running = False
        self.dicom_capture.stop()
        self.hl7_capture.stop()
        logger.info("Traffic analyzer stopped")

    def get_anomalies(self) -> list[TrafficAnomaly]:
        """Get all detected anomalies."""
        anomalies = []
        while not self._anomalies.empty():
            try:
                anomalies.append(self._anomalies.get_nowait())
            except queue.Empty:
                break
        return anomalies

    def get_stats(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "dicom": self.dicom_capture.get_stats(),
            "hl7": self.hl7_capture.get_stats(),
            "anomalies_detected": self._anomalies.qsize(),
            "tracked_connections": len(self._connection_tracker),
        }

    def _load_model(self) -> None:
        """Load anomaly detection model."""
        try:
            import tensorflow as tf

            self._model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"Loaded anomaly model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            self._model = None

    def _analyze_dicom(self, record: DICOMRecord) -> None:
        """Analyze a DICOM record for anomalies."""
        if not self._running:
            return

        anomalies_found = []

        # Check for unknown AE titles
        if self.known_ae_titles:
            if record.calling_ae and record.calling_ae not in self.known_ae_titles:
                anomalies_found.append(
                    TrafficAnomaly(
                        timestamp=record.timestamp,
                        protocol="dicom",
                        src_ip=record.src_ip,
                        dst_ip=record.dst_ip,
                        anomaly_type="unknown_ae",
                        severity="medium",
                        confidence=0.9,
                        description=f"Unknown AE title: {record.calling_ae}",
                        raw_record=record.to_dict(),
                    )
                )

        # Check for large data transfers
        self._track_data_transfer(record.src_ip, record.data_length)
        total_data = self._data_tracker.get(record.src_ip, 0)
        threshold = self.ATTACK_SIGNATURES["large_transfer"]["threshold_bytes"]

        if total_data > threshold:
            anomalies_found.append(
                TrafficAnomaly(
                    timestamp=record.timestamp,
                    protocol="dicom",
                    src_ip=record.src_ip,
                    dst_ip=record.dst_ip,
                    anomaly_type="large_transfer",
                    severity="high",
                    confidence=0.85,
                    description=f"Large data transfer: {total_data / 1_000_000:.1f} MB",
                    raw_record=record.to_dict(),
                )
            )

        # Check for rapid connections
        self._track_connection(record.src_ip)
        if self._is_rapid_connection(record.src_ip):
            anomalies_found.append(
                TrafficAnomaly(
                    timestamp=record.timestamp,
                    protocol="dicom",
                    src_ip=record.src_ip,
                    dst_ip=record.dst_ip,
                    anomaly_type="rapid_connections",
                    severity="critical",
                    confidence=0.95,
                    description="Rapid connection attempts detected",
                    raw_record=record.to_dict(),
                )
            )

        # Check for after-hours activity
        if self._is_after_hours(record.timestamp):
            anomalies_found.append(
                TrafficAnomaly(
                    timestamp=record.timestamp,
                    protocol="dicom",
                    src_ip=record.src_ip,
                    dst_ip=record.dst_ip,
                    anomaly_type="after_hours",
                    severity="low",
                    confidence=0.7,
                    description="DICOM activity outside business hours",
                    raw_record=record.to_dict(),
                )
            )

        # ML-based detection
        if self._model:
            ml_anomaly = self._ml_detect_dicom(record)
            if ml_anomaly:
                anomalies_found.append(ml_anomaly)

        # Add anomalies to buffer
        for anomaly in anomalies_found:
            self._add_anomaly(anomaly)

    def _analyze_hl7(self, record: HL7Record) -> None:
        """Analyze an HL7 record for anomalies."""
        if not self._running:
            return

        anomalies_found = []

        # Check for sensitive message types
        sensitive_types = ["ADT", "ORU", "MDM"]
        if record.message_type in sensitive_types:
            # Track for potential PHI exfiltration
            self._track_data_transfer(record.src_ip, record.message_length)
            total_data = self._data_tracker.get(record.src_ip, 0)

            if total_data > 10_000_000:  # 10MB of HL7 messages
                anomalies_found.append(
                    TrafficAnomaly(
                        timestamp=record.timestamp,
                        protocol="hl7",
                        src_ip=record.src_ip,
                        dst_ip=record.dst_ip,
                        anomaly_type="phi_exfiltration",
                        severity="critical",
                        confidence=0.8,
                        description=f"Large volume of {record.message_type} messages",
                        raw_record=record.to_dict(),
                    )
                )

        # Check for rapid message flow
        self._track_connection(record.src_ip)
        if self._is_rapid_connection(record.src_ip):
            anomalies_found.append(
                TrafficAnomaly(
                    timestamp=record.timestamp,
                    protocol="hl7",
                    src_ip=record.src_ip,
                    dst_ip=record.dst_ip,
                    anomaly_type="rapid_connections",
                    severity="high",
                    confidence=0.9,
                    description="Unusual HL7 message rate",
                    raw_record=record.to_dict(),
                )
            )

        # Check for after-hours activity
        if self._is_after_hours(record.timestamp):
            anomalies_found.append(
                TrafficAnomaly(
                    timestamp=record.timestamp,
                    protocol="hl7",
                    src_ip=record.src_ip,
                    dst_ip=record.dst_ip,
                    anomaly_type="after_hours",
                    severity="low",
                    confidence=0.7,
                    description="HL7 activity outside business hours",
                    raw_record=record.to_dict(),
                )
            )

        # Add anomalies
        for anomaly in anomalies_found:
            self._add_anomaly(anomaly)

    def _track_connection(self, ip: str) -> None:
        """Track connection timestamps for rate limiting detection."""
        now = datetime.now(timezone.utc)
        if ip not in self._connection_tracker:
            self._connection_tracker[ip] = []

        self._connection_tracker[ip].append(now)

        # Keep only last 5 minutes
        cutoff = now.timestamp() - 300
        self._connection_tracker[ip] = [
            t for t in self._connection_tracker[ip]
            if t.timestamp() > cutoff
        ]

    def _track_data_transfer(self, ip: str, bytes_transferred: int) -> None:
        """Track cumulative data transfer per IP."""
        if ip not in self._data_tracker:
            self._data_tracker[ip] = 0
        self._data_tracker[ip] += bytes_transferred

    def _is_rapid_connection(self, ip: str) -> bool:
        """Check if connection rate exceeds threshold."""
        window = self.ATTACK_SIGNATURES["rapid_connections"]["window_seconds"]
        threshold = self.ATTACK_SIGNATURES["rapid_connections"]["threshold_count"]

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - window

        connections = self._connection_tracker.get(ip, [])
        recent = [c for c in connections if c.timestamp() > cutoff]

        return len(recent) > threshold

    def _is_after_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is outside business hours."""
        start_hour, end_hour = self.ATTACK_SIGNATURES["after_hours"]["business_hours"]
        hour = timestamp.hour
        return hour < start_hour or hour >= end_hour

    def _ml_detect_dicom(self, record: DICOMRecord) -> TrafficAnomaly | None:
        """Use ML model for anomaly detection."""
        # This would use the trained autoencoder model
        # For now, return None (rule-based detection handles it)
        return None

    def _add_anomaly(self, anomaly: TrafficAnomaly) -> None:
        """Add anomaly to buffer and notify callbacks."""
        try:
            self._anomalies.put_nowait(anomaly)
        except queue.Full:
            try:
                self._anomalies.get_nowait()
                self._anomalies.put_nowait(anomaly)
            except queue.Empty:
                pass

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI for traffic analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze DICOM/HL7 traffic for security anomalies"
    )
    parser.add_argument(
        "--interface", "-i", help="Network interface for capture"
    )
    parser.add_argument(
        "--dicom-ports", type=int, nargs="+", default=[104, 11112],
        help="DICOM ports to monitor"
    )
    parser.add_argument(
        "--hl7-ports", type=int, nargs="+", default=[2575, 5000],
        help="HL7 ports to monitor"
    )
    parser.add_argument(
        "--model", "-m", help="Path to anomaly detection model"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=60,
        help="Analysis duration in seconds"
    )
    parser.add_argument(
        "--output", "-o", help="Output file for anomalies (JSON)"
    )

    args = parser.parse_args()

    def alert_handler(anomaly: TrafficAnomaly) -> None:
        severity_colors = {
            "critical": "\033[91m",  # Red
            "high": "\033[93m",  # Yellow
            "medium": "\033[94m",  # Blue
            "low": "\033[92m",  # Green
        }
        reset = "\033[0m"
        color = severity_colors.get(anomaly.severity, "")
        print(
            f"{color}[{anomaly.severity.upper()}]{reset} "
            f"{anomaly.timestamp.strftime('%H:%M:%S')} - "
            f"{anomaly.protocol.upper()}: {anomaly.description}"
        )

    analyzer = TrafficAnalyzer(
        interface=args.interface,
        dicom_ports=args.dicom_ports,
        hl7_ports=args.hl7_ports,
        model_path=args.model,
    )
    analyzer.add_alert_callback(alert_handler)

    print(f"Starting traffic analysis for {args.duration} seconds...")
    print("Press Ctrl+C to stop early\n")
    analyzer.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted")

    analyzer.stop()

    anomalies = analyzer.get_anomalies()
    print(f"\n{'=' * 60}")
    print(f"Analysis complete. Detected {len(anomalies)} anomalies")
    print(f"Stats: {json.dumps(analyzer.get_stats(), indent=2, default=str)}")

    if args.output and anomalies:
        with open(args.output, "w") as f:
            json.dump([a.to_dict() for a in anomalies], f, indent=2)
        print(f"Anomalies saved to {args.output}")


if __name__ == "__main__":
    main()
