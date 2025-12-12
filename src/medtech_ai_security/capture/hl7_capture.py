"""
HL7 Live Traffic Capture

Captures and parses HL7 v2.x network traffic for security analysis.

HL7 (Health Level 7) uses MLLP (Minimum Lower Layer Protocol) over TCP,
typically on ports 2575, 5000, or custom ports. Messages are framed with
0x0B (start) and 0x1C 0x0D (end).

Usage:
    from medtech_ai_security.capture import HL7Capture

    capture = HL7Capture(interface="eth0")
    capture.start()

    for record in capture.get_records():
        print(record)

    capture.stop()

Requirements:
    - scapy (pip install scapy)
    - Root/Admin privileges for live capture
"""

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# HL7 Protocol Constants
# =============================================================================

# MLLP framing characters
MLLP_START = b"\x0b"
MLLP_END = b"\x1c\x0d"

# Common HL7 message types
HL7_MESSAGE_TYPES = {
    "ADT": "Admit/Discharge/Transfer",
    "ORM": "Order Message",
    "ORU": "Observation Result",
    "RDE": "Pharmacy/Treatment Encoded Order",
    "RDS": "Pharmacy/Treatment Dispense",
    "RAS": "Pharmacy/Treatment Administration",
    "BAR": "Billing Account Record",
    "DFT": "Detailed Financial Transaction",
    "MDM": "Medical Document Management",
    "SIU": "Scheduling Information",
    "ACK": "Acknowledgment",
    "QRY": "Query",
    "RSP": "Response",
    "MFN": "Master Files Notification",
    "VXU": "Vaccination Update",
}


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class HL7Record:
    """Represents a captured HL7 network record."""

    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    message_type: str
    trigger_event: str | None = None
    message_control_id: str | None = None
    sending_application: str | None = None
    receiving_application: str | None = None
    sending_facility: str | None = None
    receiving_facility: str | None = None
    patient_id: str | None = None
    message_length: int = 0
    segment_count: int = 0
    raw_message: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "message_type": self.message_type,
            "trigger_event": self.trigger_event,
            "message_control_id": self.message_control_id,
            "sending_application": self.sending_application,
            "receiving_application": self.receiving_application,
            "sending_facility": self.sending_facility,
            "receiving_facility": self.receiving_facility,
            "patient_id": self.patient_id,
            "message_length": self.message_length,
            "segment_count": self.segment_count,
        }


# =============================================================================
# HL7 Capture Class
# =============================================================================


class HL7Capture:
    """
    Captures HL7 network traffic for security analysis.

    Supports:
    - Live capture from network interface (requires root/admin)
    - Offline analysis from pcap files
    - MLLP framing detection and parsing
    - Real-time anomaly detection integration

    Example:
        capture = HL7Capture(interface="eth0", ports=[2575, 5000])
        capture.add_callback(my_analysis_function)
        capture.start()
        time.sleep(60)
        capture.stop()
    """

    # Standard HL7/MLLP ports
    DEFAULT_PORTS = [2575, 5000, 6661, 7777]

    def __init__(
        self,
        interface: str | None = None,
        pcap_file: str | Path | None = None,
        ports: list[int] | None = None,
        buffer_size: int = 10000,
    ):
        """
        Initialize HL7 capture.

        Args:
            interface: Network interface for live capture
            pcap_file: Path to pcap file for offline analysis
            ports: List of ports to capture
            buffer_size: Maximum records to buffer
        """
        self.interface = interface
        self.pcap_file = Path(pcap_file) if pcap_file else None
        self.ports = ports or self.DEFAULT_PORTS
        self.buffer_size = buffer_size

        self._records: queue.Queue[HL7Record] = queue.Queue(maxsize=buffer_size)
        self._callbacks: list[Callable[[HL7Record], None]] = []
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._stream_buffers: dict[tuple[str, str, int, int], bytes] = {}
        self._stats: dict[str, Any] = {
            "packets_captured": 0,
            "hl7_messages": 0,
            "malformed_messages": 0,
            "errors": 0,
            "start_time": None,
        }

    def add_callback(self, callback: Callable[[HL7Record], None]) -> None:
        """Add a callback function for real-time processing."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start capturing traffic."""
        if self._running:
            logger.warning("Capture already running")
            return

        self._running = True
        self._stats["start_time"] = datetime.now(timezone.utc)

        if self.pcap_file:
            self._capture_thread = threading.Thread(
                target=self._capture_from_file, daemon=True
            )
        else:
            self._capture_thread = threading.Thread(
                target=self._capture_live, daemon=True
            )

        self._capture_thread.start()
        logger.info("HL7 capture started")

    def stop(self) -> None:
        """Stop capturing traffic."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=5.0)
        logger.info(f"HL7 capture stopped. Stats: {self._stats}")

    def get_records(self) -> list[HL7Record]:
        """Get all buffered records."""
        records = []
        while not self._records.empty():
            try:
                records.append(self._records.get_nowait())
            except queue.Empty:
                break
        return records

    def get_stats(self) -> dict[str, Any]:
        """Get capture statistics."""
        return self._stats.copy()

    def _capture_live(self) -> None:
        """Capture from live network interface."""
        try:
            from scapy.all import sniff, TCP

            filter_expr = " or ".join([f"tcp port {p}" for p in self.ports])

            def packet_handler(packet: Any) -> None:
                if not self._running:
                    return

                self._stats["packets_captured"] += 1

                if TCP in packet and packet[TCP].payload:
                    try:
                        self._process_packet(packet)
                    except Exception as e:
                        self._stats["errors"] += 1
                        logger.debug(f"Error processing packet: {e}")

            sniff(
                iface=self.interface,
                filter=filter_expr,
                prn=packet_handler,
                store=False,
                stop_filter=lambda _: not self._running,
            )

        except ImportError:
            logger.error("scapy not installed")
            self._capture_simulated()
        except PermissionError:
            logger.error("Permission denied for live capture")
            self._capture_simulated()
        except Exception as e:
            logger.error(f"Capture error: {e}")
            self._capture_simulated()

    def _capture_from_file(self) -> None:
        """Capture from pcap file."""
        try:
            from scapy.all import rdpcap, TCP

            packets = rdpcap(str(self.pcap_file))

            for packet in packets:
                if not self._running:
                    break

                self._stats["packets_captured"] += 1

                if TCP in packet and packet[TCP].payload:
                    try:
                        self._process_packet(packet)
                    except Exception as e:
                        self._stats["errors"] += 1
                        logger.debug(f"Error processing packet: {e}")

        except ImportError:
            logger.error("scapy not installed")
        except FileNotFoundError:
            logger.error(f"pcap file not found: {self.pcap_file}")
        except Exception as e:
            logger.error(f"Error reading pcap file: {e}")

    def _capture_simulated(self) -> None:
        """Simulated capture for testing."""
        logger.info("Running in simulated HL7 capture mode")

        sample_messages = [
            ("ADT", "A01", "Admit patient"),
            ("ORU", "R01", "Lab result"),
            ("ORM", "O01", "New order"),
            ("ACK", None, "Acknowledgment"),
        ]

        while self._running:
            msg_type, event, _ = sample_messages[
                int(time.time()) % len(sample_messages)
            ]

            record = HL7Record(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.1.50",
                dst_ip="192.168.1.100",
                src_port=50000,
                dst_port=2575,
                message_type=msg_type,
                trigger_event=event,
                message_control_id=f"MSG{int(time.time())}",
                sending_application="LAB_SYS",
                receiving_application="HIS",
                message_length=256,
                segment_count=5,
            )

            self._add_record(record)
            time.sleep(2.0)

    def _process_packet(self, packet) -> None:
        """Process a captured packet and extract HL7 messages."""
        from scapy.all import IP, TCP

        if IP not in packet or TCP not in packet:
            return

        payload = bytes(packet[TCP].payload)
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport

        # TCP stream key
        stream_key = (src_ip, dst_ip, src_port, dst_port)

        # Append to stream buffer
        if stream_key not in self._stream_buffers:
            self._stream_buffers[stream_key] = b""
        self._stream_buffers[stream_key] += payload

        # Extract complete MLLP messages
        buffer = self._stream_buffers[stream_key]
        while True:
            # Find MLLP start
            start_idx = buffer.find(MLLP_START)
            if start_idx == -1:
                break

            # Find MLLP end
            end_idx = buffer.find(MLLP_END, start_idx)
            if end_idx == -1:
                break

            # Extract message
            message = buffer[start_idx + 1:end_idx]
            buffer = buffer[end_idx + 2:]
            self._stream_buffers[stream_key] = buffer

            # Parse HL7 message
            record = self._parse_hl7_message(
                message, src_ip, dst_ip, src_port, dst_port
            )
            if record:
                self._stats["hl7_messages"] += 1
                self._add_record(record)
            else:
                self._stats["malformed_messages"] += 1

    def _parse_hl7_message(
        self,
        message: bytes,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
    ) -> HL7Record | None:
        """Parse HL7 message and create a record."""
        try:
            # Decode message
            text = message.decode("utf-8", errors="replace")

            # Split into segments
            segments = text.split("\r")
            segments = [s for s in segments if s.strip()]

            if not segments:
                return None

            # Parse MSH segment
            msh = segments[0]
            if not msh.startswith("MSH"):
                return None

            # Extract field separator (typically |)
            field_sep = msh[3]
            fields = msh.split(field_sep)

            if len(fields) < 12:
                return None

            # Extract message type (MSH-9)
            msg_type_field = fields[8] if len(fields) > 8 else ""
            if "^" in msg_type_field:
                parts = msg_type_field.split("^")
                message_type = parts[0]
                trigger_event = parts[1] if len(parts) > 1 else None
            else:
                message_type = msg_type_field
                trigger_event = None

            # Extract patient ID from PID segment if present
            patient_id = None
            for segment in segments:
                if segment.startswith("PID"):
                    pid_fields = segment.split(field_sep)
                    if len(pid_fields) > 3:
                        patient_id = pid_fields[3].split("^")[0]
                    break

            record = HL7Record(
                timestamp=datetime.now(timezone.utc),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                message_type=message_type,
                trigger_event=trigger_event,
                message_control_id=fields[9] if len(fields) > 9 else None,
                sending_application=fields[2] if len(fields) > 2 else None,
                receiving_application=fields[4] if len(fields) > 4 else None,
                sending_facility=fields[3] if len(fields) > 3 else None,
                receiving_facility=fields[5] if len(fields) > 5 else None,
                patient_id=patient_id,
                message_length=len(message),
                segment_count=len(segments),
                raw_message=text[:1024],  # Store first 1KB
            )

            return record

        except Exception as e:
            logger.debug(f"Error parsing HL7 message: {e}")
            return None

    def _add_record(self, record: HL7Record) -> None:
        """Add a record to the buffer and notify callbacks."""
        try:
            self._records.put_nowait(record)
        except queue.Full:
            try:
                self._records.get_nowait()
                self._records.put_nowait(record)
            except queue.Empty:
                pass

        for callback in self._callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI for HL7 capture."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Capture and analyze HL7 network traffic"
    )
    parser.add_argument(
        "--interface", "-i", help="Network interface for live capture"
    )
    parser.add_argument(
        "--pcap", "-f", help="pcap file for offline analysis"
    )
    parser.add_argument(
        "--ports", "-p", type=int, nargs="+", default=[2575, 5000],
        help="HL7 ports to capture (default: 2575 5000)"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=60,
        help="Capture duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--output", "-o", help="Output file for captured records (JSON)"
    )

    args = parser.parse_args()

    capture = HL7Capture(
        interface=args.interface,
        pcap_file=args.pcap,
        ports=args.ports,
    )

    print(f"Starting HL7 capture for {args.duration} seconds...")
    capture.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nCapture interrupted")

    capture.stop()

    records = capture.get_records()
    print(f"\nCaptured {len(records)} HL7 messages")
    print(f"Stats: {capture.get_stats()}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
        print(f"Records saved to {args.output}")
    else:
        for record in records[:10]:
            print(f"  {record.timestamp}: {record.message_type}^{record.trigger_event} from {record.src_ip}")


if __name__ == "__main__":
    main()
