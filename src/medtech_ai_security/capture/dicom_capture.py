"""
DICOM Live Traffic Capture

Captures and parses DICOM network traffic for security analysis.

DICOM (Digital Imaging and Communications in Medicine) uses TCP port 104 (or 11112)
for network communication. This module captures DICOM Association, C-FIND, C-STORE,
C-MOVE, and C-ECHO commands.

Usage:
    from medtech_ai_security.capture import DICOMCapture

    capture = DICOMCapture(interface="eth0")
    capture.start()

    # Process captured traffic
    for record in capture.get_records():
        print(record)

    capture.stop()

Requirements:
    - scapy (pip install scapy)
    - Root/Admin privileges for live capture
    - Or use pcap file mode for offline analysis
"""

import logging
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# DICOM Protocol Constants
# =============================================================================


class DICOMCommand(IntEnum):
    """DICOM Command Field values."""

    C_STORE_RQ = 0x0001
    C_STORE_RSP = 0x8001
    C_GET_RQ = 0x0010
    C_GET_RSP = 0x8010
    C_FIND_RQ = 0x0020
    C_FIND_RSP = 0x8020
    C_MOVE_RQ = 0x0021
    C_MOVE_RSP = 0x8021
    C_ECHO_RQ = 0x0030
    C_ECHO_RSP = 0x8030
    N_EVENT_REPORT_RQ = 0x0100
    N_EVENT_REPORT_RSP = 0x8100
    N_GET_RQ = 0x0110
    N_GET_RSP = 0x8110
    N_SET_RQ = 0x0120
    N_SET_RSP = 0x8120
    N_ACTION_RQ = 0x0130
    N_ACTION_RSP = 0x8130
    N_CREATE_RQ = 0x0140
    N_CREATE_RSP = 0x8140
    N_DELETE_RQ = 0x0150
    N_DELETE_RSP = 0x8150
    C_CANCEL_RQ = 0x0FFF


class PDUType(IntEnum):
    """DICOM PDU Type values."""

    A_ASSOCIATE_RQ = 0x01
    A_ASSOCIATE_AC = 0x02
    A_ASSOCIATE_RJ = 0x03
    P_DATA_TF = 0x04
    A_RELEASE_RQ = 0x05
    A_RELEASE_RP = 0x06
    A_ABORT = 0x07


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DICOMRecord:
    """Represents a captured DICOM network record."""

    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    pdu_type: str
    command: str | None = None
    calling_ae: str | None = None
    called_ae: str | None = None
    sop_class_uid: str | None = None
    transfer_syntax: str | None = None
    data_length: int = 0
    raw_data: bytes = field(default_factory=bytes, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "pdu_type": self.pdu_type,
            "command": self.command,
            "calling_ae": self.calling_ae,
            "called_ae": self.called_ae,
            "sop_class_uid": self.sop_class_uid,
            "transfer_syntax": self.transfer_syntax,
            "data_length": self.data_length,
        }


# =============================================================================
# DICOM Capture Class
# =============================================================================


class DICOMCapture:
    """
    Captures DICOM network traffic for security analysis.

    Supports:
    - Live capture from network interface (requires root/admin)
    - Offline analysis from pcap files
    - Real-time anomaly detection integration

    Example:
        capture = DICOMCapture(interface="eth0", ports=[104, 11112])
        capture.add_callback(my_analysis_function)
        capture.start()
        time.sleep(60)  # Capture for 60 seconds
        capture.stop()

        for record in capture.get_records():
            print(record)
    """

    # Standard DICOM ports
    DEFAULT_PORTS = [104, 11112, 4242]

    def __init__(
        self,
        interface: str | None = None,
        pcap_file: str | Path | None = None,
        ports: list[int] | None = None,
        buffer_size: int = 10000,
    ):
        """
        Initialize DICOM capture.

        Args:
            interface: Network interface for live capture (e.g., "eth0")
            pcap_file: Path to pcap file for offline analysis
            ports: List of ports to capture (default: [104, 11112, 4242])
            buffer_size: Maximum records to buffer
        """
        self.interface = interface
        self.pcap_file = Path(pcap_file) if pcap_file else None
        self.ports = ports or self.DEFAULT_PORTS
        self.buffer_size = buffer_size

        self._records: queue.Queue[DICOMRecord] = queue.Queue(maxsize=buffer_size)
        self._callbacks: list[Callable[[DICOMRecord], None]] = []
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._stats = {
            "packets_captured": 0,
            "dicom_packets": 0,
            "errors": 0,
            "start_time": None,
        }

    def add_callback(self, callback: Callable[[DICOMRecord], None]) -> None:
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
        logger.info("DICOM capture started")

    def stop(self) -> None:
        """Stop capturing traffic."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=5.0)
        logger.info(f"DICOM capture stopped. Stats: {self._stats}")

    def get_records(self) -> list[DICOMRecord]:
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
            # Import scapy here to avoid import errors if not installed
            from scapy.all import sniff, TCP

            filter_expr = " or ".join([f"tcp port {p}" for p in self.ports])

            def packet_handler(packet):
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
            logger.error(
                "scapy not installed. Install with: pip install scapy"
            )
            self._capture_simulated()
        except PermissionError:
            logger.error(
                "Permission denied. Live capture requires root/admin privileges."
            )
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
        """Simulated capture for testing when scapy not available."""
        logger.info("Running in simulated capture mode")

        while self._running:
            # Generate simulated DICOM traffic
            record = DICOMRecord(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.1.100",
                dst_ip="192.168.1.200",
                src_port=50000,
                dst_port=104,
                pdu_type="A-ASSOCIATE-RQ",
                calling_ae="WORKSTATION1",
                called_ae="PACS_SERVER",
                data_length=256,
            )

            self._add_record(record)
            time.sleep(1.0)

    def _process_packet(self, packet) -> None:
        """Process a captured packet and extract DICOM information."""
        from scapy.all import IP, TCP

        if IP not in packet or TCP not in packet:
            return

        payload = bytes(packet[TCP].payload)
        if len(payload) < 6:
            return

        # Check if this looks like a DICOM PDU
        pdu_type = payload[0]
        if pdu_type not in [1, 2, 3, 4, 5, 6, 7]:
            return

        self._stats["dicom_packets"] += 1

        # Parse PDU
        record = self._parse_pdu(
            payload,
            src_ip=packet[IP].src,
            dst_ip=packet[IP].dst,
            src_port=packet[TCP].sport,
            dst_port=packet[TCP].dport,
        )

        if record:
            self._add_record(record)

    def _parse_pdu(
        self,
        data: bytes,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
    ) -> DICOMRecord | None:
        """Parse DICOM PDU and create a record."""
        if len(data) < 6:
            return None

        pdu_type = data[0]
        pdu_length = struct.unpack(">I", data[2:6])[0]

        pdu_type_name = self._get_pdu_type_name(pdu_type)

        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            pdu_type=pdu_type_name,
            data_length=pdu_length,
            raw_data=data[:min(len(data), 1024)],  # Store first 1KB
        )

        # Parse specific PDU types
        if pdu_type == PDUType.A_ASSOCIATE_RQ and len(data) >= 74:
            record.called_ae = data[10:26].decode("ascii", errors="ignore").strip()
            record.calling_ae = data[26:42].decode("ascii", errors="ignore").strip()

        elif pdu_type == PDUType.P_DATA_TF and len(data) >= 12:
            # Try to extract command from P-DATA
            command = self._extract_command(data)
            if command:
                record.command = command

        return record

    def _get_pdu_type_name(self, pdu_type: int) -> str:
        """Get human-readable PDU type name."""
        names = {
            1: "A-ASSOCIATE-RQ",
            2: "A-ASSOCIATE-AC",
            3: "A-ASSOCIATE-RJ",
            4: "P-DATA-TF",
            5: "A-RELEASE-RQ",
            6: "A-RELEASE-RP",
            7: "A-ABORT",
        }
        return names.get(pdu_type, f"UNKNOWN-{pdu_type}")

    def _extract_command(self, data: bytes) -> str | None:
        """Extract DICOM command from P-DATA PDU."""
        # This is a simplified extraction
        # Full parsing requires understanding the DICOM command dataset
        try:
            # Look for command field in the data
            for i in range(len(data) - 2):
                if data[i:i + 2] == b"\x00\x00":  # Command Group
                    if i + 4 < len(data):
                        cmd = struct.unpack("<H", data[i + 2:i + 4])[0]
                        return self._get_command_name(cmd)
        except Exception:
            pass
        return None

    def _get_command_name(self, command: int) -> str:
        """Get human-readable command name."""
        names = {
            0x0001: "C-STORE-RQ",
            0x8001: "C-STORE-RSP",
            0x0020: "C-FIND-RQ",
            0x8020: "C-FIND-RSP",
            0x0021: "C-MOVE-RQ",
            0x8021: "C-MOVE-RSP",
            0x0030: "C-ECHO-RQ",
            0x8030: "C-ECHO-RSP",
        }
        return names.get(command, f"CMD-0x{command:04X}")

    def _add_record(self, record: DICOMRecord) -> None:
        """Add a record to the buffer and notify callbacks."""
        try:
            self._records.put_nowait(record)
        except queue.Full:
            # Remove oldest record if buffer is full
            try:
                self._records.get_nowait()
                self._records.put_nowait(record)
            except queue.Empty:
                pass

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI for DICOM capture."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Capture and analyze DICOM network traffic"
    )
    parser.add_argument(
        "--interface", "-i", help="Network interface for live capture"
    )
    parser.add_argument(
        "--pcap", "-f", help="pcap file for offline analysis"
    )
    parser.add_argument(
        "--ports", "-p", type=int, nargs="+", default=[104, 11112],
        help="DICOM ports to capture (default: 104 11112)"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=60,
        help="Capture duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--output", "-o", help="Output file for captured records (JSON)"
    )

    args = parser.parse_args()

    capture = DICOMCapture(
        interface=args.interface,
        pcap_file=args.pcap,
        ports=args.ports,
    )

    print(f"Starting DICOM capture for {args.duration} seconds...")
    capture.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nCapture interrupted")

    capture.stop()

    records = capture.get_records()
    print(f"\nCaptured {len(records)} DICOM records")
    print(f"Stats: {capture.get_stats()}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
        print(f"Records saved to {args.output}")
    else:
        for record in records[:10]:
            print(f"  {record.timestamp}: {record.pdu_type} from {record.src_ip}")


if __name__ == "__main__":
    main()
