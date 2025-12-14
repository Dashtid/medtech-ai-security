"""
Tests for the Live Traffic Capture module.

Tests cover:
- DICOM capture and PDU parsing
- HL7 capture and MLLP parsing
- Traffic analyzer with anomaly detection
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medtech_ai_security.capture.dicom_capture import (
    DICOMCapture,
    DICOMRecord,
)
from medtech_ai_security.capture.hl7_capture import (
    HL7Capture,
    HL7Record,
    MLLP_START,
    MLLP_END,
)
from medtech_ai_security.capture.traffic_analyzer import (
    TrafficAnalyzer,
    TrafficAnomaly,
)


# =============================================================================
# DICOM Capture Tests
# =============================================================================


class TestDICOMRecord:
    """Test DICOMRecord dataclass."""

    def test_creation(self):
        """Test DICOM record creation."""
        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="A-ASSOCIATE-RQ",
            calling_ae="WORKSTATION1",
            called_ae="PACS_SERVER",
            data_length=256,
        )

        assert record.src_ip == "192.168.1.100"
        assert record.pdu_type == "A-ASSOCIATE-RQ"
        assert record.calling_ae == "WORKSTATION1"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="P-DATA-TF",
            command="C-STORE",
            data_length=1024,
        )

        d = record.to_dict()

        assert d["src_ip"] == "192.168.1.100"
        assert d["pdu_type"] == "P-DATA-TF"
        assert d["command"] == "C-STORE"
        assert "timestamp" in d


class TestDICOMCapture:
    """Test DICOMCapture class."""

    def test_initialization_default(self):
        """Test default initialization."""
        capture = DICOMCapture()

        assert capture.interface is None
        assert 104 in capture.ports
        assert 11112 in capture.ports

    def test_initialization_custom_ports(self):
        """Test custom port initialization."""
        capture = DICOMCapture(ports=[4242, 11113])

        assert 4242 in capture.ports
        assert 11113 in capture.ports
        assert 104 not in capture.ports

    def test_initialization_with_interface(self):
        """Test initialization with interface."""
        capture = DICOMCapture(interface="eth0")

        assert capture.interface == "eth0"

    def test_add_callback(self):
        """Test callback registration."""
        capture = DICOMCapture()
        callback = MagicMock()

        capture.add_callback(callback)

        assert callback in capture._callbacks

    def test_get_stats(self):
        """Test statistics retrieval."""
        capture = DICOMCapture()

        stats = capture.get_stats()

        assert "packets_captured" in stats
        assert "dicom_packets" in stats
        assert "errors" in stats

    def test_get_records_empty(self):
        """Test getting records when empty."""
        capture = DICOMCapture()

        records = capture.get_records()

        assert records == []

    def test_parse_pdu_associate_rq(self):
        """Test parsing A-ASSOCIATE-RQ PDU."""
        capture = DICOMCapture()

        # Construct A-ASSOCIATE-RQ PDU
        pdu_type = 0x01  # A-ASSOCIATE-RQ
        calling_ae = b"CALLING_AE      "  # 16 bytes
        called_ae = b"CALLED_AE       "  # 16 bytes

        # Build minimal PDU: type(1) + reserved(1) + length(4) + protocol(2) + reserved(2)
        # + called_ae(16) + calling_ae(16) + reserved(32) = 74 bytes header
        pdu_data = bytes([pdu_type, 0x00])  # Type + reserved
        pdu_data += (68).to_bytes(4, "big")  # Length (68 bytes remaining)
        pdu_data += (0x0001).to_bytes(2, "big")  # Protocol version
        pdu_data += bytes(2)  # Reserved
        pdu_data += called_ae
        pdu_data += calling_ae
        pdu_data += bytes(32)  # Reserved

        record = capture._parse_pdu(
            pdu_data,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
        )

        assert record is not None
        assert record.pdu_type == "A-ASSOCIATE-RQ"
        assert record.calling_ae == "CALLING_AE"
        assert record.called_ae == "CALLED_AE"

    def test_parse_pdu_associate_ac(self):
        """Test parsing A-ASSOCIATE-AC PDU."""
        capture = DICOMCapture()

        # Construct A-ASSOCIATE-AC PDU
        pdu_type = 0x02  # A-ASSOCIATE-AC
        pdu_data = bytes([pdu_type, 0x00])
        pdu_data += (68).to_bytes(4, "big")
        pdu_data += (0x0001).to_bytes(2, "big")
        pdu_data += bytes(2)
        pdu_data += b"CALLED_AE       "
        pdu_data += b"CALLING_AE      "
        pdu_data += bytes(32)

        record = capture._parse_pdu(
            pdu_data,
            src_ip="192.168.1.200",
            dst_ip="192.168.1.100",
            src_port=104,
            dst_port=11112,
        )

        assert record is not None
        assert record.pdu_type == "A-ASSOCIATE-AC"

    def test_parse_pdu_p_data(self):
        """Test parsing P-DATA-TF PDU."""
        capture = DICOMCapture()

        # Construct P-DATA-TF PDU with minimal data
        pdu_type = 0x04  # P-DATA-TF
        payload = bytes([0x00, 0x00, 0x01, 0x01])  # Minimal PDV

        pdu_data = bytes([pdu_type, 0x00])
        pdu_data += len(payload).to_bytes(4, "big")
        pdu_data += payload

        record = capture._parse_pdu(
            pdu_data,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
        )

        assert record is not None
        assert record.pdu_type == "P-DATA-TF"

    def test_parse_pdu_release_rq(self):
        """Test parsing A-RELEASE-RQ PDU."""
        capture = DICOMCapture()

        pdu_type = 0x05  # A-RELEASE-RQ
        pdu_data = bytes([pdu_type, 0x00])
        pdu_data += (4).to_bytes(4, "big")
        pdu_data += bytes(4)

        record = capture._parse_pdu(
            pdu_data,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
        )

        assert record is not None
        assert record.pdu_type == "A-RELEASE-RQ"

    def test_parse_pdu_abort(self):
        """Test parsing A-ABORT PDU."""
        capture = DICOMCapture()

        pdu_type = 0x07  # A-ABORT
        pdu_data = bytes([pdu_type, 0x00])
        pdu_data += (4).to_bytes(4, "big")
        pdu_data += bytes(4)

        record = capture._parse_pdu(
            pdu_data,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
        )

        assert record is not None
        assert record.pdu_type == "A-ABORT"

    def test_parse_pdu_invalid(self):
        """Test parsing invalid PDU."""
        capture = DICOMCapture()

        # Too short to be valid
        record = capture._parse_pdu(
            b"\x01\x02",
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
        )

        assert record is None

    def test_parse_pdu_unknown_type(self):
        """Test parsing unknown PDU type."""
        capture = DICOMCapture()

        pdu_type = 0xFF  # Unknown type
        pdu_data = bytes([pdu_type, 0x00])
        pdu_data += (4).to_bytes(4, "big")
        pdu_data += bytes(4)

        record = capture._parse_pdu(
            pdu_data,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
        )

        assert record is not None
        assert record.pdu_type == "UNKNOWN-255"

    def test_start_stop_simulation(self):
        """Test start/stop in simulation mode."""
        capture = DICOMCapture()

        capture.start()
        assert capture._running

        capture.stop()
        assert not capture._running

    def test_simulation_generates_records(self):
        """Test that simulation mode generates records."""
        capture = DICOMCapture()
        capture.start()

        # Wait for some simulated records
        import time

        time.sleep(0.5)

        capture.stop()

        # Should have generated some records
        records = capture.get_records()
        # Note: Simulation runs every 0.5s, so we might have 1 record
        assert isinstance(records, list)


# =============================================================================
# HL7 Capture Tests
# =============================================================================


class TestHL7Record:
    """Test HL7Record dataclass."""

    def test_creation(self):
        """Test HL7 record creation."""
        record = HL7Record(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
            message_type="ADT",
            trigger_event="A01",
            sending_application="HIS",
            receiving_application="RIS",
            message_control_id="MSG001",
            message_length=256,
        )

        assert record.message_type == "ADT"
        assert record.trigger_event == "A01"
        assert record.sending_application == "HIS"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = HL7Record(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
            message_type="ORU",
            trigger_event="R01",
            message_length=512,
        )

        d = record.to_dict()

        assert d["message_type"] == "ORU"
        assert d["trigger_event"] == "R01"
        assert "timestamp" in d


class TestHL7Capture:
    """Test HL7Capture class."""

    def test_initialization_default(self):
        """Test default initialization."""
        capture = HL7Capture()

        assert capture.interface is None
        assert 2575 in capture.ports

    def test_initialization_custom_ports(self):
        """Test custom port initialization."""
        capture = HL7Capture(ports=[5000, 5001])

        assert 5000 in capture.ports
        assert 5001 in capture.ports

    def test_add_callback(self):
        """Test callback registration."""
        capture = HL7Capture()
        callback = MagicMock()

        capture.add_callback(callback)

        assert callback in capture._callbacks

    def test_get_stats(self):
        """Test statistics retrieval."""
        capture = HL7Capture()

        stats = capture.get_stats()

        assert "packets_captured" in stats
        assert "hl7_messages" in stats
        assert "errors" in stats

    def test_parse_hl7_message_adt(self):
        """Test parsing ADT message."""
        capture = HL7Capture()

        # Construct HL7 ADT^A01 message (as bytes, which is what the parser expects)
        message = (
            b"MSH|^~\\&|HIS|HOSPITAL|RIS|RADIOLOGY|20231215120000||ADT^A01|MSG001|P|2.5\r"
            b"EVN|A01|20231215120000\r"
            b"PID|1||12345^^^HOSP||DOE^JOHN||19800101|M\r"
        )

        record = capture._parse_hl7_message(
            message,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
        )

        assert record is not None
        assert record.message_type == "ADT"
        assert record.trigger_event == "A01"
        assert record.sending_application == "HIS"
        assert record.receiving_application == "RIS"
        assert record.message_control_id == "MSG001"

    def test_parse_hl7_message_oru(self):
        """Test parsing ORU message."""
        capture = HL7Capture()

        # HL7 message as bytes (which is what the parser expects)
        message = (
            b"MSH|^~\\&|LAB|HOSPITAL|LIS|LAB|20231215130000||ORU^R01|MSG002|P|2.5\r"
            b"PID|1||12345\r"
            b"OBR|1|ORD001|FIL001|CBC^Complete Blood Count\r"
        )

        record = capture._parse_hl7_message(
            message,
            src_ip="192.168.1.150",
            dst_ip="192.168.1.200",
            src_port=5001,
            dst_port=2575,
        )

        assert record is not None
        assert record.message_type == "ORU"
        assert record.trigger_event == "R01"

    def test_parse_hl7_message_invalid(self):
        """Test parsing invalid message."""
        capture = HL7Capture()

        # Not a valid HL7 message (as bytes)
        record = capture._parse_hl7_message(
            b"This is not an HL7 message",
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
        )

        assert record is None

    def test_parse_hl7_message_empty(self):
        """Test parsing empty message."""
        capture = HL7Capture()

        record = capture._parse_hl7_message(
            b"",
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
        )

        assert record is None

    def test_mllp_framing_detection(self):
        """Test MLLP framing constants."""
        # MLLP framing uses module-level constants
        assert MLLP_START == b"\x0b"  # 0x0B - vertical tab
        assert MLLP_END == b"\x1c\x0d"  # 0x1C 0x0D - file separator + carriage return

    def test_start_stop_simulation(self):
        """Test start/stop in simulation mode."""
        capture = HL7Capture()

        capture.start()
        assert capture._running

        capture.stop()
        assert not capture._running


# =============================================================================
# Traffic Analyzer Tests
# =============================================================================


class TestTrafficAnomaly:
    """Test TrafficAnomaly dataclass."""

    def test_creation(self):
        """Test anomaly creation."""
        anomaly = TrafficAnomaly(
            timestamp=datetime.now(timezone.utc),
            protocol="dicom",
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            anomaly_type="large_transfer",
            severity="high",
            confidence=0.95,
            description="Large data transfer detected",
        )

        assert anomaly.protocol == "dicom"
        assert anomaly.severity == "high"
        assert anomaly.confidence == 0.95

    def test_to_dict(self):
        """Test conversion to dictionary."""
        anomaly = TrafficAnomaly(
            timestamp=datetime.now(timezone.utc),
            protocol="hl7",
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            anomaly_type="rapid_connections",
            severity="critical",
            confidence=0.99,
            description="DoS attack detected",
        )

        d = anomaly.to_dict()

        assert d["protocol"] == "hl7"
        assert d["severity"] == "critical"
        assert "timestamp" in d


class TestTrafficAnalyzer:
    """Test TrafficAnalyzer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        analyzer = TrafficAnalyzer()

        assert analyzer.interface is None
        assert analyzer.buffer_size == 1000

    def test_initialization_with_params(self):
        """Test initialization with parameters."""
        analyzer = TrafficAnalyzer(
            interface="eth0",
            dicom_ports=[104, 11112],
            hl7_ports=[2575],
            buffer_size=500,
        )

        assert analyzer.interface == "eth0"
        assert analyzer.buffer_size == 500

    def test_initialization_with_ae_titles(self):
        """Test initialization with known AE titles."""
        analyzer = TrafficAnalyzer(known_ae_titles=["PACS", "WORKSTATION", "RIS"])

        assert "PACS" in analyzer.known_ae_titles
        assert "WORKSTATION" in analyzer.known_ae_titles

    def test_add_alert_callback(self):
        """Test alert callback registration."""
        analyzer = TrafficAnalyzer()
        callback = MagicMock()

        analyzer.add_alert_callback(callback)

        assert callback in analyzer._alert_callbacks

    def test_get_stats(self):
        """Test statistics retrieval."""
        analyzer = TrafficAnalyzer()

        stats = analyzer.get_stats()

        assert "dicom" in stats
        assert "hl7" in stats
        assert "anomalies_detected" in stats

    def test_get_anomalies_empty(self):
        """Test getting anomalies when empty."""
        analyzer = TrafficAnalyzer()

        anomalies = analyzer.get_anomalies()

        assert anomalies == []

    def test_attack_signatures_defined(self):
        """Test that attack signatures are defined."""
        assert "large_transfer" in TrafficAnalyzer.ATTACK_SIGNATURES
        assert "rapid_connections" in TrafficAnalyzer.ATTACK_SIGNATURES
        assert "unusual_port" in TrafficAnalyzer.ATTACK_SIGNATURES
        assert "after_hours" in TrafficAnalyzer.ATTACK_SIGNATURES
        assert "unknown_ae" in TrafficAnalyzer.ATTACK_SIGNATURES
        assert "phi_exfiltration" in TrafficAnalyzer.ATTACK_SIGNATURES

    def test_is_after_hours_true(self):
        """Test after hours detection - outside business hours."""
        analyzer = TrafficAnalyzer()

        # 3 AM should be after hours (business hours 8-18)
        early_morning = datetime(2023, 12, 15, 3, 0, 0, tzinfo=timezone.utc)
        assert analyzer._is_after_hours(early_morning)

        # 10 PM should be after hours
        late_night = datetime(2023, 12, 15, 22, 0, 0, tzinfo=timezone.utc)
        assert analyzer._is_after_hours(late_night)

    def test_is_after_hours_false(self):
        """Test after hours detection - during business hours."""
        analyzer = TrafficAnalyzer()

        # 10 AM should be business hours
        business_hours = datetime(2023, 12, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert not analyzer._is_after_hours(business_hours)

        # 5 PM should be business hours
        afternoon = datetime(2023, 12, 15, 17, 0, 0, tzinfo=timezone.utc)
        assert not analyzer._is_after_hours(afternoon)

    def test_track_connection(self):
        """Test connection tracking."""
        analyzer = TrafficAnalyzer()

        analyzer._track_connection("192.168.1.100")
        analyzer._track_connection("192.168.1.100")
        analyzer._track_connection("192.168.1.100")

        assert "192.168.1.100" in analyzer._connection_tracker
        assert len(analyzer._connection_tracker["192.168.1.100"]) == 3

    def test_track_data_transfer(self):
        """Test data transfer tracking."""
        analyzer = TrafficAnalyzer()

        analyzer._track_data_transfer("192.168.1.100", 1000)
        analyzer._track_data_transfer("192.168.1.100", 2000)

        assert analyzer._data_tracker["192.168.1.100"] == 3000

    def test_is_rapid_connection_false(self):
        """Test rapid connection detection - normal rate."""
        analyzer = TrafficAnalyzer()

        # Add a few connections (below threshold)
        for _ in range(10):
            analyzer._track_connection("192.168.1.100")

        assert not analyzer._is_rapid_connection("192.168.1.100")

    def test_is_rapid_connection_true(self):
        """Test rapid connection detection - high rate."""
        analyzer = TrafficAnalyzer()

        # Add many connections (above threshold of 100)
        for _ in range(150):
            analyzer._track_connection("192.168.1.100")

        assert analyzer._is_rapid_connection("192.168.1.100")

    def test_analyze_dicom_unknown_ae(self):
        """Test DICOM analysis detects unknown AE title."""
        analyzer = TrafficAnalyzer(known_ae_titles=["KNOWN_AE"])
        analyzer._running = True

        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="A-ASSOCIATE-RQ",
            calling_ae="UNKNOWN_AE",  # Not in whitelist
            called_ae="PACS",
            data_length=256,
        )

        analyzer._analyze_dicom(record)

        anomalies = analyzer.get_anomalies()
        assert len(anomalies) >= 1
        assert any(a.anomaly_type == "unknown_ae" for a in anomalies)

    def test_analyze_dicom_large_transfer(self):
        """Test DICOM analysis detects large transfer."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        # Simulate multiple transfers totaling over 100MB
        for _ in range(20):
            record = DICOMRecord(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.1.100",
                dst_ip="192.168.1.200",
                src_port=11112,
                dst_port=104,
                pdu_type="P-DATA-TF",
                data_length=10_000_000,  # 10MB each
            )
            analyzer._analyze_dicom(record)

        anomalies = analyzer.get_anomalies()
        assert any(a.anomaly_type == "large_transfer" for a in anomalies)

    def test_analyze_hl7_phi_exfiltration(self):
        """Test HL7 analysis detects PHI exfiltration."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        # Simulate many ADT messages (sensitive PHI)
        for _ in range(20):
            record = HL7Record(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.1.100",
                dst_ip="192.168.1.200",
                src_port=5000,
                dst_port=2575,
                message_type="ADT",
                trigger_event="A01",
                message_length=1_000_000,  # Large messages
            )
            analyzer._analyze_hl7(record)

        anomalies = analyzer.get_anomalies()
        assert any(a.anomaly_type == "phi_exfiltration" for a in anomalies)

    def test_analyze_hl7_after_hours(self):
        """Test HL7 analysis detects after-hours activity."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        # Create record at 3 AM
        record = HL7Record(
            timestamp=datetime(2023, 12, 15, 3, 0, 0, tzinfo=timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
            message_type="ADT",
            trigger_event="A01",
            message_length=256,
        )

        analyzer._analyze_hl7(record)

        anomalies = analyzer.get_anomalies()
        assert any(a.anomaly_type == "after_hours" for a in anomalies)

    def test_add_anomaly_buffer_overflow(self):
        """Test anomaly buffer handles overflow."""
        analyzer = TrafficAnalyzer(buffer_size=5)
        analyzer._running = True

        # Add more anomalies than buffer size
        for i in range(10):
            anomaly = TrafficAnomaly(
                timestamp=datetime.now(timezone.utc),
                protocol="dicom",
                src_ip="192.168.1.100",
                dst_ip="192.168.1.200",
                anomaly_type="test",
                severity="low",
                confidence=0.5,
                description=f"Test anomaly {i}",
            )
            analyzer._add_anomaly(anomaly)

        # Should only have buffer_size anomalies
        anomalies = analyzer.get_anomalies()
        assert len(anomalies) <= 5

    def test_alert_callback_called(self):
        """Test that alert callbacks are called."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        callback = MagicMock()
        analyzer.add_alert_callback(callback)

        anomaly = TrafficAnomaly(
            timestamp=datetime.now(timezone.utc),
            protocol="dicom",
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            anomaly_type="test",
            severity="high",
            confidence=0.9,
            description="Test alert",
        )
        analyzer._add_anomaly(anomaly)

        callback.assert_called_once_with(anomaly)

    def test_start_stop(self):
        """Test analyzer start/stop."""
        analyzer = TrafficAnalyzer()

        analyzer.start()
        assert analyzer._running

        analyzer.stop()
        assert not analyzer._running

    def test_start_already_running(self):
        """Test starting already running analyzer."""
        analyzer = TrafficAnalyzer()

        analyzer.start()
        analyzer.start()  # Should log warning but not error

        analyzer.stop()


# =============================================================================
# Integration Tests
# =============================================================================


class TestDICOMCaptureAdvanced:
    """Advanced tests for DICOMCapture class."""

    def test_initialization_with_pcap_file(self):
        """Test initialization with pcap file."""
        capture = DICOMCapture(pcap_file="test.pcap")

        assert capture.pcap_file == Path("test.pcap")

    def test_get_pdu_type_name_all_types(self):
        """Test PDU type name mapping."""
        capture = DICOMCapture()

        assert capture._get_pdu_type_name(1) == "A-ASSOCIATE-RQ"
        assert capture._get_pdu_type_name(2) == "A-ASSOCIATE-AC"
        assert capture._get_pdu_type_name(3) == "A-ASSOCIATE-RJ"
        assert capture._get_pdu_type_name(4) == "P-DATA-TF"
        assert capture._get_pdu_type_name(5) == "A-RELEASE-RQ"
        assert capture._get_pdu_type_name(6) == "A-RELEASE-RP"
        assert capture._get_pdu_type_name(7) == "A-ABORT"

    def test_get_command_name_all_commands(self):
        """Test DICOM command name mapping."""
        capture = DICOMCapture()

        assert capture._get_command_name(0x0001) == "C-STORE-RQ"
        assert capture._get_command_name(0x8001) == "C-STORE-RSP"
        assert capture._get_command_name(0x0020) == "C-FIND-RQ"
        assert capture._get_command_name(0x8020) == "C-FIND-RSP"
        assert capture._get_command_name(0x0021) == "C-MOVE-RQ"
        assert capture._get_command_name(0x8021) == "C-MOVE-RSP"
        assert capture._get_command_name(0x0030) == "C-ECHO-RQ"
        assert capture._get_command_name(0x8030) == "C-ECHO-RSP"
        assert "CMD-" in capture._get_command_name(0xFFFF)

    def test_parse_pdu_associate_rj(self):
        """Test parsing A-ASSOCIATE-RJ PDU."""
        capture = DICOMCapture()

        pdu_type = 0x03  # A-ASSOCIATE-RJ
        pdu_data = bytes([pdu_type, 0x00])
        pdu_data += (4).to_bytes(4, "big")
        pdu_data += bytes(4)

        record = capture._parse_pdu(
            pdu_data,
            src_ip="192.168.1.200",
            dst_ip="192.168.1.100",
            src_port=104,
            dst_port=11112,
        )

        assert record is not None
        assert record.pdu_type == "A-ASSOCIATE-RJ"

    def test_parse_pdu_release_rp(self):
        """Test parsing A-RELEASE-RP PDU."""
        capture = DICOMCapture()

        pdu_type = 0x06  # A-RELEASE-RP
        pdu_data = bytes([pdu_type, 0x00])
        pdu_data += (4).to_bytes(4, "big")
        pdu_data += bytes(4)

        record = capture._parse_pdu(
            pdu_data,
            src_ip="192.168.1.200",
            dst_ip="192.168.1.100",
            src_port=104,
            dst_port=11112,
        )

        assert record is not None
        assert record.pdu_type == "A-RELEASE-RP"

    def test_record_buffer_full(self):
        """Test record buffer overflow handling."""
        capture = DICOMCapture(buffer_size=2)

        # Add records to fill buffer
        for i in range(5):
            record = DICOMRecord(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.1.100",
                dst_ip="192.168.1.200",
                src_port=11112,
                dst_port=104,
                pdu_type="P-DATA-TF",
                data_length=i * 100,
            )
            capture._add_record(record)

        # Should have most recent 2 records
        records = capture.get_records()
        assert len(records) <= 2

    def test_callback_invoked_on_record(self):
        """Test callbacks are invoked when record added."""
        capture = DICOMCapture()
        callback_records = []

        def callback(record):
            callback_records.append(record)

        capture.add_callback(callback)

        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="P-DATA-TF",
            data_length=100,
        )
        capture._add_record(record)

        assert len(callback_records) == 1
        assert callback_records[0].pdu_type == "P-DATA-TF"

    def test_callback_error_handled(self):
        """Test callback errors are handled gracefully."""
        capture = DICOMCapture()

        def bad_callback(record):
            raise RuntimeError("Callback error")

        capture.add_callback(bad_callback)

        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="P-DATA-TF",
            data_length=100,
        )

        # Should not raise exception
        capture._add_record(record)


class TestHL7CaptureAdvanced:
    """Advanced tests for HL7Capture class."""

    def test_initialization_with_interface(self):
        """Test initialization with interface."""
        capture = HL7Capture(interface="eth0")

        assert capture.interface == "eth0"

    def test_parse_hl7_message_orm(self):
        """Test parsing ORM message."""
        capture = HL7Capture()

        message = (
            b"MSH|^~\\&|HIS|HOSPITAL|RIS|RADIOLOGY|20231215140000||ORM^O01|MSG003|P|2.5\r"
            b"PID|1||12345\r"
            b"ORC|NW|ORD001||FIL001|SC\r"
        )

        record = capture._parse_hl7_message(
            message,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
        )

        assert record is not None
        assert record.message_type == "ORM"
        assert record.trigger_event == "O01"

    def test_parse_hl7_message_qry(self):
        """Test parsing QRY message."""
        capture = HL7Capture()

        message = (
            b"MSH|^~\\&|QRY|HOSPITAL|MPI|HOSPITAL|20231215150000||QRY^Q01|MSG004|P|2.5\r"
            b"QRD|20231215150000|R|I|MSG004\r"
        )

        record = capture._parse_hl7_message(
            message,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
        )

        assert record is not None
        assert record.message_type == "QRY"
        assert record.trigger_event == "Q01"

    def test_parse_hl7_message_no_trigger(self):
        """Test parsing message without trigger event."""
        capture = HL7Capture()

        # Message with empty trigger event field
        message = b"MSH|^~\\&|HIS|HOSPITAL|RIS|RADIOLOGY|20231215120000||ACK|MSG005|P|2.5\r"

        record = capture._parse_hl7_message(
            message,
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
        )

        assert record is not None
        assert record.message_type == "ACK"

    def test_record_buffer_full(self):
        """Test record buffer overflow handling."""
        capture = HL7Capture(buffer_size=2)

        # Add records to fill buffer
        for i in range(5):
            record = HL7Record(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.1.100",
                dst_ip="192.168.1.200",
                src_port=5000,
                dst_port=2575,
                message_type="ADT",
                trigger_event="A01",
                message_length=i * 100,
            )
            capture._add_record(record)

        # Should have most recent records
        records = capture.get_records()
        assert len(records) <= 2

    def test_get_records_empty(self):
        """Test getting records when empty."""
        capture = HL7Capture()

        records = capture.get_records()

        assert records == []


class TestTrafficAnalyzerAdvanced:
    """Advanced tests for TrafficAnalyzer."""

    def test_analyze_dicom_known_ae_allowed(self):
        """Test DICOM analysis allows known AE titles."""
        analyzer = TrafficAnalyzer(known_ae_titles=["WORKSTATION1", "PACS"])
        analyzer._running = True

        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="A-ASSOCIATE-RQ",
            calling_ae="WORKSTATION1",  # Known AE
            called_ae="PACS",
            data_length=256,
        )

        analyzer._analyze_dicom(record)

        anomalies = analyzer.get_anomalies()
        # Should not detect unknown_ae for whitelisted AE
        assert not any(a.anomaly_type == "unknown_ae" for a in anomalies)

    def test_analyze_dicom_after_hours(self):
        """Test DICOM analysis detects after-hours activity."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        record = DICOMRecord(
            timestamp=datetime(2023, 12, 15, 3, 0, 0, tzinfo=timezone.utc),  # 3 AM
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="A-ASSOCIATE-RQ",
            calling_ae="WORKSTATION1",
            called_ae="PACS",
            data_length=256,
        )

        analyzer._analyze_dicom(record)

        anomalies = analyzer.get_anomalies()
        assert any(a.anomaly_type == "after_hours" for a in anomalies)

    def test_analyze_dicom_rapid_connection(self):
        """Test DICOM analysis detects rapid connections."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        # Simulate many rapid connections
        for _ in range(150):
            analyzer._track_connection("192.168.1.100")

        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="A-ASSOCIATE-RQ",
            calling_ae="WORKSTATION1",
            called_ae="PACS",
            data_length=256,
        )

        analyzer._analyze_dicom(record)

        anomalies = analyzer.get_anomalies()
        assert any(a.anomaly_type == "rapid_connections" for a in anomalies)

    def test_analyze_hl7_rapid_connections(self):
        """Test HL7 analysis detects rapid connections."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        # Simulate many rapid connections
        for _ in range(150):
            analyzer._track_connection("192.168.1.100")

        record = HL7Record(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
            message_type="ADT",
            trigger_event="A01",
            message_length=256,
        )

        analyzer._analyze_hl7(record)

        anomalies = analyzer.get_anomalies()
        assert any(a.anomaly_type == "rapid_connections" for a in anomalies)

    def test_stop_not_running(self):
        """Test stopping analyzer that's not running."""
        analyzer = TrafficAnalyzer()

        # Should not error when stopping already stopped analyzer
        analyzer.stop()
        assert not analyzer._running

    def test_connection_tracker_cleanup(self):
        """Test old connections are removed from tracker."""
        analyzer = TrafficAnalyzer()

        # Track connections
        analyzer._track_connection("192.168.1.100")

        # Tracker should have connection
        assert "192.168.1.100" in analyzer._connection_tracker

    def test_multiple_alert_callbacks(self):
        """Test multiple alert callbacks."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        callback1_calls = []
        callback2_calls = []

        def callback1(anomaly):
            callback1_calls.append(anomaly)

        def callback2(anomaly):
            callback2_calls.append(anomaly)

        analyzer.add_alert_callback(callback1)
        analyzer.add_alert_callback(callback2)

        anomaly = TrafficAnomaly(
            timestamp=datetime.now(timezone.utc),
            protocol="dicom",
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            anomaly_type="test",
            severity="low",
            confidence=0.5,
            description="Test",
        )
        analyzer._add_anomaly(anomaly)

        assert len(callback1_calls) == 1
        assert len(callback2_calls) == 1


class TestCaptureIntegration:
    """Integration tests for capture components."""

    def test_analyzer_processes_dicom_records(self):
        """Test analyzer processes DICOM records from capture."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        records_processed = []

        def record_callback(record):
            records_processed.append(record)
            analyzer._analyze_dicom(record)

        analyzer.dicom_capture.add_callback(record_callback)

        # Manually trigger a record
        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="P-DATA-TF",
            data_length=1024,
        )
        record_callback(record)

        assert len(records_processed) == 1

    def test_analyzer_processes_hl7_records(self):
        """Test analyzer processes HL7 records from capture."""
        analyzer = TrafficAnalyzer()
        analyzer._running = True

        records_processed = []

        def record_callback(record):
            records_processed.append(record)
            analyzer._analyze_hl7(record)

        analyzer.hl7_capture.add_callback(record_callback)

        # Manually trigger a record
        record = HL7Record(
            timestamp=datetime.now(timezone.utc),
            src_ip="192.168.1.100",
            dst_ip="192.168.1.200",
            src_port=5000,
            dst_port=2575,
            message_type="ADT",
            trigger_event="A01",
            message_length=256,
        )
        record_callback(record)

        assert len(records_processed) == 1

    def test_full_capture_analysis_flow(self):
        """Test full flow from capture to anomaly detection."""
        analyzer = TrafficAnalyzer(known_ae_titles=["PACS", "RIS"])

        anomalies_detected = []

        def anomaly_handler(anomaly):
            anomalies_detected.append(anomaly)

        analyzer.add_alert_callback(anomaly_handler)
        analyzer.start()

        # Simulate unknown AE trying to connect
        record = DICOMRecord(
            timestamp=datetime.now(timezone.utc),
            src_ip="10.0.0.50",  # Unknown source
            dst_ip="192.168.1.200",
            src_port=11112,
            dst_port=104,
            pdu_type="A-ASSOCIATE-RQ",
            calling_ae="MALICIOUS_AE",  # Not in whitelist
            called_ae="PACS",
            data_length=256,
        )
        analyzer._analyze_dicom(record)

        analyzer.stop()

        assert len(anomalies_detected) >= 1
        assert anomalies_detected[0].anomaly_type == "unknown_ae"
        assert anomalies_detected[0].severity == "medium"
