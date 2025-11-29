"""
Synthetic Medical Device Traffic Generator

Generates realistic DICOM and HL7 network traffic patterns for training
anomaly detection models. Includes both normal and attack patterns.

DICOM Protocol Features:
- Association requests/responses
- C-STORE, C-FIND, C-MOVE, C-GET operations
- Patient/Study/Series/Image level queries
- Transfer syntax negotiation

HL7 Protocol Features:
- ADT (Admit/Discharge/Transfer) messages
- ORM (Order) messages
- ORU (Results) messages
- ACK (Acknowledgment) messages

Attack Patterns:
- DICOM: Unauthorized queries, data exfiltration, malformed packets
- HL7: Message injection, identity spoofing, protocol violations

Usage:
    from medtech_ai_security.anomaly import TrafficGenerator

    generator = TrafficGenerator()

    # Generate normal traffic
    normal_packets = generator.generate_normal_traffic(n_samples=1000)

    # Generate attack traffic
    attack_packets = generator.generate_attack_traffic(n_samples=100)
"""

import json
import logging
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMCommand(Enum):
    """DICOM command types."""
    C_STORE = 0x0001
    C_GET = 0x0010
    C_FIND = 0x0020
    C_MOVE = 0x0021
    C_ECHO = 0x0030
    N_EVENT = 0x0100
    N_GET = 0x0110
    N_SET = 0x0120
    N_ACTION = 0x0130
    N_CREATE = 0x0140
    N_DELETE = 0x0150


class HL7MessageType(Enum):
    """HL7 message types."""
    ADT_A01 = "ADT^A01"  # Admit
    ADT_A02 = "ADT^A02"  # Transfer
    ADT_A03 = "ADT^A03"  # Discharge
    ADT_A04 = "ADT^A04"  # Register
    ADT_A08 = "ADT^A08"  # Update
    ORM_O01 = "ORM^O01"  # Order
    ORU_R01 = "ORU^R01"  # Result
    ACK = "ACK"          # Acknowledgment
    QRY_A19 = "QRY^A19"  # Query


class AttackType(Enum):
    """Types of attacks on medical protocols."""
    # DICOM attacks
    DICOM_UNAUTHORIZED_QUERY = "dicom_unauthorized_query"
    DICOM_DATA_EXFILTRATION = "dicom_data_exfiltration"
    DICOM_MALFORMED_PACKET = "dicom_malformed_packet"
    DICOM_BRUTE_FORCE_AET = "dicom_brute_force_aet"
    DICOM_RANSOMWARE_PAYLOAD = "dicom_ransomware_payload"
    # HL7 attacks
    HL7_MESSAGE_INJECTION = "hl7_message_injection"
    HL7_IDENTITY_SPOOFING = "hl7_identity_spoofing"
    HL7_PROTOCOL_VIOLATION = "hl7_protocol_violation"
    HL7_DATA_TAMPERING = "hl7_data_tampering"
    HL7_DOS_FLOOD = "hl7_dos_flood"


@dataclass
class DICOMPacket:
    """Represents a DICOM network packet."""

    timestamp: float
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    calling_ae: str  # Application Entity Title
    called_ae: str
    command: DICOMCommand
    message_id: int
    affected_sop_class: str
    dataset_size: int  # bytes
    transfer_syntax: str
    is_association: bool = False
    is_release: bool = False
    is_abort: bool = False
    is_attack: bool = False
    attack_type: Optional[AttackType] = None

    def to_feature_vector(self) -> np.ndarray:
        """Convert packet to numeric feature vector for ML."""
        features = [
            # Timing features
            self.timestamp % 86400,  # Time of day (seconds)
            self.timestamp % 3600,   # Time within hour

            # Network features
            hash(self.source_ip) % 1000 / 1000,
            hash(self.dest_ip) % 1000 / 1000,
            self.source_port / 65535,
            self.dest_port / 65535,

            # DICOM features
            len(self.calling_ae) / 16,
            len(self.called_ae) / 16,
            self.command.value / 0x0200,
            self.message_id / 65535,
            hash(self.affected_sop_class) % 1000 / 1000,
            min(self.dataset_size / 10_000_000, 1.0),  # Normalize to 10MB
            hash(self.transfer_syntax) % 100 / 100,

            # Flags
            float(self.is_association),
            float(self.is_release),
            float(self.is_abort),
        ]
        return np.array(features, dtype=np.float32)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "source_ip": self.source_ip,
            "dest_ip": self.dest_ip,
            "source_port": self.source_port,
            "dest_port": self.dest_port,
            "calling_ae": self.calling_ae,
            "called_ae": self.called_ae,
            "command": self.command.name,
            "message_id": self.message_id,
            "affected_sop_class": self.affected_sop_class,
            "dataset_size": self.dataset_size,
            "transfer_syntax": self.transfer_syntax,
            "is_association": self.is_association,
            "is_release": self.is_release,
            "is_abort": self.is_abort,
            "is_attack": self.is_attack,
            "attack_type": self.attack_type.value if self.attack_type else None,
        }


@dataclass
class HL7Message:
    """Represents an HL7 message."""

    timestamp: float
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    message_type: HL7MessageType
    message_control_id: str
    sending_application: str
    sending_facility: str
    receiving_application: str
    receiving_facility: str
    patient_id: str
    message_length: int
    segment_count: int
    is_attack: bool = False
    attack_type: Optional[AttackType] = None

    def to_feature_vector(self) -> np.ndarray:
        """Convert message to numeric feature vector for ML."""
        # Map message type to numeric
        msg_type_map = {t: i for i, t in enumerate(HL7MessageType)}

        features = [
            # Timing features
            self.timestamp % 86400,
            self.timestamp % 3600,

            # Network features
            hash(self.source_ip) % 1000 / 1000,
            hash(self.dest_ip) % 1000 / 1000,
            self.source_port / 65535,
            self.dest_port / 65535,

            # HL7 features
            msg_type_map.get(self.message_type, 0) / len(HL7MessageType),
            len(self.message_control_id) / 20,
            len(self.sending_application) / 20,
            len(self.sending_facility) / 20,
            len(self.receiving_application) / 20,
            len(self.receiving_facility) / 20,
            len(self.patient_id) / 20,
            min(self.message_length / 10000, 1.0),
            min(self.segment_count / 50, 1.0),

            # Entropy of patient ID (randomness indicator)
            self._string_entropy(self.patient_id) / 4,
        ]
        return np.array(features, dtype=np.float32)

    def _string_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s:
            return 0.0
        prob = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * np.log2(p) for p in prob if p > 0)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "source_ip": self.source_ip,
            "dest_ip": self.dest_ip,
            "source_port": self.source_port,
            "dest_port": self.dest_port,
            "message_type": self.message_type.value,
            "message_control_id": self.message_control_id,
            "sending_application": self.sending_application,
            "sending_facility": self.sending_facility,
            "receiving_application": self.receiving_application,
            "receiving_facility": self.receiving_facility,
            "patient_id": self.patient_id,
            "message_length": self.message_length,
            "segment_count": self.segment_count,
            "is_attack": self.is_attack,
            "attack_type": self.attack_type.value if self.attack_type else None,
        }


class TrafficGenerator:
    """
    Generates synthetic medical device network traffic.

    Creates realistic DICOM and HL7 traffic patterns including:
    - Normal clinical workflow patterns
    - Various attack patterns for training
    """

    # Common DICOM SOP Classes
    SOP_CLASSES = [
        "1.2.840.10008.5.1.4.1.1.2",      # CT Image Storage
        "1.2.840.10008.5.1.4.1.1.4",      # MR Image Storage
        "1.2.840.10008.5.1.4.1.1.7",      # Secondary Capture
        "1.2.840.10008.5.1.4.1.1.12.1",   # X-Ray Angiographic
        "1.2.840.10008.5.1.4.1.1.128",    # PET Image Storage
        "1.2.840.10008.5.1.4.1.1.1.1",    # Digital X-Ray
        "1.2.840.10008.5.1.4.1.1.6.1",    # US Image Storage
        "1.2.840.10008.1.1",              # Verification SOP
    ]

    # Transfer Syntaxes
    TRANSFER_SYNTAXES = [
        "1.2.840.10008.1.2",       # Implicit VR Little Endian
        "1.2.840.10008.1.2.1",     # Explicit VR Little Endian
        "1.2.840.10008.1.2.4.50",  # JPEG Baseline
        "1.2.840.10008.1.2.4.70",  # JPEG Lossless
        "1.2.840.10008.1.2.4.90",  # JPEG 2000
    ]

    # Realistic AE Titles
    AE_TITLES = [
        "PACS_SERVER", "CT_SCANNER_1", "CT_SCANNER_2",
        "MR_SCANNER_1", "MR_SCANNER_2", "XRAY_ROOM_1",
        "WORKSTATION_1", "WORKSTATION_2", "WORKSTATION_3",
        "ARCHIVE_SRV", "VIEWER_WEB", "MODALITY_WL",
    ]

    # Hospital applications/facilities for HL7
    HL7_APPLICATIONS = [
        "HIS", "RIS", "LIS", "PACS", "EMR", "ADT", "ORDER_ENTRY",
    ]

    HL7_FACILITIES = [
        "MAIN_HOSPITAL", "RADIOLOGY_DEPT", "LAB_DEPT",
        "CARDIOLOGY", "ONCOLOGY", "EMERGENCY", "ICU",
    ]

    def __init__(self, seed: int = 42):
        """Initialize the traffic generator."""
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

        # Define IP ranges for different device types
        self.modality_ips = [f"10.0.1.{i}" for i in range(10, 30)]
        self.workstation_ips = [f"10.0.2.{i}" for i in range(10, 50)]
        self.server_ips = [f"10.0.0.{i}" for i in range(5, 15)]
        self.external_ips = [f"192.168.{i}.{j}" for i in range(1, 5) for j in range(1, 10)]

    def generate_normal_dicom(
        self,
        n_samples: int = 1000,
        start_time: Optional[float] = None,
    ) -> list[DICOMPacket]:
        """
        Generate normal DICOM traffic patterns.

        Simulates typical radiology workflow:
        - Morning imaging rush (7-12)
        - Afternoon reporting (13-17)
        - Evening/night maintenance
        """
        packets = []

        if start_time is None:
            start_time = datetime.now().timestamp()

        current_time = start_time

        for _ in range(n_samples):
            # Simulate realistic time progression
            hour = (current_time % 86400) / 3600

            # Higher traffic during work hours
            if 7 <= hour <= 17:
                delay = self.rng.exponential(5)  # 5 second average
            elif 17 <= hour <= 22:
                delay = self.rng.exponential(30)  # 30 second average
            else:
                delay = self.rng.exponential(120)  # 2 minute average

            current_time += delay

            # Choose operation based on time of day
            if hour < 12:  # Morning - more imaging
                command = self.rng.choice([
                    DICOMCommand.C_STORE,
                    DICOMCommand.C_STORE,
                    DICOMCommand.C_STORE,
                    DICOMCommand.C_FIND,
                    DICOMCommand.C_ECHO,
                ])
            else:  # Afternoon - more viewing
                command = self.rng.choice([
                    DICOMCommand.C_FIND,
                    DICOMCommand.C_GET,
                    DICOMCommand.C_FIND,
                    DICOMCommand.C_MOVE,
                    DICOMCommand.C_ECHO,
                ])

            # Generate packet
            if command == DICOMCommand.C_STORE:
                # Modality -> PACS
                source_ip = self.rng.choice(self.modality_ips)
                dest_ip = self.rng.choice(self.server_ips)
                calling_ae = self.rng.choice([ae for ae in self.AE_TITLES if "SCANNER" in ae or "XRAY" in ae])
                called_ae = self.rng.choice([ae for ae in self.AE_TITLES if "PACS" in ae or "ARCHIVE" in ae])
                dataset_size = int(self.rng.lognormal(16, 1))  # ~10MB average
            else:
                # Workstation -> PACS
                source_ip = self.rng.choice(self.workstation_ips)
                dest_ip = self.rng.choice(self.server_ips)
                calling_ae = self.rng.choice([ae for ae in self.AE_TITLES if "WORKSTATION" in ae or "VIEWER" in ae])
                called_ae = self.rng.choice([ae for ae in self.AE_TITLES if "PACS" in ae or "ARCHIVE" in ae])
                dataset_size = int(self.rng.exponential(1000))  # Small queries

            packet = DICOMPacket(
                timestamp=current_time,
                source_ip=source_ip,
                dest_ip=dest_ip,
                source_port=int(self.rng.randint(49152, 65535)),
                dest_port=104 if self.rng.random() > 0.1 else 11112,
                calling_ae=calling_ae,
                called_ae=called_ae,
                command=command,
                message_id=int(self.rng.randint(1, 65535)),
                affected_sop_class=self.rng.choice(self.SOP_CLASSES),
                dataset_size=dataset_size,
                transfer_syntax=self.rng.choice(self.TRANSFER_SYNTAXES),
                is_association=self.rng.random() < 0.1,
                is_attack=False,
            )
            packets.append(packet)

        logger.info(f"Generated {len(packets)} normal DICOM packets")
        return packets

    def generate_normal_hl7(
        self,
        n_samples: int = 1000,
        start_time: Optional[float] = None,
    ) -> list[HL7Message]:
        """
        Generate normal HL7 traffic patterns.

        Simulates typical hospital workflow:
        - Admissions in morning
        - Orders throughout day
        - Results in afternoon
        - Discharges in evening
        """
        messages = []

        if start_time is None:
            start_time = datetime.now().timestamp()

        current_time = start_time
        patient_counter = 1000

        for _ in range(n_samples):
            hour = (current_time % 86400) / 3600

            # Time-based delay
            if 8 <= hour <= 18:
                delay = self.rng.exponential(10)
            else:
                delay = self.rng.exponential(60)

            current_time += delay

            # Choose message type based on time
            if 6 <= hour <= 10:  # Morning - admissions
                msg_type = self.rng.choice([
                    HL7MessageType.ADT_A01,
                    HL7MessageType.ADT_A04,
                    HL7MessageType.ADT_A01,
                    HL7MessageType.ORM_O01,
                ])
            elif 10 <= hour <= 16:  # Day - orders and results
                msg_type = self.rng.choice([
                    HL7MessageType.ORM_O01,
                    HL7MessageType.ORU_R01,
                    HL7MessageType.ADT_A08,
                    HL7MessageType.ACK,
                ])
            else:  # Evening - discharges
                msg_type = self.rng.choice([
                    HL7MessageType.ADT_A03,
                    HL7MessageType.ORU_R01,
                    HL7MessageType.ACK,
                    HL7MessageType.ADT_A02,
                ])

            # Generate patient ID (realistic format)
            patient_id = f"MRN{patient_counter:08d}"
            if self.rng.random() > 0.7:  # Some repeat patients
                patient_counter += 1

            message = HL7Message(
                timestamp=current_time,
                source_ip=self.rng.choice(self.server_ips),
                dest_ip=self.rng.choice(self.server_ips),
                source_port=int(self.rng.randint(49152, 65535)),
                dest_port=2575,  # Standard HL7 port
                message_type=msg_type,
                message_control_id=f"MSG{int(current_time)}",
                sending_application=self.rng.choice(self.HL7_APPLICATIONS),
                sending_facility=self.rng.choice(self.HL7_FACILITIES),
                receiving_application=self.rng.choice(self.HL7_APPLICATIONS),
                receiving_facility=self.rng.choice(self.HL7_FACILITIES),
                patient_id=patient_id,
                message_length=int(self.rng.lognormal(7, 0.5)),  # ~1KB average
                segment_count=int(self.rng.randint(5, 25)),
                is_attack=False,
            )
            messages.append(message)

        logger.info(f"Generated {len(messages)} normal HL7 messages")
        return messages

    def generate_attack_dicom(
        self,
        n_samples: int = 100,
        attack_types: Optional[list[AttackType]] = None,
        start_time: Optional[float] = None,
    ) -> list[DICOMPacket]:
        """Generate DICOM attack traffic patterns."""
        packets = []

        if start_time is None:
            start_time = datetime.now().timestamp()

        if attack_types is None:
            attack_types = [
                AttackType.DICOM_UNAUTHORIZED_QUERY,
                AttackType.DICOM_DATA_EXFILTRATION,
                AttackType.DICOM_MALFORMED_PACKET,
                AttackType.DICOM_BRUTE_FORCE_AET,
                AttackType.DICOM_RANSOMWARE_PAYLOAD,
            ]

        current_time = start_time

        for _ in range(n_samples):
            attack_type = self.rng.choice(attack_types)
            current_time += self.rng.exponential(1)  # Fast attacks

            if attack_type == AttackType.DICOM_UNAUTHORIZED_QUERY:
                # External IP querying PACS
                packet = DICOMPacket(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.external_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=104,
                    calling_ae="UNKNOWN_" + ''.join(random.choices(string.ascii_uppercase, k=4)),
                    called_ae=self.rng.choice(self.AE_TITLES),
                    command=DICOMCommand.C_FIND,
                    message_id=int(self.rng.randint(1, 65535)),
                    affected_sop_class=self.SOP_CLASSES[0],
                    dataset_size=100,
                    transfer_syntax=self.TRANSFER_SYNTAXES[0],
                    is_attack=True,
                    attack_type=attack_type,
                )

            elif attack_type == AttackType.DICOM_DATA_EXFILTRATION:
                # Large C-GET/C-MOVE from unusual source
                packet = DICOMPacket(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.external_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=104,
                    calling_ae="EXFIL_" + ''.join(random.choices(string.ascii_uppercase, k=4)),
                    called_ae="PACS_SERVER",
                    command=self.rng.choice([DICOMCommand.C_GET, DICOMCommand.C_MOVE]),
                    message_id=int(self.rng.randint(1, 65535)),
                    affected_sop_class=self.SOP_CLASSES[0],
                    dataset_size=int(self.rng.lognormal(20, 1)),  # Very large
                    transfer_syntax=self.TRANSFER_SYNTAXES[0],
                    is_attack=True,
                    attack_type=attack_type,
                )

            elif attack_type == AttackType.DICOM_MALFORMED_PACKET:
                # Invalid values
                packet = DICOMPacket(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.workstation_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=104,
                    calling_ae="A" * 20,  # Too long
                    called_ae="",  # Empty
                    command=DICOMCommand.C_STORE,
                    message_id=0,  # Invalid
                    affected_sop_class="INVALID.SOP.CLASS",
                    dataset_size=-1,  # Invalid
                    transfer_syntax="INVALID.SYNTAX",
                    is_attack=True,
                    attack_type=attack_type,
                )

            elif attack_type == AttackType.DICOM_BRUTE_FORCE_AET:
                # Rapid AE title guessing
                packet = DICOMPacket(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.external_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=104,
                    calling_ae=''.join(random.choices(string.ascii_uppercase, k=8)),
                    called_ae=''.join(random.choices(string.ascii_uppercase, k=8)),
                    command=DICOMCommand.C_ECHO,
                    message_id=int(self.rng.randint(1, 65535)),
                    affected_sop_class="1.2.840.10008.1.1",
                    dataset_size=0,
                    transfer_syntax=self.TRANSFER_SYNTAXES[0],
                    is_association=True,
                    is_attack=True,
                    attack_type=attack_type,
                )

            else:  # RANSOMWARE_PAYLOAD
                # C-STORE with suspicious characteristics
                packet = DICOMPacket(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.external_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=104,
                    calling_ae="MALWARE_SRC",
                    called_ae="PACS_SERVER",
                    command=DICOMCommand.C_STORE,
                    message_id=int(self.rng.randint(1, 65535)),
                    affected_sop_class=self.SOP_CLASSES[0],
                    dataset_size=int(self.rng.lognormal(12, 2)),  # Unusual size
                    transfer_syntax=self.TRANSFER_SYNTAXES[0],
                    is_attack=True,
                    attack_type=attack_type,
                )

            packets.append(packet)

        logger.info(f"Generated {len(packets)} DICOM attack packets")
        return packets

    def generate_attack_hl7(
        self,
        n_samples: int = 100,
        attack_types: Optional[list[AttackType]] = None,
        start_time: Optional[float] = None,
    ) -> list[HL7Message]:
        """Generate HL7 attack traffic patterns."""
        messages = []

        if start_time is None:
            start_time = datetime.now().timestamp()

        if attack_types is None:
            attack_types = [
                AttackType.HL7_MESSAGE_INJECTION,
                AttackType.HL7_IDENTITY_SPOOFING,
                AttackType.HL7_PROTOCOL_VIOLATION,
                AttackType.HL7_DATA_TAMPERING,
                AttackType.HL7_DOS_FLOOD,
            ]

        current_time = start_time

        for _ in range(n_samples):
            attack_type = self.rng.choice(attack_types)
            current_time += self.rng.exponential(0.5)  # Very fast

            if attack_type == AttackType.HL7_MESSAGE_INJECTION:
                # Injected message from external source
                message = HL7Message(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.external_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=2575,
                    message_type=HL7MessageType.ADT_A01,
                    message_control_id=f"INJ{int(current_time)}",
                    sending_application="FAKE_APP",
                    sending_facility="FAKE_FACILITY",
                    receiving_application="EMR",
                    receiving_facility="MAIN_HOSPITAL",
                    patient_id="INJECTED_" + ''.join(random.choices(string.digits, k=6)),
                    message_length=5000,
                    segment_count=50,  # Unusually many
                    is_attack=True,
                    attack_type=attack_type,
                )

            elif attack_type == AttackType.HL7_IDENTITY_SPOOFING:
                # Spoofed sending application
                message = HL7Message(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.external_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=2575,
                    message_type=HL7MessageType.ORM_O01,
                    message_control_id=f"SPF{int(current_time)}",
                    sending_application="HIS",  # Spoofed legitimate app
                    sending_facility="MAIN_HOSPITAL",
                    receiving_application="RIS",
                    receiving_facility="RADIOLOGY_DEPT",
                    patient_id="MRN" + ''.join(random.choices(string.digits, k=8)),
                    message_length=2000,
                    segment_count=15,
                    is_attack=True,
                    attack_type=attack_type,
                )

            elif attack_type == AttackType.HL7_PROTOCOL_VIOLATION:
                # Invalid message structure
                message = HL7Message(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.workstation_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=2575,
                    message_type=HL7MessageType.ADT_A01,
                    message_control_id="",  # Empty - invalid
                    sending_application="",  # Empty
                    sending_facility="",
                    receiving_application="",
                    receiving_facility="",
                    patient_id="",
                    message_length=50,  # Too short
                    segment_count=1,  # Too few
                    is_attack=True,
                    attack_type=attack_type,
                )

            elif attack_type == AttackType.HL7_DATA_TAMPERING:
                # Modified patient data
                message = HL7Message(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.server_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=2575,
                    message_type=HL7MessageType.ADT_A08,  # Update
                    message_control_id=f"TMP{int(current_time)}",
                    sending_application="HIS",
                    sending_facility="MAIN_HOSPITAL",
                    receiving_application="EMR",
                    receiving_facility="MAIN_HOSPITAL",
                    patient_id=''.join(random.choices(string.ascii_letters + string.digits, k=20)),  # Random
                    message_length=10000,  # Large update
                    segment_count=40,
                    is_attack=True,
                    attack_type=attack_type,
                )

            else:  # DOS_FLOOD
                # Rapid messages
                message = HL7Message(
                    timestamp=current_time,
                    source_ip=self.rng.choice(self.external_ips),
                    dest_ip=self.rng.choice(self.server_ips),
                    source_port=int(self.rng.randint(49152, 65535)),
                    dest_port=2575,
                    message_type=HL7MessageType.QRY_A19,
                    message_control_id=f"DOS{int(current_time * 1000)}",
                    sending_application="FLOOD",
                    sending_facility="ATTACK",
                    receiving_application="EMR",
                    receiving_facility="MAIN_HOSPITAL",
                    patient_id="*",  # Wildcard query
                    message_length=100,
                    segment_count=3,
                    is_attack=True,
                    attack_type=attack_type,
                )

            messages.append(message)

        logger.info(f"Generated {len(messages)} HL7 attack messages")
        return messages

    def generate_dataset(
        self,
        n_normal: int = 1000,
        n_attack: int = 100,
        protocol: str = "both",
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Generate a complete dataset for training.

        Args:
            n_normal: Number of normal samples
            n_attack: Number of attack samples
            protocol: "dicom", "hl7", or "both"

        Returns:
            Tuple of (features, labels, raw_packets)
        """
        all_packets = []

        if protocol in ["dicom", "both"]:
            normal_dicom = self.generate_normal_dicom(n_normal // 2 if protocol == "both" else n_normal)
            attack_dicom = self.generate_attack_dicom(n_attack // 2 if protocol == "both" else n_attack)
            all_packets.extend(normal_dicom)
            all_packets.extend(attack_dicom)

        if protocol in ["hl7", "both"]:
            normal_hl7 = self.generate_normal_hl7(n_normal // 2 if protocol == "both" else n_normal)
            attack_hl7 = self.generate_attack_hl7(n_attack // 2 if protocol == "both" else n_attack)
            all_packets.extend(normal_hl7)
            all_packets.extend(attack_hl7)

        # Shuffle
        indices = list(range(len(all_packets)))
        self.rng.shuffle(indices)
        all_packets = [all_packets[i] for i in indices]

        # Extract features and labels
        features = np.array([p.to_feature_vector() for p in all_packets])
        labels = np.array([1 if p.is_attack else 0 for p in all_packets])

        logger.info(f"Generated dataset: {len(features)} samples, {labels.sum()} attacks ({100*labels.mean():.1f}%)")

        return features, labels, all_packets

    def save_dataset(
        self,
        path: Path | str,
        n_normal: int = 1000,
        n_attack: int = 100,
        protocol: str = "both",
    ) -> dict:
        """Generate and save dataset to disk."""
        features, labels, packets = self.generate_dataset(n_normal, n_attack, protocol)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays
        np.save(path / "features.npy", features)
        np.save(path / "labels.npy", labels)

        # Save raw packets as JSON
        packets_data = [p.to_dict() for p in packets]
        with open(path / "packets.json", "w", encoding="utf-8") as f:
            json.dump(packets_data, f, indent=2)

        # Save metadata
        metadata = {
            "n_samples": len(features),
            "n_features": features.shape[1],
            "n_normal": int((labels == 0).sum()),
            "n_attack": int((labels == 1).sum()),
            "protocol": protocol,
            "generated_at": datetime.now().isoformat(),
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved dataset to {path}")
        return metadata


def main():
    """CLI entry point for traffic generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic medical device network traffic"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/anomaly/traffic",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--n-normal",
        type=int,
        default=1000,
        help="Number of normal samples",
    )
    parser.add_argument(
        "--n-attack",
        type=int,
        default=100,
        help="Number of attack samples",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["dicom", "hl7", "both"],
        default="both",
        help="Protocol to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    generator = TrafficGenerator(seed=args.seed)
    metadata = generator.save_dataset(
        path=args.output,
        n_normal=args.n_normal,
        n_attack=args.n_attack,
        protocol=args.protocol,
    )

    print("\n" + "=" * 60)
    print("TRAFFIC GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Total samples: {metadata['n_samples']}")
    print(f"Normal: {metadata['n_normal']}")
    print(f"Attack: {metadata['n_attack']}")
    print(f"Features: {metadata['n_features']}")
    print(f"Protocol: {metadata['protocol']}")


if __name__ == "__main__":
    main()
