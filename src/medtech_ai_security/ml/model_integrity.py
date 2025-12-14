"""
Model Integrity Verification Module.

Provides cryptographic verification of ML model integrity for FDA compliance
and security assurance. Implements hash-based verification, signature validation,
and chain-of-custody tracking for medical AI models.

Security Context:
- FDA Cybersecurity Guidance 2025: Model integrity verification required
- NIST AI RMF: Trustworthy AI requires integrity assurance
- Supply chain attacks on ML models are an emerging threat (NIST March 2025)

References:
- CycloneDX ML BOM specification
- FDA AI/ML Lifecycle Guidance
- NIST AI 100-2 E2025: Adversarial ML Taxonomy
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class HashAlgorithm(str, Enum):
    """Supported cryptographic hash algorithms."""

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"


class IntegrityStatus(str, Enum):
    """Model integrity verification status."""

    VERIFIED = "verified"
    TAMPERED = "tampered"
    UNKNOWN = "unknown"
    ERROR = "error"


class ModelFormat(str, Enum):
    """Supported model file formats."""

    PYTORCH = "pytorch"  # .pt, .pth
    TENSORFLOW = "tensorflow"  # .pb, SavedModel
    ONNX = "onnx"  # .onnx
    SAFETENSORS = "safetensors"  # .safetensors
    KERAS = "keras"  # .h5, .keras
    PICKLE = "pickle"  # .pkl (warning: security risk)
    JOBLIB = "joblib"  # .joblib
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelHash:
    """Cryptographic hash of a model file."""

    algorithm: HashAlgorithm
    digest: str
    file_path: str
    file_size: int
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm.value,
            "digest": self.digest,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelHash":
        """Create from dictionary."""
        return cls(
            algorithm=HashAlgorithm(data["algorithm"]),
            digest=data["digest"],
            file_path=data["file_path"],
            file_size=data["file_size"],
            computed_at=data.get(
                "computed_at", datetime.now(timezone.utc).isoformat()
            ),
        )


@dataclass
class IntegrityRecord:
    """Complete integrity record for a model."""

    model_name: str
    model_version: str
    model_format: ModelFormat
    hashes: list[ModelHash]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signed_by: str | None = None
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_format": self.model_format.value,
            "hashes": [h.to_dict() for h in self.hashes],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "signed_by": self.signed_by,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntegrityRecord":
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            model_version=data["model_version"],
            model_format=ModelFormat(data["model_format"]),
            hashes=[ModelHash.from_dict(h) for h in data["hashes"]],
            metadata=data.get("metadata", {}),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            signed_by=data.get("signed_by"),
            signature=data.get("signature"),
        )


@dataclass
class VerificationResult:
    """Result of integrity verification."""

    status: IntegrityStatus
    model_name: str
    model_version: str
    verified_hashes: list[str]  # List of verified algorithm names
    failed_hashes: list[str]  # List of failed algorithm names
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    verified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "verified_hashes": self.verified_hashes,
            "failed_hashes": self.failed_hashes,
            "message": self.message,
            "details": self.details,
            "verified_at": self.verified_at,
        }


# =============================================================================
# Model Integrity Verifier
# =============================================================================


class ModelIntegrityVerifier:
    """
    Verifies ML model integrity using cryptographic hashes.

    Provides functionality to:
    - Compute cryptographic hashes of model files
    - Generate integrity records for models
    - Verify model integrity against stored records
    - Sign integrity records with HMAC
    - Track chain of custody

    Example:
        verifier = ModelIntegrityVerifier()

        # Generate integrity record
        record = verifier.generate_record(
            model_path="model.pt",
            model_name="diagnostic_classifier",
            model_version="1.0.0"
        )

        # Later: verify integrity
        result = verifier.verify(model_path="model.pt", record=record)
        if result.status == IntegrityStatus.VERIFIED:
            print("Model integrity verified")
    """

    def __init__(
        self,
        algorithms: list[HashAlgorithm] | None = None,
        chunk_size: int = 8192,
    ) -> None:
        """
        Initialize the verifier.

        Args:
            algorithms: Hash algorithms to use (default: SHA256, SHA3_256)
            chunk_size: Chunk size for reading large files
        """
        self.algorithms = algorithms or [
            HashAlgorithm.SHA256,
            HashAlgorithm.SHA3_256,
        ]
        self.chunk_size = chunk_size

    def compute_hash(
        self,
        file_path: str | Path,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> ModelHash:
        """
        Compute cryptographic hash of a model file.

        Args:
            file_path: Path to the model file
            algorithm: Hash algorithm to use

        Returns:
            ModelHash with computed digest
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        hasher = self._get_hasher(algorithm)
        file_size = path.stat().st_size

        with open(path, "rb") as f:
            self._update_hash_from_file(hasher, f)

        return ModelHash(
            algorithm=algorithm,
            digest=hasher.hexdigest(),
            file_path=str(path.absolute()),
            file_size=file_size,
        )

    def compute_all_hashes(
        self,
        file_path: str | Path,
    ) -> list[ModelHash]:
        """
        Compute hashes using all configured algorithms.

        Args:
            file_path: Path to the model file

        Returns:
            List of ModelHash objects
        """
        return [
            self.compute_hash(file_path, algorithm)
            for algorithm in self.algorithms
        ]

    def generate_record(
        self,
        model_path: str | Path,
        model_name: str,
        model_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> IntegrityRecord:
        """
        Generate a complete integrity record for a model.

        Args:
            model_path: Path to the model file
            model_name: Name of the model
            model_version: Version string
            metadata: Additional metadata to include

        Returns:
            IntegrityRecord with all hash values
        """
        path = Path(model_path)
        hashes = self.compute_all_hashes(path)
        model_format = self._detect_format(path)

        return IntegrityRecord(
            model_name=model_name,
            model_version=model_version,
            model_format=model_format,
            hashes=hashes,
            metadata=metadata or {},
        )

    def verify(
        self,
        model_path: str | Path,
        record: IntegrityRecord,
    ) -> VerificationResult:
        """
        Verify model integrity against a stored record.

        Args:
            model_path: Path to the model file
            record: IntegrityRecord to verify against

        Returns:
            VerificationResult with status and details
        """
        path = Path(model_path)
        verified_hashes: list[str] = []
        failed_hashes: list[str] = []
        details: dict[str, Any] = {}

        if not path.exists():
            return VerificationResult(
                status=IntegrityStatus.ERROR,
                model_name=record.model_name,
                model_version=record.model_version,
                verified_hashes=[],
                failed_hashes=[],
                message=f"Model file not found: {model_path}",
            )

        for stored_hash in record.hashes:
            try:
                computed = self.compute_hash(path, stored_hash.algorithm)
                if hmac.compare_digest(computed.digest, stored_hash.digest):
                    verified_hashes.append(stored_hash.algorithm.value)
                    details[stored_hash.algorithm.value] = "verified"
                else:
                    failed_hashes.append(stored_hash.algorithm.value)
                    details[stored_hash.algorithm.value] = {
                        "expected": stored_hash.digest[:16] + "...",
                        "computed": computed.digest[:16] + "...",
                    }
            except Exception as e:
                failed_hashes.append(stored_hash.algorithm.value)
                details[stored_hash.algorithm.value] = f"error: {str(e)}"

        if failed_hashes:
            status = IntegrityStatus.TAMPERED
            message = f"Model integrity verification FAILED: {len(failed_hashes)} hash(es) do not match"
        elif verified_hashes:
            status = IntegrityStatus.VERIFIED
            message = f"Model integrity verified with {len(verified_hashes)} hash algorithm(s)"
        else:
            status = IntegrityStatus.UNKNOWN
            message = "No hashes to verify"

        return VerificationResult(
            status=status,
            model_name=record.model_name,
            model_version=record.model_version,
            verified_hashes=verified_hashes,
            failed_hashes=failed_hashes,
            message=message,
            details=details,
        )

    def sign_record(
        self,
        record: IntegrityRecord,
        secret_key: bytes,
        signer_id: str,
    ) -> IntegrityRecord:
        """
        Sign an integrity record with HMAC.

        Args:
            record: IntegrityRecord to sign
            secret_key: Secret key for HMAC
            signer_id: Identifier of the signer

        Returns:
            IntegrityRecord with signature
        """
        # Create canonical JSON representation
        record_data = record.to_dict()
        record_data.pop("signature", None)
        record_data.pop("signed_by", None)
        canonical = json.dumps(record_data, sort_keys=True)

        # Compute HMAC signature
        signature = hmac.new(
            secret_key,
            canonical.encode(),
            hashlib.sha256,
        ).hexdigest()

        # Return new record with signature
        return IntegrityRecord(
            model_name=record.model_name,
            model_version=record.model_version,
            model_format=record.model_format,
            hashes=record.hashes,
            metadata=record.metadata,
            created_at=record.created_at,
            signed_by=signer_id,
            signature=signature,
        )

    def verify_signature(
        self,
        record: IntegrityRecord,
        secret_key: bytes,
    ) -> bool:
        """
        Verify the HMAC signature on an integrity record.

        Args:
            record: IntegrityRecord with signature
            secret_key: Secret key for HMAC

        Returns:
            True if signature is valid
        """
        if not record.signature:
            return False

        # Recreate canonical JSON
        record_data = record.to_dict()
        stored_signature = record_data.pop("signature", None)
        record_data.pop("signed_by", None)
        canonical = json.dumps(record_data, sort_keys=True)

        # Compute expected signature
        expected = hmac.new(
            secret_key,
            canonical.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, stored_signature or "")

    def save_record(
        self,
        record: IntegrityRecord,
        output_path: str | Path,
    ) -> None:
        """
        Save an integrity record to a JSON file.

        Args:
            record: IntegrityRecord to save
            output_path: Path for output file
        """
        path = Path(output_path)
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        logger.info(f"Saved integrity record to {path}")

    def load_record(
        self,
        record_path: str | Path,
    ) -> IntegrityRecord:
        """
        Load an integrity record from a JSON file.

        Args:
            record_path: Path to the record file

        Returns:
            IntegrityRecord loaded from file
        """
        path = Path(record_path)
        with open(path) as f:
            data = json.load(f)
        return IntegrityRecord.from_dict(data)

    def _get_hasher(self, algorithm: HashAlgorithm) -> Any:
        """Get the appropriate hashlib hasher."""
        hashers = {
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA384: hashlib.sha384,
            HashAlgorithm.SHA512: hashlib.sha512,
            HashAlgorithm.SHA3_256: hashlib.sha3_256,
            HashAlgorithm.SHA3_512: hashlib.sha3_512,
            HashAlgorithm.BLAKE2B: hashlib.blake2b,
        }
        return hashers[algorithm]()

    def _update_hash_from_file(self, hasher: Any, f: BinaryIO) -> None:
        """Update hasher with file contents in chunks."""
        while chunk := f.read(self.chunk_size):
            hasher.update(chunk)

    def _detect_format(self, path: Path) -> ModelFormat:
        """Detect model format from file extension."""
        suffix = path.suffix.lower()
        format_map = {
            ".pt": ModelFormat.PYTORCH,
            ".pth": ModelFormat.PYTORCH,
            ".pb": ModelFormat.TENSORFLOW,
            ".onnx": ModelFormat.ONNX,
            ".safetensors": ModelFormat.SAFETENSORS,
            ".h5": ModelFormat.KERAS,
            ".keras": ModelFormat.KERAS,
            ".pkl": ModelFormat.PICKLE,
            ".joblib": ModelFormat.JOBLIB,
        }
        return format_map.get(suffix, ModelFormat.UNKNOWN)


# =============================================================================
# Chain of Custody Tracker
# =============================================================================


@dataclass
class CustodyEvent:
    """Single event in the chain of custody."""

    event_type: str  # "created", "transferred", "verified", "deployed"
    timestamp: str
    actor: str
    location: str | None = None
    notes: str | None = None
    integrity_status: IntegrityStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "location": self.location,
            "notes": self.notes,
            "integrity_status": (
                self.integrity_status.value if self.integrity_status else None
            ),
        }


@dataclass
class ChainOfCustody:
    """Chain of custody record for a model."""

    model_name: str
    model_version: str
    integrity_record: IntegrityRecord
    events: list[CustodyEvent] = field(default_factory=list)

    def add_event(
        self,
        event_type: str,
        actor: str,
        location: str | None = None,
        notes: str | None = None,
        integrity_status: IntegrityStatus | None = None,
    ) -> None:
        """Add a custody event."""
        event = CustodyEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            location=location,
            notes=notes,
            integrity_status=integrity_status,
        )
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "integrity_record": self.integrity_record.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def verify_model(
    model_path: str | Path,
    record_path: str | Path,
) -> VerificationResult:
    """
    Convenience function to verify a model against a stored record.

    Args:
        model_path: Path to the model file
        record_path: Path to the integrity record JSON

    Returns:
        VerificationResult with status
    """
    verifier = ModelIntegrityVerifier()
    record = verifier.load_record(record_path)
    return verifier.verify(model_path, record)


def generate_model_record(
    model_path: str | Path,
    model_name: str,
    model_version: str,
    output_path: str | Path | None = None,
) -> IntegrityRecord:
    """
    Convenience function to generate an integrity record.

    Args:
        model_path: Path to the model file
        model_name: Name of the model
        model_version: Version string
        output_path: Optional path to save the record

    Returns:
        IntegrityRecord
    """
    verifier = ModelIntegrityVerifier()
    record = verifier.generate_record(
        model_path=model_path,
        model_name=model_name,
        model_version=model_version,
    )

    if output_path:
        verifier.save_record(record, output_path)

    return record
