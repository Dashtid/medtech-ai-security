"""
Tests for the Model Integrity Verification module.

Tests cover:
- Hash computation for different algorithms
- Integrity record generation and verification
- Signature creation and verification
- Chain of custody tracking
"""

import json
import tempfile
from pathlib import Path

import pytest

from medtech_ai_security.ml.model_integrity import (
    ChainOfCustody,
    CustodyEvent,
    HashAlgorithm,
    IntegrityRecord,
    IntegrityStatus,
    ModelFormat,
    ModelHash,
    ModelIntegrityVerifier,
    VerificationResult,
    generate_model_record,
    verify_model,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_model_file():
    """Create a temporary file to simulate a model."""
    content = b"Mock model weights data " * 100
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt", mode="wb") as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def verifier():
    """Create a ModelIntegrityVerifier instance."""
    return ModelIntegrityVerifier()


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enumeration values."""

    def test_hash_algorithm_values(self):
        """Test HashAlgorithm enum values."""
        assert HashAlgorithm.SHA256.value == "sha256"
        assert HashAlgorithm.SHA512.value == "sha512"
        assert HashAlgorithm.SHA3_256.value == "sha3_256"
        assert HashAlgorithm.BLAKE2B.value == "blake2b"

    def test_integrity_status_values(self):
        """Test IntegrityStatus enum values."""
        assert IntegrityStatus.VERIFIED.value == "verified"
        assert IntegrityStatus.TAMPERED.value == "tampered"
        assert IntegrityStatus.UNKNOWN.value == "unknown"
        assert IntegrityStatus.ERROR.value == "error"

    def test_model_format_values(self):
        """Test ModelFormat enum values."""
        assert ModelFormat.PYTORCH.value == "pytorch"
        assert ModelFormat.ONNX.value == "onnx"
        assert ModelFormat.SAFETENSORS.value == "safetensors"


# =============================================================================
# ModelHash Tests
# =============================================================================


class TestModelHash:
    """Test ModelHash dataclass."""

    def test_creation(self):
        """Test creating a ModelHash."""
        hash_obj = ModelHash(
            algorithm=HashAlgorithm.SHA256,
            digest="abc123",
            file_path="/path/to/model.pt",
            file_size=1024,
        )
        assert hash_obj.algorithm == HashAlgorithm.SHA256
        assert hash_obj.digest == "abc123"
        assert hash_obj.file_size == 1024

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hash_obj = ModelHash(
            algorithm=HashAlgorithm.SHA256,
            digest="abc123",
            file_path="/path/to/model.pt",
            file_size=1024,
        )
        d = hash_obj.to_dict()
        assert d["algorithm"] == "sha256"
        assert d["digest"] == "abc123"
        assert d["file_size"] == 1024

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "algorithm": "sha256",
            "digest": "abc123",
            "file_path": "/path/to/model.pt",
            "file_size": 1024,
        }
        hash_obj = ModelHash.from_dict(d)
        assert hash_obj.algorithm == HashAlgorithm.SHA256
        assert hash_obj.digest == "abc123"


# =============================================================================
# IntegrityRecord Tests
# =============================================================================


class TestIntegrityRecord:
    """Test IntegrityRecord dataclass."""

    def test_creation(self):
        """Test creating an IntegrityRecord."""
        hash_obj = ModelHash(
            algorithm=HashAlgorithm.SHA256,
            digest="abc123",
            file_path="/path/to/model.pt",
            file_size=1024,
        )
        record = IntegrityRecord(
            model_name="test_model",
            model_version="1.0.0",
            model_format=ModelFormat.PYTORCH,
            hashes=[hash_obj],
        )
        assert record.model_name == "test_model"
        assert record.model_version == "1.0.0"
        assert len(record.hashes) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hash_obj = ModelHash(
            algorithm=HashAlgorithm.SHA256,
            digest="abc123",
            file_path="/path/to/model.pt",
            file_size=1024,
        )
        record = IntegrityRecord(
            model_name="test_model",
            model_version="1.0.0",
            model_format=ModelFormat.PYTORCH,
            hashes=[hash_obj],
        )
        d = record.to_dict()
        assert d["model_name"] == "test_model"
        assert d["model_format"] == "pytorch"
        assert len(d["hashes"]) == 1


# =============================================================================
# ModelIntegrityVerifier Tests
# =============================================================================


class TestModelIntegrityVerifier:
    """Test ModelIntegrityVerifier class."""

    def test_initialization(self, verifier):
        """Test verifier initialization."""
        assert len(verifier.algorithms) >= 1
        assert verifier.chunk_size > 0

    def test_initialization_custom_algorithms(self):
        """Test verifier with custom algorithms."""
        verifier = ModelIntegrityVerifier(
            algorithms=[HashAlgorithm.SHA512, HashAlgorithm.BLAKE2B]
        )
        assert HashAlgorithm.SHA512 in verifier.algorithms
        assert HashAlgorithm.BLAKE2B in verifier.algorithms

    def test_compute_hash_sha256(self, verifier, sample_model_file):
        """Test computing SHA256 hash."""
        hash_obj = verifier.compute_hash(sample_model_file, HashAlgorithm.SHA256)

        assert hash_obj.algorithm == HashAlgorithm.SHA256
        assert len(hash_obj.digest) == 64  # SHA256 produces 64 hex chars
        assert hash_obj.file_size > 0

    def test_compute_hash_sha512(self, verifier, sample_model_file):
        """Test computing SHA512 hash."""
        hash_obj = verifier.compute_hash(sample_model_file, HashAlgorithm.SHA512)

        assert hash_obj.algorithm == HashAlgorithm.SHA512
        assert len(hash_obj.digest) == 128  # SHA512 produces 128 hex chars

    def test_compute_hash_sha3(self, verifier, sample_model_file):
        """Test computing SHA3-256 hash."""
        hash_obj = verifier.compute_hash(sample_model_file, HashAlgorithm.SHA3_256)

        assert hash_obj.algorithm == HashAlgorithm.SHA3_256
        assert len(hash_obj.digest) == 64

    def test_compute_hash_deterministic(self, verifier, sample_model_file):
        """Test that hash computation is deterministic."""
        hash1 = verifier.compute_hash(sample_model_file, HashAlgorithm.SHA256)
        hash2 = verifier.compute_hash(sample_model_file, HashAlgorithm.SHA256)

        assert hash1.digest == hash2.digest

    def test_compute_hash_file_not_found(self, verifier):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            verifier.compute_hash("/nonexistent/path/model.pt")

    def test_compute_all_hashes(self, verifier, sample_model_file):
        """Test computing all configured hashes."""
        hashes = verifier.compute_all_hashes(sample_model_file)

        assert len(hashes) == len(verifier.algorithms)
        algorithms = [h.algorithm for h in hashes]
        for algo in verifier.algorithms:
            assert algo in algorithms

    def test_generate_record(self, verifier, sample_model_file):
        """Test generating an integrity record."""
        record = verifier.generate_record(
            model_path=sample_model_file,
            model_name="test_model",
            model_version="1.0.0",
        )

        assert record.model_name == "test_model"
        assert record.model_version == "1.0.0"
        assert record.model_format == ModelFormat.PYTORCH
        assert len(record.hashes) >= 1

    def test_generate_record_with_metadata(self, verifier, sample_model_file):
        """Test generating record with metadata."""
        metadata = {"trained_on": "dataset_v1", "epochs": 100}
        record = verifier.generate_record(
            model_path=sample_model_file,
            model_name="test_model",
            model_version="1.0.0",
            metadata=metadata,
        )

        assert record.metadata == metadata

    def test_verify_unchanged_file(self, verifier, sample_model_file):
        """Test verification of unchanged file."""
        record = verifier.generate_record(
            model_path=sample_model_file,
            model_name="test_model",
            model_version="1.0.0",
        )

        result = verifier.verify(sample_model_file, record)

        assert result.status == IntegrityStatus.VERIFIED
        assert len(result.verified_hashes) > 0
        assert len(result.failed_hashes) == 0

    def test_verify_modified_file(self, verifier, sample_model_file):
        """Test verification fails for modified file."""
        record = verifier.generate_record(
            model_path=sample_model_file,
            model_name="test_model",
            model_version="1.0.0",
        )

        # Modify the file
        with open(sample_model_file, "ab") as f:
            f.write(b"tampered data")

        result = verifier.verify(sample_model_file, record)

        assert result.status == IntegrityStatus.TAMPERED
        assert len(result.failed_hashes) > 0

    def test_verify_missing_file(self, verifier):
        """Test verification of missing file."""
        hash_obj = ModelHash(
            algorithm=HashAlgorithm.SHA256,
            digest="abc123",
            file_path="/nonexistent/model.pt",
            file_size=1024,
        )
        record = IntegrityRecord(
            model_name="test",
            model_version="1.0.0",
            model_format=ModelFormat.PYTORCH,
            hashes=[hash_obj],
        )

        result = verifier.verify("/nonexistent/model.pt", record)

        assert result.status == IntegrityStatus.ERROR

    def test_sign_record(self, verifier, sample_model_file):
        """Test signing an integrity record."""
        record = verifier.generate_record(
            model_path=sample_model_file,
            model_name="test_model",
            model_version="1.0.0",
        )

        secret_key = b"supersecretkey12345"
        signed = verifier.sign_record(record, secret_key, "test_signer")

        assert signed.signed_by == "test_signer"
        assert signed.signature is not None
        assert len(signed.signature) == 64  # HMAC-SHA256

    def test_verify_signature_valid(self, verifier, sample_model_file):
        """Test signature verification with valid key."""
        record = verifier.generate_record(
            model_path=sample_model_file,
            model_name="test_model",
            model_version="1.0.0",
        )

        secret_key = b"supersecretkey12345"
        signed = verifier.sign_record(record, secret_key, "test_signer")

        assert verifier.verify_signature(signed, secret_key)

    def test_verify_signature_invalid_key(self, verifier, sample_model_file):
        """Test signature verification with invalid key."""
        record = verifier.generate_record(
            model_path=sample_model_file,
            model_name="test_model",
            model_version="1.0.0",
        )

        secret_key = b"supersecretkey12345"
        wrong_key = b"wrongkey12345"
        signed = verifier.sign_record(record, secret_key, "test_signer")

        assert not verifier.verify_signature(signed, wrong_key)

    def test_save_and_load_record(self, verifier, sample_model_file):
        """Test saving and loading integrity records."""
        record = verifier.generate_record(
            model_path=sample_model_file,
            model_name="test_model",
            model_version="1.0.0",
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            record_path = f.name

        try:
            verifier.save_record(record, record_path)
            loaded = verifier.load_record(record_path)

            assert loaded.model_name == record.model_name
            assert loaded.model_version == record.model_version
            assert len(loaded.hashes) == len(record.hashes)
        finally:
            Path(record_path).unlink()


# =============================================================================
# Chain of Custody Tests
# =============================================================================


class TestChainOfCustody:
    """Test ChainOfCustody class."""

    def test_creation(self):
        """Test creating a chain of custody."""
        hash_obj = ModelHash(
            algorithm=HashAlgorithm.SHA256,
            digest="abc123",
            file_path="/path/to/model.pt",
            file_size=1024,
        )
        record = IntegrityRecord(
            model_name="test_model",
            model_version="1.0.0",
            model_format=ModelFormat.PYTORCH,
            hashes=[hash_obj],
        )
        custody = ChainOfCustody(
            model_name="test_model",
            model_version="1.0.0",
            integrity_record=record,
        )

        assert custody.model_name == "test_model"
        assert len(custody.events) == 0

    def test_add_event(self):
        """Test adding custody events."""
        hash_obj = ModelHash(
            algorithm=HashAlgorithm.SHA256,
            digest="abc123",
            file_path="/path/to/model.pt",
            file_size=1024,
        )
        record = IntegrityRecord(
            model_name="test_model",
            model_version="1.0.0",
            model_format=ModelFormat.PYTORCH,
            hashes=[hash_obj],
        )
        custody = ChainOfCustody(
            model_name="test_model",
            model_version="1.0.0",
            integrity_record=record,
        )

        custody.add_event(
            event_type="created",
            actor="developer@example.com",
            location="dev_server",
        )

        assert len(custody.events) == 1
        assert custody.events[0].event_type == "created"
        assert custody.events[0].actor == "developer@example.com"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_generate_model_record(self, sample_model_file):
        """Test generate_model_record function."""
        record = generate_model_record(
            model_path=sample_model_file,
            model_name="test",
            model_version="1.0.0",
        )

        assert record.model_name == "test"
        assert len(record.hashes) > 0

    def test_generate_model_record_with_output(self, sample_model_file):
        """Test generate_model_record with output path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            output_path = f.name

        try:
            record = generate_model_record(
                model_path=sample_model_file,
                model_name="test",
                model_version="1.0.0",
                output_path=output_path,
            )

            assert Path(output_path).exists()
            with open(output_path) as f:
                data = json.load(f)
            assert data["model_name"] == "test"
        finally:
            Path(output_path).unlink()

    def test_verify_model(self, sample_model_file):
        """Test verify_model function."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            record_path = f.name

        try:
            generate_model_record(
                model_path=sample_model_file,
                model_name="test",
                model_version="1.0.0",
                output_path=record_path,
            )

            result = verify_model(sample_model_file, record_path)

            assert result.status == IntegrityStatus.VERIFIED
        finally:
            Path(record_path).unlink()
