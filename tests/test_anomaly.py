"""Unit tests for Phase 3: Anomaly Detection for Medical Device Traffic."""

import numpy as np
import pytest

from medtech_ai_security.anomaly.detector import (
    AnomalyDetector,
    Autoencoder,
    DetectionResult,
)
from medtech_ai_security.anomaly.traffic_generator import (
    AttackType,
    DICOMCommand,
    DICOMPacket,
    HL7MessageType,
    TrafficGenerator,
)


class TestDICOMCommand:
    """Test DICOM command enumeration."""

    def test_dicom_commands_exist(self):
        """Test DICOM commands are defined."""
        assert DICOMCommand.C_STORE.value == 0x0001
        assert DICOMCommand.C_FIND.value == 0x0020
        assert DICOMCommand.C_ECHO.value == 0x0030

    def test_all_commands_have_values(self):
        """Test all DICOM commands have unique values."""
        values = [cmd.value for cmd in DICOMCommand]
        assert len(values) == len(set(values))  # All unique


class TestHL7MessageType:
    """Test HL7 message type enumeration."""

    def test_hl7_message_types_exist(self):
        """Test HL7 message types are defined."""
        assert HL7MessageType.ADT_A01.value == "ADT^A01"
        assert HL7MessageType.ORM_O01.value == "ORM^O01"
        assert HL7MessageType.ACK.value == "ACK"


class TestAttackType:
    """Test attack type enumeration."""

    def test_attack_types_exist(self):
        """Test attack types are defined."""
        assert AttackType.DICOM_DATA_EXFILTRATION.value == "dicom_data_exfiltration"
        assert AttackType.HL7_MESSAGE_INJECTION.value == "hl7_message_injection"

    def test_attack_types_count(self):
        """Test expected number of attack types."""
        assert len(AttackType) >= 10  # At least 10 attack types


class TestDICOMPacket:
    """Test DICOM packet dataclass."""

    @pytest.fixture
    def sample_packet(self):
        """Create a sample DICOM packet."""
        return DICOMPacket(
            timestamp=1234567890.0,
            source_ip="192.168.1.100",
            dest_ip="192.168.1.200",
            source_port=11112,
            dest_port=104,
            calling_ae="WORKSTATION1",
            called_ae="PACS_SERVER",
            command=DICOMCommand.C_STORE,
            message_id=1234,
            affected_sop_class="1.2.840.10008.5.1.4.1.1.2",
            dataset_size=1024000,
            transfer_syntax="1.2.840.10008.1.2",
        )

    def test_packet_creation(self, sample_packet):
        """Test creating a DICOM packet."""
        assert sample_packet.source_ip == "192.168.1.100"
        assert sample_packet.command == DICOMCommand.C_STORE
        assert sample_packet.is_attack is False

    def test_packet_to_feature_vector(self, sample_packet):
        """Test converting packet to feature vector."""
        features = sample_packet.to_feature_vector()

        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) == 16  # 16-dimensional feature vector
        assert np.all(np.isfinite(features))  # No NaN or inf

    def test_packet_to_dict(self, sample_packet):
        """Test converting packet to dictionary."""
        result = sample_packet.to_dict()

        assert isinstance(result, dict)
        assert result["source_ip"] == "192.168.1.100"
        assert result["command"] == "C_STORE"

    def test_attack_packet(self):
        """Test attack packet has correct flags."""
        packet = DICOMPacket(
            timestamp=1234567890.0,
            source_ip="10.0.0.1",
            dest_ip="192.168.1.200",
            source_port=54321,
            dest_port=104,
            calling_ae="ATTACKER",
            called_ae="PACS",
            command=DICOMCommand.C_FIND,
            message_id=9999,
            affected_sop_class="1.2.3.4.5",
            dataset_size=999999999,
            transfer_syntax="1.2.3",
            is_attack=True,
            attack_type=AttackType.DICOM_DATA_EXFILTRATION,
        )

        assert packet.is_attack is True
        assert packet.attack_type == AttackType.DICOM_DATA_EXFILTRATION


class TestTrafficGenerator:
    """Test TrafficGenerator functionality."""

    @pytest.fixture
    def generator(self):
        """Create a traffic generator."""
        return TrafficGenerator(seed=42)

    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        # TrafficGenerator stores seed internally via rng
        assert generator.rng is not None

    def test_generate_normal_dicom_traffic(self, generator):
        """Test generating normal DICOM traffic."""
        packets = generator.generate_normal_dicom(n_samples=10)

        assert len(packets) == 10
        assert all(isinstance(p, DICOMPacket) for p in packets)
        assert all(p.is_attack is False for p in packets)

    def test_generate_normal_hl7_traffic(self, generator):
        """Test generating normal HL7 traffic."""
        messages = generator.generate_normal_hl7(n_samples=10)

        assert len(messages) == 10
        # HL7 messages should also be generated
        assert all(m.is_attack is False for m in messages)

    def test_generate_attack_traffic(self, generator):
        """Test generating attack traffic."""
        packets = generator.generate_attack_dicom(n_samples=10)

        assert len(packets) == 10
        assert all(p.is_attack is True for p in packets)
        assert all(p.attack_type is not None for p in packets)

    def test_generate_dataset(self, generator):
        """Test generating complete dataset with mixed traffic."""
        features, labels, packets = generator.generate_dataset(n_normal=50, n_attack=10)

        assert len(features) == len(labels)
        assert len(packets) == len(labels)
        # Check we have both normal and attack samples
        assert (labels == 0).sum() > 0
        assert (labels == 1).sum() > 0

    def test_traffic_to_features(self, generator):
        """Test converting traffic to feature matrix."""
        packets = generator.generate_normal_dicom(n_samples=20)
        features = np.array([p.to_feature_vector() for p in packets])

        assert isinstance(features, np.ndarray)
        assert features.shape == (20, 16)
        assert features.dtype == np.float32

    def test_save_dataset(self, generator, tmp_path):
        """Test saving dataset."""
        output_dir = tmp_path / "traffic"

        metadata = generator.save_dataset(
            path=output_dir,
            n_normal=10,
            n_attack=5,
            protocol="dicom",
        )

        assert output_dir.exists()
        assert (output_dir / "features.npy").exists()
        assert (output_dir / "labels.npy").exists()
        assert (output_dir / "packets.json").exists()
        assert metadata["n_samples"] > 0

    def test_reproducibility_with_seed(self):
        """Test generator produces reproducible results with same seed."""
        gen1 = TrafficGenerator(seed=123)
        gen2 = TrafficGenerator(seed=123)

        packets1 = gen1.generate_normal_dicom(n_samples=5)
        packets2 = gen2.generate_normal_dicom(n_samples=5)

        # Should produce same packets
        for p1, p2 in zip(packets1, packets2):
            assert p1.source_ip == p2.source_ip
            assert p1.command == p2.command


class TestAutoencoder:
    """Test Autoencoder implementation."""

    @pytest.fixture
    def autoencoder(self):
        """Create an autoencoder."""
        return Autoencoder(
            input_dim=16,
            latent_dim=4,
            hidden_dims=[12, 8],
            seed=42,
        )

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        return np.random.rand(100, 16).astype(np.float32)

    def test_autoencoder_initialization(self, autoencoder):
        """Test autoencoder initializes correctly."""
        assert autoencoder.input_dim == 16
        assert autoencoder.latent_dim == 4
        assert len(autoencoder.encoder_weights) == 3  # 16->12->8->4
        assert len(autoencoder.decoder_weights) == 3  # 4->8->12->16

    def test_encoder_weight_shapes(self, autoencoder):
        """Test encoder weights have correct shapes."""
        expected_shapes = [(16, 12), (12, 8), (8, 4)]

        for i, (weight, expected) in enumerate(zip(autoencoder.encoder_weights, expected_shapes)):
            assert weight.shape == expected, f"Encoder weight {i} has wrong shape"

    def test_decoder_weight_shapes(self, autoencoder):
        """Test decoder weights have correct shapes."""
        expected_shapes = [(4, 8), (8, 12), (12, 16)]

        for i, (weight, expected) in enumerate(zip(autoencoder.decoder_weights, expected_shapes)):
            assert weight.shape == expected, f"Decoder weight {i} has wrong shape"

    def test_encode(self, autoencoder, sample_data):
        """Test encoding data to latent space."""
        latent = autoencoder.encode(sample_data)

        assert latent.shape == (100, 4)
        assert np.all(np.isfinite(latent))

    def test_decode(self, autoencoder, sample_data):
        """Test decoding from latent space."""
        latent = autoencoder.encode(sample_data)
        reconstructed = autoencoder.decode(latent)

        assert reconstructed.shape == (100, 16)
        assert np.all(np.isfinite(reconstructed))

    def test_forward(self, autoencoder, sample_data):
        """Test forward pass (encode + decode)."""
        reconstructed, enc_acts, dec_acts = autoencoder.forward(sample_data)

        assert reconstructed.shape == sample_data.shape
        assert np.all(np.isfinite(reconstructed))
        assert len(enc_acts) > 0
        assert len(dec_acts) > 0

    def test_reconstruction_error(self, autoencoder, sample_data):
        """Test reconstruction error computation."""
        errors = autoencoder.reconstruction_error(sample_data)

        assert len(errors) == len(sample_data)
        assert all(e >= 0 for e in errors)
        assert np.all(np.isfinite(errors))

    def test_fit(self, autoencoder, sample_data):
        """Test training the autoencoder."""
        initial_error = np.mean(autoencoder.reconstruction_error(sample_data))

        history = autoencoder.fit(sample_data, epochs=10, batch_size=32)

        final_error = np.mean(autoencoder.reconstruction_error(sample_data))

        # Error should decrease after training
        assert final_error < initial_error
        assert "train_loss" in history
        assert len(history["train_loss"]) <= 10  # May stop early


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_result_creation(self):
        """Test creating a detection result."""
        result = DetectionResult(
            sample_index=0,
            reconstruction_error=0.05,
            anomaly_score=0.85,
            threshold=0.1,
            is_anomaly=False,
            confidence=0.95,
        )

        assert result.sample_index == 0
        assert result.is_anomaly is False
        assert result.confidence == 0.95

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = DetectionResult(
            sample_index=1,
            reconstruction_error=0.25,
            anomaly_score=0.75,
            threshold=0.1,
            is_anomaly=True,
            confidence=0.8,
        )

        result_dict = result.to_dict()

        assert result_dict["sample_index"] == 1
        assert result_dict["is_anomaly"] is True
        assert isinstance(result_dict["confidence"], float)


class TestAnomalyDetector:
    """Test AnomalyDetector functionality."""

    @pytest.fixture
    def detector(self):
        """Create an anomaly detector."""
        return AnomalyDetector(
            latent_dim=4,
            threshold_percentile=95,
            seed=42,
        )

    @pytest.fixture
    def normal_traffic(self):
        """Generate normal traffic features."""
        np.random.seed(42)
        return np.random.rand(200, 16).astype(np.float32) * 0.5 + 0.25

    @pytest.fixture
    def attack_traffic(self):
        """Generate attack traffic features (different distribution)."""
        np.random.seed(43)
        return np.random.rand(20, 16).astype(np.float32) * 0.8 + 0.6

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.is_fitted is False
        assert detector.threshold == 0.0

    def test_fit(self, detector, normal_traffic):
        """Test training the detector."""
        result = detector.fit(
            normal_traffic,
            epochs=20,
            batch_size=32,
        )

        assert result is detector  # Returns self
        assert detector.is_fitted is True
        assert detector.threshold is not None
        assert detector.threshold > 0
        assert detector.train_history is not None

    def test_detect_normal_traffic(self, detector, normal_traffic):
        """Test detection on normal traffic."""
        detector.fit(normal_traffic, epochs=20)

        results = detector.detect(normal_traffic[:10])

        assert len(results) == 10
        # Most normal traffic should not be flagged as anomaly
        anomaly_rate = sum(r.is_anomaly for r in results) / len(results)
        assert anomaly_rate < 0.5  # Less than 50% false positives

    def test_detect_attack_traffic(self, detector, normal_traffic, attack_traffic):
        """Test detection on attack traffic."""
        detector.fit(normal_traffic, epochs=50)

        results = detector.detect(attack_traffic)

        # Attack traffic should have higher anomaly scores
        assert len(results) == len(attack_traffic)
        # At least some attacks should be detected
        detection_rate = sum(r.is_anomaly for r in results) / len(results)
        assert detection_rate > 0.3  # At least 30% detection

    def test_detect_batch(self, detector, normal_traffic):
        """Test batch detection."""
        detector.fit(normal_traffic, epochs=20)

        is_anomaly, scores, errors = detector.detect_batch(normal_traffic[:10])

        assert len(is_anomaly) == 10
        assert len(scores) == 10
        assert len(errors) == 10
        assert all(isinstance(a, (bool, np.bool_)) for a in is_anomaly)

    def test_save_and_load_model(self, detector, normal_traffic, tmp_path):
        """Test saving and loading detector model."""
        detector.fit(normal_traffic, epochs=20)
        original_threshold = detector.threshold

        model_path = tmp_path / "detector_model"
        detector.save(model_path)

        # Load into new detector
        new_detector = AnomalyDetector.load(model_path)

        assert new_detector.is_fitted is True
        assert new_detector.threshold == pytest.approx(original_threshold, rel=0.01)

    def test_get_top_anomalies(self, detector, normal_traffic, attack_traffic):
        """Test getting top anomalies."""
        detector.fit(normal_traffic, epochs=20)

        # Combine normal and attack traffic
        test_data = np.vstack([normal_traffic[:20], attack_traffic])
        top_anomalies = detector.get_top_anomalies(test_data, top_k=5)

        assert len(top_anomalies) == 5
        # Top anomalies should have high scores
        assert all(r.anomaly_score >= 0 for r in top_anomalies)

    def test_evaluate(self, detector, normal_traffic, attack_traffic):
        """Test evaluation metrics."""
        detector.fit(normal_traffic, epochs=30)

        # Create test set with labels
        X_test = np.vstack([normal_traffic[:50], attack_traffic])
        y_test = np.array([0] * 50 + [1] * len(attack_traffic))

        metrics = detector.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1
