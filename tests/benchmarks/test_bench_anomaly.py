"""Benchmark tests for anomaly detection module.

Run with:
    pytest tests/benchmarks/test_bench_anomaly.py --benchmark-only
    pytest tests/benchmarks/test_bench_anomaly.py --benchmark-compare
"""

import numpy as np
import pytest

from medtech_ai_security.anomaly import (
    AnomalyDetector,
    DICOMPacket,
    TrafficGenerator,
)
from medtech_ai_security.anomaly.traffic_generator import DICOMCommand


@pytest.fixture
def sample_traffic_data():
    """Generate sample network traffic data for benchmarking."""
    np.random.seed(42)
    # Simulated traffic features: 1000 packets, 16 features (AnomalyDetector default)
    data = np.random.rand(1000, 16).astype(np.float32)
    return data


@pytest.fixture
def large_traffic_data():
    """Generate large traffic dataset for stress testing."""
    np.random.seed(42)
    # 10000 packets for performance testing
    data = np.random.rand(10000, 16).astype(np.float32)
    return data


@pytest.fixture
def trained_detector(sample_traffic_data):
    """Create and train an anomaly detector."""
    detector = AnomalyDetector(latent_dim=4, threshold_percentile=95.0)
    detector.fit(sample_traffic_data, epochs=20, verbose=False)
    return detector


class TestAnomalyDetectionBenchmarks:
    """Benchmark suite for anomaly detection operations."""

    def test_bench_detector_fit(self, benchmark, sample_traffic_data):
        """Benchmark detector fitting/training."""

        def run_fit():
            detector = AnomalyDetector(latent_dim=4)
            detector.fit(sample_traffic_data, epochs=20, verbose=False)
            return detector

        result = benchmark(run_fit)
        assert result is not None
        assert result.is_fitted

    def test_bench_detect_anomalies_small(self, benchmark, trained_detector, sample_traffic_data):
        """Benchmark anomaly detection on small dataset."""

        def run_detect():
            return trained_detector.detect(sample_traffic_data)

        result = benchmark(run_detect)
        assert result is not None
        assert len(result) == len(sample_traffic_data)

    def test_bench_detect_anomalies_large(self, benchmark, trained_detector, large_traffic_data):
        """Benchmark anomaly detection on large dataset."""
        # Train on large data first
        detector = AnomalyDetector(latent_dim=4)
        detector.fit(large_traffic_data, epochs=10, verbose=False)

        def run_detect():
            return detector.detect(large_traffic_data)

        result = benchmark(run_detect)
        assert result is not None
        assert len(result) == len(large_traffic_data)

    def test_bench_detect_batch(self, benchmark, trained_detector, sample_traffic_data):
        """Benchmark batch anomaly detection."""

        def run_batch():
            return trained_detector.detect_batch(sample_traffic_data)

        result = benchmark(run_batch)
        is_anomaly, scores, errors = result
        assert len(is_anomaly) == len(sample_traffic_data)

    def test_bench_get_top_anomalies(self, benchmark, trained_detector, sample_traffic_data):
        """Benchmark getting top anomalies."""

        def run_top():
            return trained_detector.get_top_anomalies(sample_traffic_data, top_k=50)

        result = benchmark(run_top)
        assert len(result) <= 50


class TestProtocolBenchmarks:
    """Benchmark suite for protocol-specific operations."""

    @pytest.fixture
    def dicom_packets(self):
        """Generate simulated DICOM packets."""
        np.random.seed(42)
        packets = []
        for i in range(500):
            packet = DICOMPacket(
                timestamp=float(i),
                source_ip=f"192.168.1.{i % 256}",
                dest_ip="192.168.1.100",
                source_port=np.random.randint(49152, 65535),
                dest_port=104,  # DICOM port
                calling_ae="WORKSTATION_1",
                called_ae="PACS_SERVER",
                command=DICOMCommand.C_STORE,
                message_id=i,
                affected_sop_class="1.2.840.10008.5.1.4.1.1.2",
                dataset_size=np.random.randint(100, 10000),
                transfer_syntax="1.2.840.10008.1.2",
            )
            packets.append(packet)
        return packets

    @pytest.fixture
    def hl7_traffic(self):
        """Generate simulated HL7 traffic data (16 features for detector)."""
        np.random.seed(42)
        return np.random.rand(500, 16).astype(np.float32)

    def test_bench_dicom_packet_creation(self, benchmark):
        """Benchmark DICOM packet creation."""

        def create_packets():
            packets = []
            for i in range(100):
                packet = DICOMPacket(
                    timestamp=float(i),
                    source_ip=f"192.168.1.{i % 256}",
                    dest_ip="192.168.1.100",
                    source_port=49152 + i,
                    dest_port=104,
                    calling_ae="WORKSTATION_1",
                    called_ae="PACS_SERVER",
                    command=DICOMCommand.C_STORE,
                    message_id=i,
                    affected_sop_class="1.2.840.10008.5.1.4.1.1.2",
                    dataset_size=1000,
                    transfer_syntax="1.2.840.10008.1.2",
                )
                packets.append(packet)
            return packets

        result = benchmark(create_packets)
        assert len(result) == 100

    def test_bench_dicom_feature_vector(self, benchmark, dicom_packets):
        """Benchmark DICOM feature vector extraction."""

        def extract_features():
            return [p.to_feature_vector() for p in dicom_packets[:100]]

        result = benchmark(extract_features)
        assert len(result) == 100
        assert result[0].shape == (16,)

    def test_bench_hl7_detection(self, benchmark, hl7_traffic):
        """Benchmark HL7-specific anomaly detection."""
        detector = AnomalyDetector(latent_dim=4)
        detector.fit(hl7_traffic, epochs=20, verbose=False)

        def run_detect():
            return detector.detect(hl7_traffic)

        result = benchmark(run_detect)
        assert result is not None
        assert len(result) == len(hl7_traffic)


class TestDetectorPersistenceBenchmarks:
    """Benchmark suite for detector save/load operations."""

    def test_bench_detector_save(self, benchmark, trained_detector, tmp_path):
        """Benchmark detector save operation."""
        save_path = tmp_path / "detector"

        def run_save():
            trained_detector.save(save_path)

        benchmark(run_save)
        assert (save_path / "detector_config.json").exists()

    def test_bench_detector_load(self, benchmark, trained_detector, tmp_path):
        """Benchmark detector load operation."""
        save_path = tmp_path / "detector"
        trained_detector.save(save_path)

        def run_load():
            return AnomalyDetector.load(save_path)

        result = benchmark(run_load)
        assert result is not None
        assert result.is_fitted


class TestExplainabilityBenchmarks:
    """Benchmark suite for anomaly explanation operations."""

    def test_bench_explain_anomaly(self, benchmark, trained_detector, sample_traffic_data):
        """Benchmark anomaly explanation generation."""
        # Get detection results with contributions
        results = trained_detector.detect(sample_traffic_data[:10], return_contributions=True)

        def run_explain():
            return trained_detector.explain_anomaly(results[0])

        result = benchmark(run_explain)
        assert result is not None
        assert "is_anomaly" in result

    def test_bench_evaluate_detector(self, benchmark, trained_detector, sample_traffic_data):
        """Benchmark detector evaluation."""
        # Create synthetic labels (10% anomalies)
        labels = np.zeros(len(sample_traffic_data))
        labels[:100] = 1

        def run_evaluate():
            return trained_detector.evaluate(sample_traffic_data, labels)

        result = benchmark(run_evaluate)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result


class TestTrafficGeneratorBenchmarks:
    """Benchmark suite for traffic generation."""

    def test_bench_generate_normal_dicom(self, benchmark):
        """Benchmark normal DICOM traffic generation."""
        generator = TrafficGenerator(seed=42)

        def run_generate():
            return generator.generate_normal_dicom(n_samples=500)

        result = benchmark(run_generate)
        assert len(result) == 500

    def test_bench_generate_attack_dicom(self, benchmark):
        """Benchmark attack DICOM traffic generation."""
        generator = TrafficGenerator(seed=42)

        def run_generate():
            return generator.generate_attack_dicom(n_samples=100)

        result = benchmark(run_generate)
        assert len(result) == 100
        assert all(p.is_attack for p in result)

    def test_bench_generate_dataset(self, benchmark):
        """Benchmark complete dataset generation."""
        generator = TrafficGenerator(seed=42)

        def run_generate():
            return generator.generate_dataset(n_normal=500, n_attack=50, protocol="dicom")

        features, labels, packets = benchmark(run_generate)
        assert len(features) == 550
        assert len(labels) == 550
