"""Benchmark tests for anomaly detection module.

Run with:
    pytest tests/benchmarks/test_bench_anomaly.py --benchmark-only
    pytest tests/benchmarks/test_bench_anomaly.py --benchmark-compare
"""

import numpy as np
import pytest


@pytest.fixture
def sample_traffic_data():
    """Generate sample network traffic data for benchmarking."""
    np.random.seed(42)
    # Simulated traffic features: 1000 packets, 20 features
    return np.random.randn(1000, 20).astype(np.float32)


@pytest.fixture
def large_traffic_data():
    """Generate large traffic dataset for stress testing."""
    np.random.seed(42)
    # 10000 packets for performance testing
    return np.random.randn(10000, 20).astype(np.float32)


class TestAnomalyDetectionBenchmarks:
    """Benchmark suite for anomaly detection operations."""

    def test_bench_preprocess_traffic(self, benchmark, sample_traffic_data):
        """Benchmark traffic data preprocessing."""
        from medtech_ai_security.anomaly.detector import preprocess_traffic

        result = benchmark(preprocess_traffic, sample_traffic_data)
        assert result is not None

    def test_bench_detect_anomalies_small(self, benchmark, sample_traffic_data):
        """Benchmark anomaly detection on small dataset."""
        from medtech_ai_security.anomaly.detector import AnomalyDetector

        detector = AnomalyDetector()
        # Warm up
        detector.fit(sample_traffic_data)

        result = benchmark(detector.predict, sample_traffic_data)
        assert len(result) == len(sample_traffic_data)

    def test_bench_detect_anomalies_large(self, benchmark, large_traffic_data):
        """Benchmark anomaly detection on large dataset."""
        from medtech_ai_security.anomaly.detector import AnomalyDetector

        detector = AnomalyDetector()
        detector.fit(large_traffic_data[:5000])  # Train on subset

        result = benchmark(detector.predict, large_traffic_data)
        assert len(result) == len(large_traffic_data)

    def test_bench_reconstruction_error(self, benchmark, sample_traffic_data):
        """Benchmark reconstruction error calculation."""
        from medtech_ai_security.anomaly.detector import AnomalyDetector

        detector = AnomalyDetector()
        detector.fit(sample_traffic_data)

        result = benchmark(detector.reconstruction_error, sample_traffic_data)
        assert len(result) == len(sample_traffic_data)

    def test_bench_feature_extraction(self, benchmark, sample_traffic_data):
        """Benchmark feature extraction from raw traffic."""
        from medtech_ai_security.anomaly.detector import extract_features

        # Simulate raw packet data
        raw_packets = [
            {"src_ip": f"192.168.1.{i}", "dst_port": 80 + (i % 100), "size": 64 + i}
            for i in range(1000)
        ]

        result = benchmark(extract_features, raw_packets)
        assert result.shape[0] == 1000


class TestProtocolBenchmarks:
    """Benchmark suite for protocol-specific operations."""

    @pytest.fixture
    def dicom_traffic(self):
        """Generate simulated DICOM traffic data."""
        np.random.seed(42)
        # DICOM-specific feature set
        return np.random.randn(500, 25).astype(np.float32)

    @pytest.fixture
    def hl7_traffic(self):
        """Generate simulated HL7 traffic data."""
        np.random.seed(42)
        return np.random.randn(500, 18).astype(np.float32)

    def test_bench_dicom_detection(self, benchmark, dicom_traffic):
        """Benchmark DICOM-specific anomaly detection."""
        from medtech_ai_security.anomaly.detector import DICOMAnomalyDetector

        detector = DICOMAnomalyDetector()
        detector.fit(dicom_traffic)

        result = benchmark(detector.predict, dicom_traffic)
        assert len(result) == len(dicom_traffic)

    def test_bench_hl7_detection(self, benchmark, hl7_traffic):
        """Benchmark HL7-specific anomaly detection."""
        from medtech_ai_security.anomaly.detector import HL7AnomalyDetector

        detector = HL7AnomalyDetector()
        detector.fit(hl7_traffic)

        result = benchmark(detector.predict, hl7_traffic)
        assert len(result) == len(hl7_traffic)


class TestAutoencoderBenchmarks:
    """Benchmark suite for autoencoder model operations."""

    @pytest.fixture
    def trained_autoencoder(self, sample_traffic_data):
        """Get a pre-trained autoencoder for benchmarking inference."""
        from medtech_ai_security.anomaly.detector import TrafficAutoencoder

        model = TrafficAutoencoder(input_dim=20, latent_dim=8)
        model.fit(sample_traffic_data, epochs=5, batch_size=32)
        return model

    def test_bench_autoencoder_forward(self, benchmark, trained_autoencoder, sample_traffic_data):
        """Benchmark autoencoder forward pass."""
        result = benchmark(trained_autoencoder.predict, sample_traffic_data)
        assert result.shape == sample_traffic_data.shape

    def test_bench_autoencoder_encode(self, benchmark, trained_autoencoder, sample_traffic_data):
        """Benchmark autoencoder encoding (compression)."""
        result = benchmark(trained_autoencoder.encode, sample_traffic_data)
        assert result.shape[1] == 8  # Latent dimension

    def test_bench_autoencoder_training_step(self, benchmark, sample_traffic_data):
        """Benchmark single training step."""
        from medtech_ai_security.anomaly.detector import TrafficAutoencoder

        model = TrafficAutoencoder(input_dim=20, latent_dim=8)

        def training_step():
            model.fit(sample_traffic_data, epochs=1, batch_size=64, verbose=0)

        benchmark(training_step)


# Skip benchmarks if dependencies not available
def pytest_configure(config):
    """Configure benchmark markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark test"
    )
