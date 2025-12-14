"""
Tests for the Federated Learning module.

Tests cover:
- Privacy mechanisms (differential privacy, secure aggregation)
- Model aggregation strategies (FedAvg, FedProx, FedNova, Robust)
- Federated client lifecycle
- Federated server coordination
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from medtech_ai_security.federated.aggregator import (
    AggregationResult,
    ClientUpdate,
    FedAvgAggregator,
    FedNovaAggregator,
    FedProxAggregator,
    RobustAggregator,
    create_aggregator,
)
from medtech_ai_security.federated.client import (
    ClientConfig,
    ClientState,
    FederatedClient,
    TrainingMetrics,
)
from medtech_ai_security.federated.privacy import (
    CombinedPrivacy,
    DifferentialPrivacy,
    PrivacyBudget,
    PrivacyMetrics,
    SecureAggregation,
)
from medtech_ai_security.federated.server import (
    ClientInfo,
    FederatedServer,
    RoundInfo,
    ServerConfig,
    ServerState,
)

# =============================================================================
# Privacy Module Tests
# =============================================================================


class TestPrivacyBudget:
    """Test PrivacyBudget dataclass."""

    def test_initial_budget(self):
        """Test initial budget state."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert budget.consumed_epsilon == 0.0
        assert budget.consumed_delta == 0.0

    def test_remaining_epsilon(self):
        """Test remaining epsilon calculation."""
        budget = PrivacyBudget(epsilon=1.0)
        budget.consumed_epsilon = 0.3

        assert budget.remaining_epsilon == 0.7

    def test_is_exhausted(self):
        """Test budget exhaustion detection."""
        budget = PrivacyBudget(epsilon=1.0)

        assert not budget.is_exhausted

        budget.consumed_epsilon = 1.0
        assert budget.is_exhausted

    def test_consume_success(self):
        """Test successful budget consumption."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

        result = budget.consume(0.5, 1e-6)

        assert result is True
        assert budget.consumed_epsilon == 0.5
        assert budget.consumed_delta == 1e-6

    def test_consume_exceeds_budget(self):
        """Test consumption exceeding budget."""
        budget = PrivacyBudget(epsilon=1.0)
        budget.consumed_epsilon = 0.8

        result = budget.consume(0.5)  # Would exceed 1.0

        assert result is False
        assert budget.consumed_epsilon == 0.8  # Unchanged


class TestPrivacyMetrics:
    """Test PrivacyMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PrivacyMetrics(
            total_rounds=10,
            epsilon_per_round=[0.1] * 10,
            noise_multipliers=[1.0] * 10,
            clipping_rates=[0.2] * 10,
        )

        result = metrics.to_dict()

        assert result["total_rounds"] == 10
        assert result["cumulative_epsilon"] == 1.0
        assert result["average_noise_multiplier"] == 1.0
        assert result["average_clipping_rate"] == 0.2

    def test_empty_metrics(self):
        """Test empty metrics."""
        metrics = PrivacyMetrics()
        result = metrics.to_dict()

        assert result["total_rounds"] == 0
        assert result["cumulative_epsilon"] == 0
        assert result["average_noise_multiplier"] == 0.0


class TestDifferentialPrivacy:
    """Test DifferentialPrivacy engine."""

    def test_initialization(self):
        """Test DP initialization."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)

        assert dp.budget.epsilon == 1.0
        assert dp.budget.delta == 1e-5
        assert dp.max_grad_norm == 1.0
        assert dp.noise_multiplier > 0

    def test_noise_multiplier_calculation(self):
        """Test noise multiplier auto-calculation."""
        # Lower epsilon should result in higher noise
        dp_low = DifferentialPrivacy(epsilon=0.1)
        dp_high = DifferentialPrivacy(epsilon=10.0)

        assert dp_low.noise_multiplier > dp_high.noise_multiplier

    def test_clip_gradients(self):
        """Test gradient clipping."""
        dp = DifferentialPrivacy(max_grad_norm=1.0)

        # Create gradient with norm > 1.0
        large_grad = np.array([10.0, 10.0])  # norm ~14.14
        gradients = [large_grad]

        clipped, rate = dp.clip_gradients(gradients)

        # Should be clipped to norm 1.0
        assert np.linalg.norm(clipped[0]) <= 1.0 + 1e-6
        assert rate == 1.0  # 100% clipped

    def test_clip_gradients_no_clipping_needed(self):
        """Test gradient clipping when not needed."""
        dp = DifferentialPrivacy(max_grad_norm=10.0)

        small_grad = np.array([0.1, 0.1])
        gradients = [small_grad]

        clipped, rate = dp.clip_gradients(gradients)

        np.testing.assert_array_almost_equal(clipped[0], small_grad)
        assert rate == 0.0  # Nothing clipped

    def test_add_noise(self):
        """Test noise addition."""
        dp = DifferentialPrivacy(noise_multiplier=1.0, max_grad_norm=1.0)

        gradients = [np.zeros((10, 10))]
        noisy = dp.add_noise(gradients)

        # Noisy gradients should not be all zeros
        assert not np.allclose(noisy[0], 0)

    def test_apply(self):
        """Test full DP application."""
        dp = DifferentialPrivacy(epsilon=1.0, max_grad_norm=1.0)

        gradients = [np.random.randn(50, 50)]
        protected = dp.apply(gradients)

        assert len(protected) == 1
        assert protected[0].shape == (50, 50)

    def test_apply_exhausted_budget(self):
        """Test application with exhausted budget."""
        dp = DifferentialPrivacy(epsilon=0.01)
        dp.budget.consumed_epsilon = 0.01  # Exhaust budget

        gradients = [np.random.randn(10, 10)]
        protected = dp.apply(gradients)

        # Should return zeros when budget exhausted
        np.testing.assert_array_equal(protected[0], np.zeros((10, 10)))

    def test_get_metrics(self):
        """Test metrics retrieval."""
        dp = DifferentialPrivacy(epsilon=1.0)
        dp.apply([np.random.randn(10, 10)])

        metrics = dp.get_metrics()

        assert "total_rounds" in metrics
        assert "remaining_epsilon" in metrics
        assert metrics["total_rounds"] == 1


class TestSecureAggregation:
    """Test SecureAggregation engine."""

    def test_initialization(self):
        """Test secure aggregation initialization."""
        sa = SecureAggregation(num_clients=5)

        assert sa.num_clients == 5
        assert sa.threshold == 3  # Default: 2/3 of clients

    def test_custom_threshold(self):
        """Test custom threshold."""
        sa = SecureAggregation(num_clients=10, threshold=5)

        assert sa.threshold == 5

    def test_register_client(self):
        """Test client registration."""
        sa = SecureAggregation(num_clients=5)

        seed = sa.register_client("client_1")

        assert len(seed) == 32  # 256 bits
        assert "client_1" in sa._client_seeds

    def test_generate_mask(self):
        """Test mask generation."""
        sa = SecureAggregation(num_clients=3)
        sa.register_client("client_1")

        masks = sa.generate_mask("client_1", [(10, 10), (5, 5)])

        assert len(masks) == 2
        assert masks[0].shape == (10, 10)
        assert masks[1].shape == (5, 5)

    def test_generate_mask_unregistered(self):
        """Test mask generation for unregistered client."""
        sa = SecureAggregation(num_clients=3)

        with pytest.raises(ValueError, match="not registered"):
            sa.generate_mask("unknown", [(10,)])

    def test_mask_gradients(self):
        """Test gradient masking."""
        sa = SecureAggregation(num_clients=3)
        sa.register_client("client_1")

        gradients = [np.ones((5, 5))]
        masked = sa.mask_gradients("client_1", gradients)

        # Masked should be different from original
        assert not np.allclose(masked[0], gradients[0])

    def test_aggregate_below_threshold(self):
        """Test aggregation with too few clients."""
        sa = SecureAggregation(num_clients=5, threshold=3)

        updates = {
            "client_1": [np.ones((5,))],
            "client_2": [np.ones((5,))],
        }

        result = sa.aggregate(updates)

        assert result is None

    def test_aggregate_success(self):
        """Test successful aggregation."""
        sa = SecureAggregation(num_clients=3, threshold=2)

        # Register clients and create masked updates
        for i in range(3):
            sa.register_client(f"client_{i}")

        updates = {}
        for i in range(3):
            gradients = [np.ones((5,)) * (i + 1)]
            updates[f"client_{i}"] = sa.mask_gradients(f"client_{i}", gradients)

        result = sa.aggregate(updates)

        assert result is not None
        assert len(result) == 1
        # Average of 1, 2, 3 = 2
        np.testing.assert_array_almost_equal(result[0], np.ones(5) * 2)

    def test_get_metrics(self):
        """Test metrics retrieval."""
        sa = SecureAggregation(num_clients=5)
        sa.register_client("client_1")

        metrics = sa.get_metrics()

        assert metrics["num_clients"] == 5
        assert metrics["registered_clients"] == 1


class TestCombinedPrivacy:
    """Test CombinedPrivacy engine."""

    def test_initialization(self):
        """Test combined privacy initialization."""
        cp = CombinedPrivacy(epsilon=1.0, num_clients=3)

        assert cp.dp.budget.epsilon == 1.0
        assert cp.secure_agg.num_clients == 3

    def test_apply(self):
        """Test combined privacy application."""
        cp = CombinedPrivacy(epsilon=1.0, num_clients=3)

        gradients = [np.random.randn(10, 10)]
        protected = cp.apply(gradients)

        assert len(protected) == 1
        assert protected[0].shape == (10, 10)

    def test_get_metrics(self):
        """Test combined metrics."""
        cp = CombinedPrivacy(epsilon=1.0, num_clients=3)
        cp.apply([np.random.randn(10, 10)])

        metrics = cp.get_metrics()

        assert "differential_privacy" in metrics
        assert "secure_aggregation" in metrics


# =============================================================================
# Aggregator Module Tests
# =============================================================================


class TestClientUpdate:
    """Test ClientUpdate dataclass."""

    def test_creation(self):
        """Test client update creation."""
        weights = [np.random.randn(10, 10)]
        update = ClientUpdate(
            client_id="client_1",
            weights=weights,
            num_samples=100,
            local_epochs=2,
            metrics={"loss": 0.5},
        )

        assert update.client_id == "client_1"
        assert update.num_samples == 100
        assert update.local_epochs == 2
        assert update.metrics["loss"] == 0.5


class TestAggregationResult:
    """Test AggregationResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AggregationResult(
            global_weights=[np.zeros((5, 5))],
            num_clients=3,
            total_samples=1000,
            round_metrics={"loss": 0.3},
        )

        d = result.to_dict()

        assert d["num_clients"] == 3
        assert d["total_samples"] == 1000
        assert "global_weights" not in d  # Weights excluded


class TestFedAvgAggregator:
    """Test FedAvg aggregation."""

    @pytest.fixture
    def sample_updates(self):
        """Create sample client updates."""
        np.random.seed(42)
        return [
            ClientUpdate(
                client_id=f"client_{i}",
                weights=[np.ones((5, 5)) * (i + 1)],
                num_samples=(i + 1) * 100,
                metrics={"loss": 0.5 - i * 0.1},
            )
            for i in range(3)
        ]

    def test_initialization(self):
        """Test FedAvg initialization."""
        agg = FedAvgAggregator(min_clients=2)

        assert agg.min_clients == 2
        assert agg.round_number == 0

    def test_aggregate_below_minimum(self):
        """Test aggregation with too few clients."""
        agg = FedAvgAggregator(min_clients=5)
        updates = [
            ClientUpdate("c1", [np.ones((5,))], 100),
            ClientUpdate("c2", [np.ones((5,))], 100),
        ]

        result = agg.aggregate(updates)

        assert result is None

    def test_aggregate_success(self, sample_updates):
        """Test successful FedAvg aggregation."""
        agg = FedAvgAggregator(min_clients=2)

        result = agg.aggregate(sample_updates)

        assert result is not None
        assert result.num_clients == 3
        assert result.total_samples == 600  # 100 + 200 + 300

    def test_weighted_average(self):
        """Test that FedAvg performs weighted averaging."""
        agg = FedAvgAggregator(min_clients=2)

        # Client 1: weight=1, samples=100
        # Client 2: weight=2, samples=200
        updates = [
            ClientUpdate("c1", [np.ones((5,)) * 1], 100),
            ClientUpdate("c2", [np.ones((5,)) * 2], 200),
        ]

        result = agg.aggregate(updates)

        # Weighted avg: (1*100 + 2*200) / 300 = 500/300 = 1.667
        expected = np.ones(5) * (100 + 400) / 300
        np.testing.assert_array_almost_equal(result.global_weights[0], expected)

    def test_shape_validation(self):
        """Test shape validation between clients."""
        agg = FedAvgAggregator(min_clients=2)

        updates = [
            ClientUpdate("c1", [np.ones((5, 5))], 100),
            ClientUpdate("c2", [np.ones((10, 10))], 100),  # Different shape
        ]

        result = agg.aggregate(updates)

        assert result is None

    def test_round_tracking(self, sample_updates):
        """Test round number tracking."""
        agg = FedAvgAggregator(min_clients=2)

        agg.aggregate(sample_updates)
        assert agg.round_number == 1

        agg.aggregate(sample_updates)
        assert agg.round_number == 2


class TestFedProxAggregator:
    """Test FedProx aggregation."""

    def test_initialization(self):
        """Test FedProx initialization with mu."""
        agg = FedProxAggregator(min_clients=2, mu=0.1)

        assert agg.mu == 0.1

    def test_proximal_term(self):
        """Test proximal term computation."""
        agg = FedProxAggregator(mu=0.1)

        local = [np.ones((5,)) * 2]
        global_w = [np.ones((5,)) * 1]

        term = agg.get_proximal_term(local, global_w)

        # (0.1/2) * ||[1,1,1,1,1]||^2 = 0.05 * 5 = 0.25
        assert abs(term - 0.25) < 1e-6

    def test_proximal_gradient(self):
        """Test proximal gradient computation."""
        agg = FedProxAggregator(mu=0.1)

        local = [np.ones((5,)) * 2]
        global_w = [np.ones((5,)) * 1]

        grads = agg.compute_proximal_gradient(local, global_w)

        # mu * (local - global) = 0.1 * [1,1,1,1,1]
        expected = np.ones(5) * 0.1
        np.testing.assert_array_almost_equal(grads[0], expected)


class TestFedNovaAggregator:
    """Test FedNova aggregation."""

    def test_aggregate_with_varying_epochs(self):
        """Test FedNova handles varying local epochs."""
        agg = FedNovaAggregator(min_clients=2)

        updates = [
            ClientUpdate("c1", [np.ones((5,)) * 1], 100, local_epochs=1),
            ClientUpdate("c2", [np.ones((5,)) * 2], 100, local_epochs=5),
        ]

        result = agg.aggregate(updates)

        assert result is not None
        assert "tau_effective" in result.round_metrics


class TestRobustAggregator:
    """Test Robust (median) aggregation."""

    def test_median_aggregation(self):
        """Test coordinate-wise median."""
        agg = RobustAggregator(min_clients=2)

        updates = [
            ClientUpdate("c1", [np.array([1.0, 2.0, 3.0])], 100),
            ClientUpdate("c2", [np.array([2.0, 3.0, 4.0])], 100),
            ClientUpdate("c3", [np.array([100.0, 100.0, 100.0])], 100),  # Outlier
        ]

        result = agg.aggregate(updates)

        # Median should be resistant to outlier
        expected = np.array([2.0, 3.0, 4.0])  # Median values
        np.testing.assert_array_almost_equal(result.global_weights[0], expected)


class TestCreateAggregator:
    """Test aggregator factory function."""

    def test_create_fedavg(self):
        """Test FedAvg creation."""
        agg = create_aggregator("fedavg", min_clients=3)

        assert isinstance(agg, FedAvgAggregator)
        assert agg.min_clients == 3

    def test_create_fedprox(self):
        """Test FedProx creation with mu."""
        agg = create_aggregator("fedprox", min_clients=2, mu=0.05)

        assert isinstance(agg, FedProxAggregator)
        assert agg.mu == 0.05

    def test_create_fednova(self):
        """Test FedNova creation."""
        agg = create_aggregator("fednova")

        assert isinstance(agg, FedNovaAggregator)

    def test_create_robust(self):
        """Test Robust creation."""
        agg = create_aggregator("robust")

        assert isinstance(agg, RobustAggregator)

    def test_unknown_strategy(self):
        """Test unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_aggregator("unknown_strategy")


# =============================================================================
# Client Module Tests
# =============================================================================


class TestClientConfig:
    """Test ClientConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClientConfig(
            server_address="localhost:50051",
            client_id="test",
            local_data_path="data.csv",
        )

        assert config.batch_size == 32
        assert config.local_epochs == 1
        assert config.learning_rate == 0.01
        assert config.privacy_epsilon == 1.0


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.9,
            samples_processed=1000,
            training_time=10.5,
            epochs_completed=3,
        )

        d = metrics.to_dict()

        assert d["loss"] == 0.5
        assert d["accuracy"] == 0.9
        assert d["samples_processed"] == 1000


class TestClientState:
    """Test ClientState dataclass."""

    def test_initial_state(self):
        """Test initial client state."""
        state = ClientState()

        assert state.status == "initialized"
        assert state.current_round == 0
        assert state.error_message is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = ClientState(status="training", current_round=5)

        d = state.to_dict()

        assert d["status"] == "training"
        assert d["current_round"] == 5


class TestFederatedClient:
    """Test FederatedClient class."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client."""
        return FederatedClient(
            server_address="localhost:50051",
            client_id="test_client",
            local_data_path=str(tmp_path / "data.csv"),
        )

    def test_initialization(self, client):
        """Test client initialization."""
        assert client.config.client_id == "test_client"
        assert client.state.status == "initialized"

    def test_add_callback(self, client):
        """Test callback registration."""
        callback = MagicMock()
        client.add_callback(callback)

        assert callback in client._callbacks

    def test_get_state(self, client):
        """Test state retrieval."""
        state = client.get_state()

        assert "status" in state
        assert "config" in state
        assert "privacy_metrics" in state
        assert state["config"]["client_id"] == "test_client"

    def test_generate_synthetic_data(self, client):
        """Test synthetic data generation."""
        client._generate_synthetic_data()

        assert client._local_data is not None
        assert client._local_data.shape[1] == 16  # Feature dimension

    def test_train_local(self, client):
        """Test local training."""
        # Setup with small weight initialization to avoid overflow
        client._generate_synthetic_data()
        client._global_weights = [
            np.random.randn(16, 64).astype(np.float32) * 0.01,
            np.random.randn(64, 32).astype(np.float32) * 0.01,
            np.random.randn(32, 16).astype(np.float32) * 0.01,
            np.random.randn(16, 32).astype(np.float32) * 0.01,
            np.random.randn(32, 64).astype(np.float32) * 0.01,
            np.random.randn(64, 16).astype(np.float32) * 0.01,
        ]

        metrics = client._train_local()

        assert not np.isnan(metrics.loss)
        assert metrics.loss >= 0
        assert metrics.samples_processed > 0
        assert metrics.epochs_completed == client.config.local_epochs

    def test_leave_federation(self, client):
        """Test leaving federation."""
        client._running = True
        client.leave_federation()

        assert not client._running
        assert client.state.status == "disconnected"


# =============================================================================
# Server Module Tests
# =============================================================================


class TestServerConfig:
    """Test ServerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 50051
        assert config.min_clients == 2
        assert config.rounds == 100


class TestClientInfo:
    """Test ClientInfo dataclass."""

    def test_creation(self):
        """Test client info creation."""
        info = ClientInfo(client_id="hospital_a")

        assert info.client_id == "hospital_a"
        assert info.rounds_participated == 0
        assert info.status == "registered"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = ClientInfo(client_id="test", rounds_participated=5)

        d = info.to_dict()

        assert d["client_id"] == "test"
        assert d["rounds_participated"] == 5


class TestRoundInfo:
    """Test RoundInfo dataclass."""

    def test_creation(self):
        """Test round info creation."""
        info = RoundInfo(round_number=1)

        assert info.round_number == 1
        assert info.status == "pending"
        assert info.participating_clients == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = RoundInfo(
            round_number=5,
            status="completed",
            total_samples=1000,
        )

        d = info.to_dict()

        assert d["round_number"] == 5
        assert d["status"] == "completed"


class TestServerState:
    """Test ServerState dataclass."""

    def test_initial_state(self):
        """Test initial server state."""
        state = ServerState()

        assert state.status == "initialized"
        assert state.current_round == 0
        assert state.model_version == 0


class TestFederatedServer:
    """Test FederatedServer class."""

    @pytest.fixture
    def server(self):
        """Create test server."""
        return FederatedServer(
            model_architecture="anomaly_detector",
            min_clients=2,
            rounds=10,
        )

    def test_initialization(self, server):
        """Test server initialization."""
        assert server.model_architecture == "anomaly_detector"
        assert server.config.min_clients == 2
        assert server.config.rounds == 10
        assert server.state.status == "initialized"

    def test_register_client(self, server):
        """Test client registration."""
        response = server.register_client("hospital_a")

        assert response["status"] == "registered"
        assert response["client_id"] == "hospital_a"
        assert "hospital_a" in server._clients

    def test_register_duplicate_client(self, server):
        """Test duplicate client registration."""
        server.register_client("hospital_a")
        response = server.register_client("hospital_a")

        # Should reactivate, not create duplicate
        assert response["status"] == "registered"
        assert len([c for c in server._clients if c == "hospital_a"]) == 1

    def test_unregister_client(self, server):
        """Test client unregistration."""
        server.register_client("hospital_a")
        server.unregister_client("hospital_a")

        assert server._clients["hospital_a"].status == "dropped"

    def test_get_client_list(self, server):
        """Test getting client list."""
        server.register_client("hospital_a")
        server.register_client("hospital_b")

        clients = server.get_client_list()

        assert len(clients) == 2
        assert any(c["client_id"] == "hospital_a" for c in clients)

    def test_initialize_model(self, server):
        """Test model initialization."""
        server._initialize_model()

        assert server._global_weights is not None
        assert server.state.model_version == 1

    def test_get_global_model(self, server):
        """Test getting global model."""
        server._initialize_model()
        model_info = server.get_global_model()

        assert model_info["model_version"] == 1
        assert "weights_shapes" in model_info

    def test_submit_update(self, server):
        """Test submitting client update."""
        server.register_client("hospital_a")
        server._initialize_model()

        response = server.submit_update(
            client_id="hospital_a",
            weights=[np.random.randn(16, 64)],
            num_samples=100,
            metrics={"loss": 0.5},
        )

        assert response["status"] == "accepted"
        assert "hospital_a" in server._round_updates

    def test_submit_update_unregistered(self, server):
        """Test update from unregistered client."""
        response = server.submit_update(
            client_id="unknown",
            weights=[np.zeros((5,))],
            num_samples=100,
        )

        assert "error" in response

    def test_get_status(self, server):
        """Test status retrieval."""
        server.register_client("hospital_a")
        server._initialize_model()

        status = server.get_status()

        assert "state" in status
        assert "config" in status
        assert "clients" in status
        assert status["clients"]["registered"] == 1

    def test_get_audit_log(self, server):
        """Test audit log retrieval."""
        server.register_client("hospital_a")

        log = server.get_audit_log()

        assert len(log) >= 1
        assert any(entry["action"] == "client_registered" for entry in log)

    def test_pause_resume(self, server):
        """Test pause and resume."""
        server.state.status = "running"

        server.pause()
        assert server.state.status == "paused"

        server.resume()
        assert server.state.status == "running"

    def test_add_callback(self, server):
        """Test callback registration."""
        callback = MagicMock()
        server.add_callback(callback)

        assert callback in server._callbacks


# =============================================================================
# Integration Tests
# =============================================================================


class TestFederatedIntegration:
    """Integration tests for federated learning components."""

    def test_full_aggregation_flow(self):
        """Test complete aggregation flow."""
        # Create server
        server = FederatedServer(min_clients=2, rounds=5)
        server._initialize_model()

        # Register clients
        server.register_client("client_1")
        server.register_client("client_2")
        server.register_client("client_3")

        # Simulate client updates
        for client_id in ["client_1", "client_2", "client_3"]:
            weights = [w + np.random.randn(*w.shape) * 0.1 for w in server._global_weights]
            server.submit_update(
                client_id=client_id,
                weights=weights,
                num_samples=np.random.randint(100, 500),
                metrics={"loss": np.random.uniform(0.1, 0.5)},
            )

        # Perform aggregation
        updates = list(server._round_updates.values())
        result = server._aggregator.aggregate(updates)

        assert result is not None
        assert result.num_clients == 3

    def test_privacy_preserved_aggregation(self):
        """Test aggregation with privacy preservation."""
        dp = DifferentialPrivacy(epsilon=1.0, max_grad_norm=1.0)
        aggregator = FedAvgAggregator(min_clients=2)

        # Create updates with DP applied
        updates = []
        for i in range(3):
            original_weights = [np.random.randn(10, 10)]
            protected_weights = dp.apply(original_weights)
            updates.append(
                ClientUpdate(
                    client_id=f"client_{i}",
                    weights=protected_weights,
                    num_samples=100,
                )
            )

        result = aggregator.aggregate(updates)

        assert result is not None
        # DP metrics should show rounds consumed
        assert dp.get_metrics()["total_rounds"] == 3

    def test_client_server_communication_mock(self):
        """Test mocked client-server communication."""
        server = FederatedServer(min_clients=1, rounds=3)
        server._initialize_model()
        server.register_client("test_client")

        # Get initial model
        model = server.get_global_model()
        assert model["model_version"] == 1

        # Simulate training and update
        weights = [
            np.array(w) + np.random.randn(*np.array(w).shape) * 0.01 for w in model["weights"]
        ]

        response = server.submit_update(
            client_id="test_client",
            weights=weights,
            num_samples=500,
            local_epochs=2,
            metrics={"loss": 0.3, "accuracy": 0.85},
        )

        assert response["status"] == "accepted"

        # Check client info updated
        client = server._clients["test_client"]
        assert client.rounds_participated == 1
        assert client.total_samples == 500


class TestFederatedServerAdvanced:
    """Advanced tests for federated server."""

    def test_server_with_secure_aggregation(self):
        """Test server with secure aggregation enabled."""
        config = ServerConfig(
            min_clients=2,
            rounds=5,
            enable_secure_aggregation=True,
        )
        server = FederatedServer(
            model_architecture="anomaly_detector",
            config=config,
        )

        assert server._secure_agg is not None
        server._initialize_model()
        assert server._global_weights is not None

    def test_server_with_generic_mlp_architecture(self):
        """Test server with non-anomaly_detector architecture."""
        server = FederatedServer(
            model_architecture="mlp",
            min_clients=2,
            rounds=5,
        )
        server._initialize_model()

        # Generic MLP has different weight shapes
        assert len(server._global_weights) == 3
        assert server._global_weights[0].shape == (100, 64)

    def test_get_global_model_not_initialized(self):
        """Test getting model before initialization."""
        server = FederatedServer(min_clients=2, rounds=5)

        model = server.get_global_model()
        assert "error" in model

    def test_callback_notification(self):
        """Test callback notifications."""
        server = FederatedServer(min_clients=2, rounds=5)
        events = []

        def callback(event, data):
            events.append(event)

        server.add_callback(callback)
        server.register_client("test_client")

        assert "client_registered" in events

    def test_callback_error_handling(self):
        """Test callback error handling doesn't crash server."""
        server = FederatedServer(min_clients=2, rounds=5)

        def bad_callback(event, data):
            raise ValueError("Callback error")

        server.add_callback(bad_callback)
        # Should not raise
        server.register_client("test_client")

    def test_resume_non_paused(self):
        """Test resume when not paused."""
        server = FederatedServer(min_clients=2, rounds=5)
        server.state.status = "running"

        server.resume()  # Should do nothing
        assert server.state.status == "running"

    def test_unregister_nonexistent_client(self):
        """Test unregistering client that doesn't exist."""
        server = FederatedServer(min_clients=2, rounds=5)
        # Should not raise
        server.unregister_client("nonexistent")

    def test_audit_log_entries(self):
        """Test audit log contains expected entries."""
        server = FederatedServer(min_clients=2, rounds=5)
        server.register_client("client_a")
        server.register_client("client_b")
        server.unregister_client("client_a")

        log = server.get_audit_log()

        assert len(log) >= 3
        actions = [entry["action"] for entry in log]
        assert actions.count("client_registered") == 2
        assert "client_unregistered" in actions

    def test_start_non_blocking(self):
        """Test non-blocking server start."""
        server = FederatedServer(min_clients=5, rounds=1)
        server.start(blocking=False)

        assert server._running
        assert server.state.status == "running"

        server.stop()
        assert not server._running

    def test_wait_for_clients_insufficient(self):
        """Test _wait_for_clients returns False when insufficient."""
        server = FederatedServer(min_clients=3, rounds=5)
        server.register_client("client_1")

        result = server._wait_for_clients()
        assert result is False

    def test_wait_for_clients_sufficient(self):
        """Test _wait_for_clients returns True when sufficient."""
        server = FederatedServer(min_clients=2, rounds=5)
        server.register_client("client_1")
        server.register_client("client_2")

        result = server._wait_for_clients()
        assert result is True

    def test_start_round(self):
        """Test starting a new round."""
        server = FederatedServer(min_clients=2, rounds=5)
        server._initialize_model()

        events = []
        server.add_callback(lambda e, d: events.append(e))

        round_info = server._start_round()

        assert round_info.round_number == 1
        assert round_info.status == "in_progress"
        assert server.state.current_round == 1
        assert "round_started" in events

    def test_server_state_to_dict(self):
        """Test ServerState to_dict conversion."""
        server = FederatedServer(min_clients=2, rounds=5)
        server.register_client("client_1")
        server._initialize_model()

        state_dict = server.state.to_dict()

        assert "status" in state_dict
        assert "current_round" in state_dict
        assert "registered_clients" in state_dict
        assert "model_version" in state_dict

    def test_client_info_update_on_submit(self):
        """Test client info is updated on update submission."""
        server = FederatedServer(min_clients=2, rounds=5)
        server.register_client("client_1")
        server._initialize_model()

        # Submit multiple updates
        for i in range(3):
            weights = [w.copy() for w in server._global_weights]
            server.submit_update(
                client_id="client_1",
                weights=weights,
                num_samples=100 + i * 50,
                metrics={"loss": 0.5 - i * 0.1},
            )

        client = server._clients["client_1"]
        assert client.rounds_participated == 3
        assert client.total_samples == 100 + 150 + 200

    def test_server_config_from_config_object(self):
        """Test server configuration from config object."""
        config = ServerConfig(
            min_clients=3,
            rounds=20,
            host="0.0.0.0",
            port=50052,
            aggregation_strategy="fedprox",
            checkpoint_interval=5,
        )
        server = FederatedServer(
            model_architecture="anomaly_detector",
            config=config,
        )

        assert server.config.min_clients == 3
        assert server.config.rounds == 20
        assert server.config.host == "0.0.0.0"
        assert server.config.port == 50052
        assert server.config.aggregation_strategy == "fedprox"
        assert server.config.checkpoint_interval == 5


class TestFederatedClientAdvanced:
    """Advanced tests for federated client."""

    def test_client_state_to_dict(self):
        """Test ClientState to_dict conversion."""
        client = FederatedClient(
            server_address="localhost:50051",
            client_id="test_client",
            local_data_path="data/test.csv",
        )

        state = client.get_state()

        assert "status" in state
        assert "current_round" in state
        assert "config" in state
        assert state["config"]["client_id"] == "test_client"

    def test_client_callback_registration(self):
        """Test client callback registration and notification."""
        client = FederatedClient(
            server_address="localhost:50051",
            client_id="test_client",
            local_data_path="data/test.csv",
        )

        events = []
        client.add_callback(lambda e, d: events.append(e))
        client.leave_federation()

        assert "left_federation" in events

    def test_training_metrics_to_dict(self):
        """Test TrainingMetrics to_dict conversion."""
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.85,
            samples_processed=1000,
            training_time=45.5,
            epochs_completed=5,
        )

        d = metrics.to_dict()

        assert d["loss"] == 0.5
        assert d["accuracy"] == 0.85
        assert d["samples_processed"] == 1000
        assert d["training_time"] == 45.5
        assert d["epochs_completed"] == 5


class TestServerRoundAggregation:
    """Tests for server round aggregation."""

    @pytest.fixture
    def server_with_clients(self):
        """Create server with registered clients."""
        config = ServerConfig(min_clients=2, rounds=10)
        server = FederatedServer(
            model_architecture="anomaly_detector",
            config=config,
        )
        server.register_client("client_1")
        server.register_client("client_2")
        server._initialize_model()
        return server

    def test_aggregate_round_insufficient_updates(self, server_with_clients):
        """Test aggregation fails with insufficient updates."""
        server = server_with_clients

        # Create round info
        from medtech_ai_security.federated.server import RoundInfo

        round_info = RoundInfo(
            round_number=1,
            started_at=datetime.now(timezone.utc),
            total_samples=0,
            participating_clients=[],
            status="in_progress",
        )

        # Clear updates to have fewer than min_clients
        server._round_updates.clear()

        # Aggregate should fail
        server._aggregate_round(round_info)

        assert round_info.status == "failed"

    def test_aggregate_round_success(self, server_with_clients):
        """Test successful round aggregation."""
        server = server_with_clients

        # Submit updates from both clients
        weights = [w.copy() for w in server._global_weights]
        server.submit_update("client_1", weights, num_samples=100, metrics={"loss": 0.5})
        server.submit_update("client_2", weights, num_samples=150, metrics={"loss": 0.4})

        from medtech_ai_security.federated.server import RoundInfo

        round_info = RoundInfo(
            round_number=1,
            started_at=datetime.now(timezone.utc),
            total_samples=250,
            participating_clients=["client_1", "client_2"],
            status="in_progress",
        )

        server._aggregate_round(round_info)

        assert round_info.status == "completed"
        assert round_info.completed_at is not None


class TestServerCheckpointing:
    """Tests for server checkpointing."""

    @pytest.fixture
    def server(self):
        """Create a server with initialized model."""
        config = ServerConfig(min_clients=2, rounds=10)
        server = FederatedServer(
            model_architecture="anomaly_detector",
            config=config,
        )
        server._initialize_model()
        return server

    def test_save_checkpoint(self, server, tmp_path):
        """Test saving a checkpoint."""
        server.config.checkpoint_dir = str(tmp_path)
        server.state.current_round = 5

        server._save_checkpoint()

        checkpoint_files = list(tmp_path.glob("checkpoint_*.npz"))
        assert len(checkpoint_files) == 1
        assert "round_5" in str(checkpoint_files[0])

    def test_load_checkpoint_success(self, server, tmp_path):
        """Test loading a checkpoint successfully."""
        # First save a checkpoint
        server.config.checkpoint_dir = str(tmp_path)
        server.state.current_round = 5
        server.state.model_version = 3
        server._save_checkpoint()

        # Create new server and load checkpoint
        new_server = FederatedServer(
            model_architecture="anomaly_detector",
            config=ServerConfig(min_clients=2, rounds=10),
        )
        new_server._initialize_model()

        checkpoint_file = list(tmp_path.glob("checkpoint_*.npz"))[0]
        result = new_server.load_checkpoint(str(checkpoint_file))

        assert result is True
        assert new_server.state.current_round == 5
        assert new_server.state.model_version == 3

    def test_load_checkpoint_failure(self, server, tmp_path):
        """Test loading checkpoint from non-existent file."""
        result = server.load_checkpoint(str(tmp_path / "nonexistent.npz"))

        assert result is False


class TestServerStatusMethods:
    """Tests for server status methods."""

    @pytest.fixture
    def server_with_activity(self):
        """Create server with some activity."""
        config = ServerConfig(min_clients=2, rounds=10)
        server = FederatedServer(
            model_architecture="anomaly_detector",
            config=config,
        )
        server.register_client("client_1")
        server.register_client("client_2")
        server._initialize_model()
        return server

    def test_get_status_comprehensive(self, server_with_activity):
        """Test get_status returns comprehensive information."""
        server = server_with_activity

        status = server.get_status()

        assert "state" in status
        assert "config" in status
        assert "clients" in status
        assert "round_history_summary" in status

        # Check config details
        assert status["config"]["min_clients"] == 2
        assert status["config"]["rounds"] == 10

        # Check client counts
        assert status["clients"]["registered"] == 2
        assert status["clients"]["active"] == 2

    def test_get_audit_log_limit(self, server_with_activity):
        """Test get_audit_log respects limit."""
        server = server_with_activity

        # Generate multiple audit entries
        for i in range(5):
            server.register_client(f"temp_client_{i}")

        log = server.get_audit_log(limit=3)

        assert len(log) <= 3

    def test_client_status_tracking(self, server_with_activity):
        """Test client status is tracked correctly."""
        server = server_with_activity

        # Check initial status
        status = server.get_status()
        assert status["clients"]["registered"] == 2
        assert status["clients"]["active"] == 2

        # Unregister a client - sets status to "dropped"
        server.unregister_client("client_1")

        # Client is now dropped, so active count should decrease
        status = server.get_status()
        assert status["clients"]["active"] == 1
        assert server._clients["client_1"].status == "dropped"


class TestRoundInfoDataclass:
    """Tests for RoundInfo dataclass."""

    def test_round_info_to_dict(self):
        """Test RoundInfo to_dict conversion."""
        from medtech_ai_security.federated.server import RoundInfo

        round_info = RoundInfo(
            round_number=5,
            started_at=datetime.now(timezone.utc),
            total_samples=1000,
            participating_clients=["client_1", "client_2"],
            status="completed",
            aggregated_loss=0.45,
        )

        d = round_info.to_dict()

        assert d["round_number"] == 5
        assert d["total_samples"] == 1000
        assert d["status"] == "completed"
        assert d["aggregated_loss"] == 0.45
        assert len(d["participating_clients"]) == 2

    def test_round_info_with_completion_time(self):
        """Test RoundInfo with completion time."""
        from medtech_ai_security.federated.server import RoundInfo

        started = datetime.now(timezone.utc)
        completed = datetime.now(timezone.utc)

        round_info = RoundInfo(
            round_number=1,
            started_at=started,
            completed_at=completed,
            total_samples=500,
            participating_clients=["client_1"],
            status="completed",
        )

        d = round_info.to_dict()
        assert d["completed_at"] is not None


class TestClientUpdateSubmission:
    """Tests for client update submission."""

    @pytest.fixture
    def server(self):
        """Create server with clients."""
        config = ServerConfig(min_clients=2, rounds=10)
        server = FederatedServer(
            model_architecture="anomaly_detector",
            config=config,
        )
        server.register_client("client_1")
        server._initialize_model()
        return server

    def test_submit_update_updates_client_stats(self, server):
        """Test that submitting updates updates client statistics."""
        weights = [w.copy() for w in server._global_weights]

        response = server.submit_update(
            client_id="client_1",
            weights=weights,
            num_samples=100,
            local_epochs=5,
            metrics={"loss": 0.5, "accuracy": 0.85},
        )

        assert response["status"] == "accepted"

        client = server._clients["client_1"]
        assert client.rounds_participated == 1
        assert client.total_samples == 100
        assert client.average_loss == 0.5

    def test_submit_multiple_updates_averages_loss(self, server):
        """Test multiple updates correctly average loss."""
        weights = [w.copy() for w in server._global_weights]

        # First update
        server.submit_update(
            client_id="client_1",
            weights=weights,
            num_samples=100,
            metrics={"loss": 1.0},
        )

        # Clear for second update
        server._round_updates.clear()

        # Second update
        server.submit_update(
            client_id="client_1",
            weights=weights,
            num_samples=100,
            metrics={"loss": 0.5},
        )

        client = server._clients["client_1"]
        # Average of 1.0 and 0.5 = 0.75
        assert abs(client.average_loss - 0.75) < 0.01
