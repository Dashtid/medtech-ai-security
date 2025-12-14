"""
Federated Learning Client for Medical Institutions

Enables local model training on institution-specific data while
participating in collaborative federated learning.

HIPAA Compliance:
    - Raw patient data never leaves the institution
    - Only model updates (with DP noise) are shared
    - Local data remains under institution control

Usage:
    client = FederatedClient(
        server_address="coordinator.example.com:50051",
        client_id="hospital_a",
        local_data_path="data/traffic.csv"
    )
    client.join_federation()
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from medtech_ai_security.federated.privacy import DifferentialPrivacy, PrivacyEngine

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ClientConfig:
    """Configuration for federated client."""

    server_address: str
    client_id: str
    local_data_path: str | Path
    batch_size: int = 32
    local_epochs: int = 1
    learning_rate: float = 0.01
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    max_grad_norm: float = 1.0
    reconnect_interval: int = 30  # seconds


@dataclass
class TrainingMetrics:
    """Metrics from local training."""

    loss: float
    accuracy: float | None = None
    samples_processed: int = 0
    training_time: float = 0.0
    epochs_completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "samples_processed": self.samples_processed,
            "training_time": self.training_time,
            "epochs_completed": self.epochs_completed,
        }


@dataclass
class ClientState:
    """Current state of the client."""

    status: str = "initialized"  # initialized, connected, training, idle, error
    current_round: int = 0
    total_rounds_participated: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "current_round": self.current_round,
            "total_rounds_participated": self.total_rounds_participated,
            "last_update": self.last_update.isoformat(),
            "error_message": self.error_message,
        }


# =============================================================================
# Federated Client
# =============================================================================


class FederatedClient:
    """
    Federated learning client for medical institutions.

    Workflow:
        1. Connect to federated server
        2. Receive global model
        3. Train locally on institution data
        4. Apply differential privacy to gradients
        5. Send update to server
        6. Repeat for each round

    Example:
        client = FederatedClient(
            server_address="coordinator:50051",
            client_id="hospital_a",
            local_data_path="data/traffic.csv"
        )
        client.join_federation()
    """

    def __init__(
        self,
        server_address: str,
        client_id: str,
        local_data_path: str | Path,
        config: ClientConfig | None = None,
    ):
        """
        Initialize federated client.

        Args:
            server_address: Address of federated server (host:port)
            client_id: Unique identifier for this client
            local_data_path: Path to local training data
            config: Optional detailed configuration
        """
        self.config = config or ClientConfig(
            server_address=server_address,
            client_id=client_id,
            local_data_path=local_data_path,
        )

        self.state = ClientState()
        self._model: Any = None
        self._global_weights: list[np.ndarray] | None = None
        self._local_data: np.ndarray | None = None

        # Privacy engine
        self._privacy_engine: PrivacyEngine = DifferentialPrivacy(
            epsilon=self.config.privacy_epsilon,
            delta=self.config.privacy_delta,
            max_grad_norm=self.config.max_grad_norm,
        )

        # Threading
        self._running = False
        self._training_thread: threading.Thread | None = None
        self._callbacks: list[Callable[[str, dict[str, Any]], None]] = []

        logger.info(f"Initialized FederatedClient: {client_id}")

    def add_callback(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Add callback for state changes."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: str, data: dict[str, Any]) -> None:
        """Notify all callbacks of an event."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def join_federation(self, blocking: bool = True) -> None:
        """
        Join the federated learning federation.

        Args:
            blocking: If True, blocks until federation ends or error
        """
        logger.info(f"Joining federation at {self.config.server_address}")
        self._running = True
        self.state.status = "connecting"

        if blocking:
            self._federation_loop()
        else:
            self._training_thread = threading.Thread(
                target=self._federation_loop,
                daemon=True,
            )
            self._training_thread.start()

    def leave_federation(self) -> None:
        """Leave the federation gracefully."""
        logger.info("Leaving federation")
        self._running = False
        self.state.status = "disconnected"
        self._notify_callbacks("left_federation", {})

    def get_state(self) -> dict[str, Any]:
        """Get current client state."""
        return {
            **self.state.to_dict(),
            "config": {
                "server_address": self.config.server_address,
                "client_id": self.config.client_id,
                "local_epochs": self.config.local_epochs,
                "privacy_epsilon": self.config.privacy_epsilon,
            },
            "privacy_metrics": self._privacy_engine.get_metrics(),
        }

    # =========================================================================
    # Federation Protocol
    # =========================================================================

    def _federation_loop(self) -> None:
        """Main federation loop."""
        try:
            # Load local data
            self._load_local_data()

            # Connect to server
            if not self._connect_to_server():
                return

            self.state.status = "connected"
            self._notify_callbacks("connected", {"server": self.config.server_address})

            # Main training loop
            while self._running:
                # Wait for round start signal
                round_info = self._wait_for_round()
                if round_info is None:
                    continue

                self.state.current_round = round_info.get("round", 0)
                self.state.status = "training"

                # Receive global model
                self._global_weights = self._receive_global_model()
                if self._global_weights is None:
                    continue

                # Perform local training
                metrics = self._train_local()

                # Apply privacy and send update
                self._send_update(metrics)

                self.state.total_rounds_participated += 1
                self.state.status = "idle"
                self.state.last_update = datetime.now(timezone.utc)

                self._notify_callbacks(
                    "round_completed",
                    {
                        "round": self.state.current_round,
                        "metrics": metrics.to_dict(),
                    },
                )

        except Exception as e:
            logger.error(f"Federation error: {e}")
            self.state.status = "error"
            self.state.error_message = str(e)
            self._notify_callbacks("error", {"message": str(e)})

    def _connect_to_server(self) -> bool:
        """
        Connect to federated server.

        In production, this would establish a gRPC connection.
        """
        logger.info(f"Connecting to {self.config.server_address}")

        # Simulated connection (in production, use gRPC)
        # For now, we'll simulate with local file-based communication
        try:
            # Simulate handshake
            time.sleep(0.1)
            logger.info(f"Connected as {self.config.client_id}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _wait_for_round(self) -> dict[str, Any] | None:
        """Wait for next training round signal."""
        # In production, this would be a gRPC stream
        # Simulating with polling
        time.sleep(1)  # Poll interval

        if not self._running:
            return None

        # Simulate round signal
        return {"round": self.state.current_round + 1}

    def _receive_global_model(self) -> list[np.ndarray] | None:
        """
        Receive global model from server.

        Returns:
            Global model weights
        """
        logger.debug("Receiving global model")

        # In production, this would deserialize model from server
        # For simulation, create a random model
        if self._global_weights is None:
            # Initial model - create random weights matching anomaly detector
            self._global_weights = [
                np.random.randn(16, 64).astype(np.float32),  # Encoder layer 1
                np.random.randn(64, 32).astype(np.float32),  # Encoder layer 2
                np.random.randn(32, 16).astype(np.float32),  # Latent space
                np.random.randn(16, 32).astype(np.float32),  # Decoder layer 1
                np.random.randn(32, 64).astype(np.float32),  # Decoder layer 2
                np.random.randn(64, 16).astype(np.float32),  # Output layer
            ]

        return self._global_weights

    def _send_update(self, metrics: TrainingMetrics) -> None:
        """
        Send model update to server.

        Applies differential privacy before sending.
        """
        if self._global_weights is None:
            return

        # Apply differential privacy
        protected_weights = self._privacy_engine.apply(self._global_weights)

        # In production, this would serialize and send via gRPC
        logger.info(
            f"Sending update for round {self.state.current_round}: " f"loss={metrics.loss:.4f}"
        )

        # Update local weights for next round
        self._global_weights = protected_weights

    # =========================================================================
    # Local Training
    # =========================================================================

    def _load_local_data(self) -> None:
        """Load local training data."""
        data_path = Path(self.config.local_data_path)

        if data_path.exists():
            # Load actual data
            if data_path.suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(data_path)
                self._local_data = df.values.astype(np.float32)
            elif data_path.suffix == ".npy":
                self._local_data = np.load(data_path).astype(np.float32)
            else:
                logger.warning(f"Unknown data format: {data_path.suffix}")
                self._generate_synthetic_data()
        else:
            logger.warning(f"Data file not found: {data_path}")
            self._generate_synthetic_data()

        assert self._local_data is not None, "Local data not loaded"
        logger.info(f"Loaded {len(self._local_data)} samples")

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic training data for simulation."""
        np.random.seed(hash(self.config.client_id) % 2**32)

        # Generate synthetic network traffic features
        num_samples = np.random.randint(500, 2000)
        num_features = 16  # Match anomaly detector input

        # Normal traffic (80%)
        normal_samples = int(num_samples * 0.8)
        normal_data = np.random.randn(normal_samples, num_features) * 0.5

        # Anomalous traffic (20%)
        anomaly_samples = num_samples - normal_samples
        anomaly_data = np.random.randn(anomaly_samples, num_features) * 2.0 + 1.5

        self._local_data = np.vstack([normal_data, anomaly_data]).astype(np.float32)
        np.random.shuffle(self._local_data)

    def _train_local(self) -> TrainingMetrics:
        """
        Perform local training on institution data.

        Returns:
            Training metrics
        """
        if self._local_data is None or self._global_weights is None:
            return TrainingMetrics(loss=float("inf"))

        start_time = time.time()
        total_loss = 0.0
        samples_processed = 0

        # Simple gradient descent training (in production, use TF/PyTorch)
        for epoch in range(self.config.local_epochs):
            # Shuffle data
            indices = np.random.permutation(len(self._local_data))

            epoch_loss = 0.0
            for i in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[i : i + self.config.batch_size]
                batch = self._local_data[batch_indices]

                # Forward pass (simplified autoencoder)
                loss, gradients = self._compute_loss_and_gradients(batch)
                epoch_loss += loss * len(batch)
                samples_processed += len(batch)

                # Update weights
                for j, grad in enumerate(gradients):
                    self._global_weights[j] -= self.config.learning_rate * grad

            epoch_loss /= len(self._local_data)
            total_loss = epoch_loss

            logger.debug(f"Epoch {epoch + 1}/{self.config.local_epochs}: loss={epoch_loss:.4f}")

        training_time = time.time() - start_time

        return TrainingMetrics(
            loss=total_loss,
            samples_processed=samples_processed,
            training_time=training_time,
            epochs_completed=self.config.local_epochs,
        )

    def _compute_loss_and_gradients(self, batch: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """
        Compute loss and gradients for a batch.

        Simplified autoencoder computation for demonstration.
        In production, use TensorFlow/PyTorch.
        """
        assert self._global_weights is not None, "Global weights not initialized"

        # Simple forward pass through autoencoder layers
        # This is a simplified version - real implementation would use proper backprop

        # Encoder
        h1 = np.maximum(0, batch @ self._global_weights[0])  # ReLU
        h2 = np.maximum(0, h1 @ self._global_weights[1])
        latent = np.maximum(0, h2 @ self._global_weights[2])

        # Decoder
        h3 = np.maximum(0, latent @ self._global_weights[3])
        h4 = np.maximum(0, h3 @ self._global_weights[4])
        reconstruction = h4 @ self._global_weights[5]

        # Reconstruction loss (MSE)
        loss = np.mean((batch - reconstruction) ** 2)

        # Simplified gradients (not actual backprop, just noise for simulation)
        gradients = [np.random.randn(*w.shape) * 0.01 * loss for w in self._global_weights]

        return loss, gradients


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI for federated learning client."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Federated learning client for medical institutions"
    )
    parser.add_argument(
        "--server",
        "-s",
        default="localhost:50051",
        help="Federated server address",
    )
    parser.add_argument(
        "--client-id",
        "-c",
        required=True,
        help="Unique client identifier",
    )
    parser.add_argument(
        "--data-path",
        "-d",
        default="data/local_traffic.csv",
        help="Path to local training data",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=1,
        help="Local epochs per round",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Privacy budget (epsilon)",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=10,
        help="Number of rounds to participate",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def event_handler(event: str, data: dict[str, Any]) -> None:
        """Handle client events."""
        print(f"[{event.upper()}] {json.dumps(data, default=str)}")

    config = ClientConfig(
        server_address=args.server,
        client_id=args.client_id,
        local_data_path=args.data_path,
        local_epochs=args.epochs,
        privacy_epsilon=args.epsilon,
    )

    client = FederatedClient(
        server_address=args.server,
        client_id=args.client_id,
        local_data_path=args.data_path,
        config=config,
    )
    client.add_callback(event_handler)

    print(f"Starting federated client: {args.client_id}")
    print(f"Server: {args.server}")
    print(f"Local epochs: {args.epochs}")
    print(f"Privacy epsilon: {args.epsilon}")
    print("-" * 50)

    try:
        # Run for specified rounds
        client.join_federation(blocking=False)

        for _ in range(args.rounds):
            time.sleep(2)  # Wait between rounds
            state = client.get_state()
            print(f"Round {state['current_round']}: {state['status']}")

        client.leave_federation()

    except KeyboardInterrupt:
        print("\nInterrupted - leaving federation")
        client.leave_federation()

    print(f"\nFinal state: {json.dumps(client.get_state(), indent=2, default=str)}")


if __name__ == "__main__":
    main()
