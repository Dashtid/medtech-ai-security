"""
Federated Learning Server (Coordinator)

Orchestrates federated learning across multiple medical institutions,
aggregating model updates without accessing raw patient data.

Features:
    - Client registration and management
    - Round-based training coordination
    - Multiple aggregation strategies (FedAvg, FedProx, FedNova)
    - Model versioning and checkpointing
    - Audit logging for compliance

Usage:
    server = FederatedServer(
        model_architecture="anomaly_detector",
        min_clients=3,
        rounds=100
    )
    server.start()
"""

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from medtech_ai_security.federated.aggregator import (
    ClientUpdate,
    ModelAggregator,
    create_aggregator,
)
from medtech_ai_security.federated.privacy import SecureAggregation

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ServerConfig:
    """Configuration for federated server."""

    host: str = "0.0.0.0"  # nosec B104 - Intentional for container/multi-client deployment
    port: int = 50051
    min_clients: int = 2
    max_clients: int = 100
    rounds: int = 100
    round_timeout: int = 300  # seconds
    aggregation_strategy: str = "fedavg"
    checkpoint_interval: int = 10  # rounds
    checkpoint_dir: str = "checkpoints/federated"
    enable_secure_aggregation: bool = True


@dataclass
class ClientInfo:
    """Information about a registered client."""

    client_id: str
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rounds_participated: int = 0
    total_samples: int = 0
    average_loss: float = 0.0
    status: str = "registered"  # registered, active, training, dropped

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "client_id": self.client_id,
            "registered_at": self.registered_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "rounds_participated": self.rounds_participated,
            "total_samples": self.total_samples,
            "average_loss": self.average_loss,
            "status": self.status,
        }


@dataclass
class RoundInfo:
    """Information about a training round."""

    round_number: int
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    participating_clients: list[str] = field(default_factory=list)
    total_samples: int = 0
    aggregated_loss: float | None = None
    status: str = "pending"  # pending, in_progress, completed, failed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_number": self.round_number,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "participating_clients": self.participating_clients,
            "total_samples": self.total_samples,
            "aggregated_loss": self.aggregated_loss,
            "status": self.status,
        }


@dataclass
class ServerState:
    """Current state of the server."""

    status: str = "initialized"  # initialized, running, paused, stopped
    current_round: int = 0
    total_rounds: int = 0
    registered_clients: int = 0
    active_clients: int = 0
    model_version: int = 0
    last_checkpoint: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "registered_clients": self.registered_clients,
            "active_clients": self.active_clients,
            "model_version": self.model_version,
            "last_checkpoint": (self.last_checkpoint.isoformat() if self.last_checkpoint else None),
        }


# =============================================================================
# Federated Server
# =============================================================================


class FederatedServer:
    """
    Federated learning coordinator server.

    Manages the federated training process:
        1. Accepts client registrations
        2. Broadcasts global model
        3. Collects client updates
        4. Aggregates using specified strategy
        5. Updates global model
        6. Repeats for configured rounds

    Example:
        server = FederatedServer(
            model_architecture="anomaly_detector",
            min_clients=3,
            rounds=100
        )
        server.start()
    """

    def __init__(
        self,
        model_architecture: str = "anomaly_detector",
        min_clients: int = 2,
        rounds: int = 100,
        config: ServerConfig | None = None,
    ):
        """
        Initialize federated server.

        Args:
            model_architecture: Name of model architecture to train
            min_clients: Minimum clients before starting training
            rounds: Total training rounds
            config: Optional detailed configuration
        """
        self.model_architecture = model_architecture
        self.config = config or ServerConfig(
            min_clients=min_clients,
            rounds=rounds,
        )

        self.state = ServerState(total_rounds=self.config.rounds)
        self._clients: dict[str, ClientInfo] = {}
        self._round_updates: dict[str, ClientUpdate] = {}
        self._round_history: list[RoundInfo] = []

        # Global model weights
        self._global_weights: list[np.ndarray] | None = None

        # Aggregator
        self._aggregator: ModelAggregator = create_aggregator(
            strategy=self.config.aggregation_strategy,
            min_clients=self.config.min_clients,
        )

        # Secure aggregation (optional)
        self._secure_agg: SecureAggregation | None = None
        if self.config.enable_secure_aggregation:
            self._secure_agg = SecureAggregation(
                num_clients=self.config.max_clients,
                threshold=self.config.min_clients,
            )

        # Threading
        self._running = False
        self._server_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Callbacks
        self._callbacks: list[Callable[[str, dict[str, Any]], None]] = []

        # Audit log
        self._audit_log: list[dict[str, Any]] = []

        logger.info(
            f"Initialized FederatedServer: {model_architecture}, "
            f"min_clients={min_clients}, rounds={rounds}"
        )

    def add_callback(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Add callback for server events."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: str, data: dict[str, Any]) -> None:
        """Notify all callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _log_audit(self, action: str, details: dict[str, Any]) -> None:
        """Log audit event for compliance."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details,
        }
        self._audit_log.append(entry)
        logger.info(f"AUDIT: {action} - {json.dumps(details, default=str)}")

    # =========================================================================
    # Server Lifecycle
    # =========================================================================

    def start(self, blocking: bool = True) -> None:
        """
        Start the federated server.

        Args:
            blocking: If True, blocks until server stops
        """
        logger.info(f"Starting federated server on {self.config.host}:{self.config.port}")

        self._running = True
        self.state.status = "running"

        # Initialize global model
        self._initialize_model()

        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self._log_audit(
            "server_started",
            {
                "host": self.config.host,
                "port": self.config.port,
                "min_clients": self.config.min_clients,
                "rounds": self.config.rounds,
            },
        )

        self._notify_callbacks("server_started", self.state.to_dict())

        if blocking:
            self._server_loop()
        else:
            self._server_thread = threading.Thread(
                target=self._server_loop,
                daemon=True,
            )
            self._server_thread.start()

    def stop(self) -> None:
        """Stop the federated server gracefully."""
        logger.info("Stopping federated server")
        self._running = False
        self.state.status = "stopped"

        # Save final checkpoint
        self._save_checkpoint()

        self._log_audit(
            "server_stopped",
            {
                "rounds_completed": self.state.current_round,
                "final_model_version": self.state.model_version,
            },
        )

        self._notify_callbacks("server_stopped", self.state.to_dict())

    def pause(self) -> None:
        """Pause training (clients remain connected)."""
        self.state.status = "paused"
        self._notify_callbacks("server_paused", {})

    def resume(self) -> None:
        """Resume paused training."""
        if self.state.status == "paused":
            self.state.status = "running"
            self._notify_callbacks("server_resumed", {})

    # =========================================================================
    # Client Management
    # =========================================================================

    def register_client(self, client_id: str) -> dict[str, Any]:
        """
        Register a new client.

        Args:
            client_id: Unique client identifier

        Returns:
            Registration response with initial model
        """
        with self._lock:
            if client_id in self._clients:
                logger.warning(f"Client {client_id} already registered")
                self._clients[client_id].status = "active"
                self._clients[client_id].last_seen = datetime.now(timezone.utc)
            else:
                self._clients[client_id] = ClientInfo(client_id=client_id)
                self.state.registered_clients = len(self._clients)

                self._log_audit("client_registered", {"client_id": client_id})
                logger.info(f"Client registered: {client_id}")

            # Register with secure aggregation
            if self._secure_agg:
                self._secure_agg.register_client(client_id)

        self._notify_callbacks("client_registered", {"client_id": client_id})

        return {
            "status": "registered",
            "client_id": client_id,
            "current_round": self.state.current_round,
            "model_version": self.state.model_version,
        }

    def unregister_client(self, client_id: str) -> None:
        """Unregister a client."""
        with self._lock:
            if client_id in self._clients:
                self._clients[client_id].status = "dropped"
                self._log_audit("client_unregistered", {"client_id": client_id})
                logger.info(f"Client unregistered: {client_id}")

    def get_client_list(self) -> list[dict[str, Any]]:
        """Get list of all registered clients."""
        return [client.to_dict() for client in self._clients.values()]

    # =========================================================================
    # Model Distribution
    # =========================================================================

    def get_global_model(self) -> dict[str, Any]:
        """
        Get current global model for distribution.

        Returns:
            Model metadata and weights
        """
        if self._global_weights is None:
            return {"error": "Model not initialized"}

        return {
            "model_version": self.state.model_version,
            "round": self.state.current_round,
            "weights_shapes": [w.shape for w in self._global_weights],
            # In production, serialize weights properly
            "weights": [w.tolist() for w in self._global_weights],
        }

    def _initialize_model(self) -> None:
        """Initialize global model weights."""
        if self.model_architecture == "anomaly_detector":
            # Autoencoder architecture: 16 -> 64 -> 32 -> 16 -> 32 -> 64 -> 16
            self._global_weights = [
                np.random.randn(16, 64).astype(np.float32) * 0.1,
                np.random.randn(64, 32).astype(np.float32) * 0.1,
                np.random.randn(32, 16).astype(np.float32) * 0.1,
                np.random.randn(16, 32).astype(np.float32) * 0.1,
                np.random.randn(32, 64).astype(np.float32) * 0.1,
                np.random.randn(64, 16).astype(np.float32) * 0.1,
            ]
        else:
            # Generic MLP
            self._global_weights = [
                np.random.randn(100, 64).astype(np.float32) * 0.1,
                np.random.randn(64, 32).astype(np.float32) * 0.1,
                np.random.randn(32, 10).astype(np.float32) * 0.1,
            ]

        self.state.model_version = 1
        logger.info(f"Initialized model: {self.model_architecture}")

    # =========================================================================
    # Training Coordination
    # =========================================================================

    def _server_loop(self) -> None:
        """Main server loop."""
        while self._running and self.state.current_round < self.config.rounds:
            if self.state.status == "paused":
                time.sleep(1)
                continue

            # Wait for minimum clients
            if not self._wait_for_clients():
                continue

            # Start new round
            round_info = self._start_round()

            # Collect updates from clients
            self._collect_updates(round_info)

            # Aggregate updates
            self._aggregate_round(round_info)

            # Checkpoint periodically
            if self.state.current_round % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            # Brief pause between rounds
            time.sleep(1)

        logger.info("Training completed")
        self.state.status = "completed"

    def _wait_for_clients(self) -> bool:
        """Wait for minimum clients to be available."""
        active_clients = sum(
            1 for c in self._clients.values() if c.status in ("registered", "active")
        )
        self.state.active_clients = active_clients

        if active_clients < self.config.min_clients:
            logger.debug(f"Waiting for clients: {active_clients}/{self.config.min_clients}")
            time.sleep(2)
            return False

        return True

    def _start_round(self) -> RoundInfo:
        """Start a new training round."""
        self.state.current_round += 1
        round_info = RoundInfo(round_number=self.state.current_round)
        round_info.status = "in_progress"

        # Clear previous round updates
        self._round_updates.clear()

        self._log_audit("round_started", {"round": self.state.current_round})
        self._notify_callbacks("round_started", round_info.to_dict())

        logger.info(f"Started round {self.state.current_round}")
        return round_info

    def _collect_updates(self, round_info: RoundInfo) -> None:
        """
        Collect updates from clients for this round.

        In production, this would receive updates via gRPC.
        For simulation, we generate synthetic updates.
        """
        start_time = time.time()

        # Simulate collecting updates from active clients
        for client_id, client_info in self._clients.items():
            if client_info.status not in ("registered", "active"):
                continue

            # Simulate client training and update
            update = self._simulate_client_update(client_id)
            if update:
                self._round_updates[client_id] = update
                round_info.participating_clients.append(client_id)
                round_info.total_samples += update.num_samples

            # Check timeout
            if time.time() - start_time > self.config.round_timeout:
                logger.warning("Round timeout reached")
                break

        logger.info(
            f"Round {round_info.round_number}: " f"Collected {len(self._round_updates)} updates"
        )

    def _simulate_client_update(self, client_id: str) -> ClientUpdate | None:
        """Simulate receiving an update from a client."""
        # In production, this would deserialize actual client update
        if self._global_weights is None:
            return None

        # Simulate training with some noise
        np.random.seed(hash(client_id + str(self.state.current_round)) % 2**32)

        updated_weights = [
            w + np.random.randn(*w.shape).astype(np.float32) * 0.01 for w in self._global_weights
        ]

        return ClientUpdate(
            client_id=client_id,
            weights=updated_weights,
            num_samples=np.random.randint(100, 1000),
            local_epochs=1,
            metrics={"loss": np.random.uniform(0.1, 0.5)},
        )

    def submit_update(
        self,
        client_id: str,
        weights: list[np.ndarray],
        num_samples: int,
        local_epochs: int = 1,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Submit update from client (called by gRPC handler).

        Args:
            client_id: Client identifier
            weights: Updated model weights
            num_samples: Number of local samples used
            local_epochs: Number of local epochs trained
            metrics: Optional training metrics

        Returns:
            Acknowledgment response
        """
        with self._lock:
            if client_id not in self._clients:
                return {"error": "Client not registered"}

            update = ClientUpdate(
                client_id=client_id,
                weights=weights,
                num_samples=num_samples,
                local_epochs=local_epochs,
                metrics=metrics,
            )

            self._round_updates[client_id] = update
            self._clients[client_id].last_seen = datetime.now(timezone.utc)
            self._clients[client_id].rounds_participated += 1
            self._clients[client_id].total_samples += num_samples

            if metrics and "loss" in metrics:
                # Update rolling average loss
                old_avg = self._clients[client_id].average_loss
                rounds = self._clients[client_id].rounds_participated
                self._clients[client_id].average_loss = (
                    old_avg * (rounds - 1) + metrics["loss"]
                ) / rounds

        logger.debug(f"Received update from {client_id}: {num_samples} samples")

        return {
            "status": "accepted",
            "round": self.state.current_round,
        }

    def _aggregate_round(self, round_info: RoundInfo) -> None:
        """Aggregate client updates for this round."""
        if len(self._round_updates) < self.config.min_clients:
            logger.warning(
                f"Not enough updates: {len(self._round_updates)} < " f"{self.config.min_clients}"
            )
            round_info.status = "failed"
            return

        updates = list(self._round_updates.values())

        # Perform aggregation
        result = self._aggregator.aggregate(updates)

        if result is None:
            logger.error("Aggregation failed")
            round_info.status = "failed"
            return

        # Update global model
        self._global_weights = result.global_weights
        self.state.model_version += 1

        # Update round info
        round_info.completed_at = datetime.now(timezone.utc)
        round_info.aggregated_loss = result.round_metrics.get("loss")
        round_info.status = "completed"

        self._round_history.append(round_info)

        self._log_audit(
            "round_completed",
            {
                "round": round_info.round_number,
                "clients": len(updates),
                "total_samples": round_info.total_samples,
                "aggregated_loss": round_info.aggregated_loss,
            },
        )

        self._notify_callbacks("round_completed", round_info.to_dict())

        logger.info(
            f"Round {round_info.round_number} completed: "
            f"{len(updates)} clients, {round_info.total_samples} samples"
        )

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        if self._global_weights is None:
            return

        checkpoint_path = Path(self.config.checkpoint_dir) / (
            f"checkpoint_round_{self.state.current_round}.npz"
        )

        np.savez(
            checkpoint_path,
            *self._global_weights,
            model_version=self.state.model_version,
            round=self.state.current_round,
        )

        self.state.last_checkpoint = datetime.now(timezone.utc)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if successful
        """
        try:
            data = np.load(checkpoint_path)
            # Load weights (numbered arrays)
            weights = []
            i = 0
            while f"arr_{i}" in data:
                weights.append(data[f"arr_{i}"])
                i += 1

            self._global_weights = weights
            self.state.model_version = int(data.get("model_version", 1))
            self.state.current_round = int(data.get("round", 0))

            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive server status."""
        return {
            "state": self.state.to_dict(),
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "min_clients": self.config.min_clients,
                "max_clients": self.config.max_clients,
                "rounds": self.config.rounds,
                "aggregation_strategy": self.config.aggregation_strategy,
            },
            "clients": {
                "registered": len(self._clients),
                "active": sum(
                    1 for c in self._clients.values() if c.status in ("registered", "active")
                ),
            },
            "round_history_summary": {
                "total_rounds": len(self._round_history),
                "last_round": (self._round_history[-1].to_dict() if self._round_history else None),
            },
        }

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI for federated learning server."""
    import argparse

    parser = argparse.ArgumentParser(description="Federated learning coordinator server")
    parser.add_argument(
        "--host",
        "-H",
        default="0.0.0.0",  # nosec B104 - Intentional for federated multi-client deployment
        help="Server host",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=50051,
        help="Server port",
    )
    parser.add_argument(
        "--min-clients",
        "-m",
        type=int,
        default=2,
        help="Minimum clients before starting",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=100,
        help="Total training rounds",
    )
    parser.add_argument(
        "--aggregation",
        "-a",
        choices=["fedavg", "fedprox", "fednova", "robust"],
        default="fedavg",
        help="Aggregation strategy",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/federated",
        help="Checkpoint directory",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def event_handler(event: str, data: dict[str, Any]) -> None:
        """Handle server events."""
        print(f"[{event.upper()}] {json.dumps(data, default=str)}")

    config = ServerConfig(
        host=args.host,
        port=args.port,
        min_clients=args.min_clients,
        rounds=args.rounds,
        aggregation_strategy=args.aggregation,
        checkpoint_dir=args.checkpoint_dir,
    )

    server = FederatedServer(
        model_architecture="anomaly_detector",
        config=config,
    )
    server.add_callback(event_handler)

    print("Starting federated server")
    print(f"Address: {args.host}:{args.port}")
    print(f"Min clients: {args.min_clients}")
    print(f"Rounds: {args.rounds}")
    print(f"Aggregation: {args.aggregation}")
    print("-" * 50)

    try:
        # Simulate some clients for testing
        server.start(blocking=False)

        # Register some test clients
        for i in range(3):
            server.register_client(f"test_client_{i}")

        # Let it run
        while server.state.status == "running":
            time.sleep(5)
            status = server.get_status()
            print(f"Round {status['state']['current_round']}/{args.rounds}")

    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()

    print(f"\nFinal status: {json.dumps(server.get_status(), indent=2, default=str)}")


if __name__ == "__main__":
    main()
