"""
Model Aggregation Strategies for Federated Learning

Implements various aggregation algorithms for combining model updates
from multiple clients into a global model.

Supported Algorithms:
    - FedAvg: Weighted average by sample count
    - FedProx: FedAvg with proximal regularization
    - FedNova: Normalized averaging for heterogeneous settings
    - Median: Robust aggregation via coordinate-wise median

References:
    - McMahan et al., "Communication-Efficient Learning" (2017) - FedAvg
    - Li et al., "Federated Optimization in Heterogeneous Networks" (2020) - FedProx
    - Wang et al., "Tackling the Objective Inconsistency Problem" (2020) - FedNova
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ClientUpdate:
    """Model update from a single client."""

    client_id: str
    weights: list[np.ndarray]
    num_samples: int
    local_epochs: int = 1
    metrics: dict[str, float] | None = None


@dataclass
class AggregationResult:
    """Result of model aggregation."""

    global_weights: list[np.ndarray]
    num_clients: int
    total_samples: int
    round_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding weights)."""
        return {
            "num_clients": self.num_clients,
            "total_samples": self.total_samples,
            "round_metrics": self.round_metrics,
        }


# =============================================================================
# Base Aggregator
# =============================================================================


class ModelAggregator(ABC):
    """Base class for model aggregation strategies."""

    def __init__(self, min_clients: int = 2):
        """
        Initialize aggregator.

        Args:
            min_clients: Minimum clients required for aggregation
        """
        self.min_clients = min_clients
        self.round_number = 0

    @abstractmethod
    def aggregate(self, updates: list[ClientUpdate]) -> AggregationResult | None:
        """
        Aggregate client updates into global model.

        Args:
            updates: List of client updates

        Returns:
            Aggregation result or None if not enough clients
        """
        pass

    def _check_minimum_clients(self, updates: list[ClientUpdate]) -> bool:
        """Check if minimum client requirement is met."""
        if len(updates) < self.min_clients:
            logger.warning(f"Not enough clients: {len(updates)} < {self.min_clients}")
            return False
        return True

    def _validate_shapes(self, updates: list[ClientUpdate]) -> bool:
        """Validate that all updates have compatible shapes."""
        if not updates:
            return False

        reference_shapes = [w.shape for w in updates[0].weights]

        for update in updates[1:]:
            if len(update.weights) != len(reference_shapes):
                logger.error(f"Client {update.client_id} has different number of layers")
                return False

            for i, (w, ref_shape) in enumerate(zip(update.weights, reference_shapes)):
                if w.shape != ref_shape:
                    logger.error(
                        f"Client {update.client_id} layer {i} shape mismatch: "
                        f"{w.shape} vs {ref_shape}"
                    )
                    return False

        return True


# =============================================================================
# FedAvg Aggregator
# =============================================================================


class FedAvgAggregator(ModelAggregator):
    """
    Federated Averaging (FedAvg) aggregation.

    Computes weighted average of client weights, where weights are
    proportional to the number of local samples.

    Algorithm:
        w_global = sum(n_k * w_k) / sum(n_k)

    Where:
        n_k = number of samples at client k
        w_k = model weights from client k
    """

    def aggregate(self, updates: list[ClientUpdate]) -> AggregationResult | None:
        """
        Aggregate using FedAvg.

        Args:
            updates: Client updates with weights and sample counts

        Returns:
            Aggregated global model
        """
        if not self._check_minimum_clients(updates):
            return None

        if not self._validate_shapes(updates):
            return None

        # Calculate total samples for weighting
        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            logger.error("Total samples is zero")
            return None

        # Initialize aggregated weights with zeros
        num_layers = len(updates[0].weights)
        aggregated = [np.zeros_like(updates[0].weights[i]) for i in range(num_layers)]

        # Weighted sum
        for update in updates:
            weight_factor = update.num_samples / total_samples
            for i, layer_weights in enumerate(update.weights):
                aggregated[i] += weight_factor * layer_weights

        # Compute round metrics
        client_metrics: list[dict[str, float] | None] = [u.metrics for u in updates]
        round_metrics = self._aggregate_metrics(client_metrics)

        self.round_number += 1

        logger.info(
            f"FedAvg round {self.round_number}: "
            f"{len(updates)} clients, {total_samples} total samples"
        )

        return AggregationResult(
            global_weights=aggregated,
            num_clients=len(updates),
            total_samples=total_samples,
            round_metrics=round_metrics,
        )

    def _aggregate_metrics(self, client_metrics: list[dict[str, float] | None]) -> dict[str, float]:
        """Aggregate client metrics."""
        if not client_metrics:
            return {}

        # Filter out None values
        valid_metrics = [m for m in client_metrics if m is not None]
        if not valid_metrics:
            return {}

        # Get all metric keys
        all_keys: set[str] = set()
        for m in valid_metrics:
            all_keys.update(m.keys())

        # Average each metric
        aggregated = {}
        for key in all_keys:
            values = [m.get(key, 0.0) for m in valid_metrics if key in m]
            if values:
                aggregated[key] = float(np.mean(values))

        return aggregated


# =============================================================================
# FedProx Aggregator
# =============================================================================


class FedProxAggregator(FedAvgAggregator):
    """
    Federated Proximal (FedProx) aggregation.

    Extends FedAvg with a proximal term to handle system heterogeneity
    (clients with different compute capabilities or data distributions).

    The proximal term is applied during local training, not aggregation.
    This class tracks the proximal coefficient (mu) for client-side use.

    Local objective:
        h_k(w) = F_k(w) + (mu/2) * ||w - w_global||^2

    Where:
        F_k(w) = local loss function
        mu = proximal coefficient
        w_global = global model from previous round
    """

    def __init__(self, min_clients: int = 2, mu: float = 0.01):
        """
        Initialize FedProx aggregator.

        Args:
            min_clients: Minimum clients for aggregation
            mu: Proximal coefficient (higher = more regularization)
        """
        super().__init__(min_clients)
        self.mu = mu

    def get_proximal_term(
        self, local_weights: list[np.ndarray], global_weights: list[np.ndarray]
    ) -> float:
        """
        Compute proximal regularization term.

        Args:
            local_weights: Current local model weights
            global_weights: Global model weights

        Returns:
            Proximal term value: (mu/2) * ||w - w_global||^2
        """
        total = 0.0
        for local, global_w in zip(local_weights, global_weights):
            diff = local - global_w
            total += np.sum(diff**2)
        return (self.mu / 2) * total

    def compute_proximal_gradient(
        self, local_weights: list[np.ndarray], global_weights: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Compute gradient of proximal term.

        This should be added to the local gradient during training:
            gradient += mu * (w_local - w_global)

        Args:
            local_weights: Current local model weights
            global_weights: Global model weights

        Returns:
            Proximal gradient for each layer
        """
        return [
            self.mu * (local - global_w) for local, global_w in zip(local_weights, global_weights)
        ]


# =============================================================================
# FedNova Aggregator
# =============================================================================


class FedNovaAggregator(ModelAggregator):
    """
    Federated Normalized Averaging (FedNova) aggregation.

    Normalizes client updates by their local training progress to handle
    heterogeneous local epochs across clients.

    Key insight: FedAvg's weighted average is biased when clients perform
    different numbers of local steps. FedNova corrects this bias.

    Algorithm:
        1. Normalize each client's update by their local steps
        2. Aggregate normalized updates
        3. Scale by average local steps
    """

    def aggregate(self, updates: list[ClientUpdate]) -> AggregationResult | None:
        """
        Aggregate using FedNova.

        Args:
            updates: Client updates with weights, samples, and local epochs

        Returns:
            Aggregated global model with normalized averaging
        """
        if not self._check_minimum_clients(updates):
            return None

        if not self._validate_shapes(updates):
            return None

        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            return None

        # Calculate normalization factors
        # tau_i = local_epochs for client i
        # p_i = n_i / n (sample weight)
        tau_eff = 0.0  # Effective tau
        for update in updates:
            p_i = update.num_samples / total_samples
            tau_eff += p_i * update.local_epochs

        # Initialize aggregated weights
        num_layers = len(updates[0].weights)
        aggregated = [np.zeros_like(updates[0].weights[i]) for i in range(num_layers)]

        # Normalized weighted sum
        for update in updates:
            p_i = update.num_samples / total_samples
            # Normalize by local epochs, then weight by samples
            norm_factor = p_i / update.local_epochs

            for i, layer_weights in enumerate(update.weights):
                aggregated[i] += norm_factor * layer_weights

        # Scale by effective tau
        aggregated = [w * tau_eff for w in aggregated]

        self.round_number += 1

        logger.info(
            f"FedNova round {self.round_number}: " f"{len(updates)} clients, tau_eff={tau_eff:.2f}"
        )

        return AggregationResult(
            global_weights=aggregated,
            num_clients=len(updates),
            total_samples=total_samples,
            round_metrics={"tau_effective": tau_eff},
        )


# =============================================================================
# Robust Aggregator (Byzantine-tolerant)
# =============================================================================


class RobustAggregator(ModelAggregator):
    """
    Robust aggregation using coordinate-wise median.

    Provides Byzantine fault tolerance by using median instead of mean,
    which is robust to up to 50% malicious clients.

    Use cases:
        - Untrusted client environments
        - Potential adversarial participants
        - Data poisoning defense
    """

    def aggregate(self, updates: list[ClientUpdate]) -> AggregationResult | None:
        """
        Aggregate using coordinate-wise median.

        Args:
            updates: Client updates

        Returns:
            Median-aggregated global model
        """
        if not self._check_minimum_clients(updates):
            return None

        if not self._validate_shapes(updates):
            return None

        # Stack weights for each layer
        num_layers = len(updates[0].weights)
        aggregated = []

        for layer_idx in range(num_layers):
            # Stack all client weights for this layer
            stacked = np.stack([u.weights[layer_idx] for u in updates])
            # Coordinate-wise median
            median_weights = np.median(stacked, axis=0)
            aggregated.append(median_weights)

        total_samples = sum(u.num_samples for u in updates)
        self.round_number += 1

        logger.info(
            f"Robust aggregation round {self.round_number}: " f"{len(updates)} clients (median)"
        )

        return AggregationResult(
            global_weights=aggregated,
            num_clients=len(updates),
            total_samples=total_samples,
            round_metrics={},  # Median aggregation used
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_aggregator(
    strategy: str = "fedavg",
    min_clients: int = 2,
    **kwargs: Any,
) -> ModelAggregator:
    """
    Create an aggregator instance.

    Args:
        strategy: Aggregation strategy ("fedavg", "fedprox", "fednova", "robust")
        min_clients: Minimum clients for aggregation
        **kwargs: Strategy-specific arguments

    Returns:
        Configured aggregator instance
    """
    strategies: dict[str, type[ModelAggregator]] = {
        "fedavg": FedAvgAggregator,
        "fedprox": FedProxAggregator,
        "fednova": FedNovaAggregator,
        "robust": RobustAggregator,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies)}")

    aggregator_class = strategies[strategy]

    # Filter kwargs to only those accepted by the class
    if strategy == "fedprox":
        return FedProxAggregator(min_clients=min_clients, mu=kwargs.get("mu", 0.01))
    else:
        return aggregator_class(min_clients=min_clients)


# =============================================================================
# CLI for Testing
# =============================================================================


def main() -> None:
    """Test aggregation strategies."""
    import argparse

    parser = argparse.ArgumentParser(description="Test aggregation strategies")
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox", "fednova", "robust"],
        default="fedavg",
        help="Aggregation strategy",
    )
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")

    args = parser.parse_args()

    # Create test updates
    np.random.seed(42)

    updates = []
    for i in range(args.clients):
        weights = [
            np.random.randn(100, 50),
            np.random.randn(50, 10),
        ]
        updates.append(
            ClientUpdate(
                client_id=f"client_{i}",
                weights=weights,
                num_samples=np.random.randint(100, 1000),
                local_epochs=np.random.randint(1, 5),
                metrics={"loss": np.random.uniform(0.1, 1.0)},
            )
        )

    aggregator = create_aggregator(args.strategy)

    print(f"Testing {args.strategy} with {args.clients} clients")
    result = aggregator.aggregate(updates)

    if result:
        print(f"Aggregation successful:")
        print(f"  Clients: {result.num_clients}")
        print(f"  Total samples: {result.total_samples}")
        print(f"  Metrics: {result.round_metrics}")
        print(f"  Weight shapes: {[w.shape for w in result.global_weights]}")
    else:
        print("Aggregation failed")


if __name__ == "__main__":
    main()
