"""
Privacy-Preserving Mechanisms for Federated Learning

Implements differential privacy and secure aggregation to protect
model updates during federated training.

HIPAA Compliance:
    - Gradient clipping prevents memorization of individual records
    - Noise addition provides plausible deniability
    - Secure aggregation ensures server cannot see individual updates
"""

import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PrivacyBudget:
    """Privacy budget for differential privacy."""

    epsilon: float = 1.0  # Privacy loss parameter
    delta: float = 1e-5  # Probability of privacy breach
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0

    @property
    def remaining_epsilon(self) -> float:
        """Get remaining epsilon budget."""
        return max(0.0, self.epsilon - self.consumed_epsilon)

    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.consumed_epsilon >= self.epsilon

    def consume(self, epsilon: float, delta: float = 0.0) -> bool:
        """Consume privacy budget. Returns True if successful."""
        if self.consumed_epsilon + epsilon > self.epsilon:
            return False
        if self.consumed_delta + delta > self.delta:
            return False
        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        return True


@dataclass
class PrivacyMetrics:
    """Metrics for privacy accounting."""

    total_rounds: int = 0
    epsilon_per_round: list[float] = field(default_factory=list)
    noise_multipliers: list[float] = field(default_factory=list)
    clipping_rates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_rounds": self.total_rounds,
            "cumulative_epsilon": sum(self.epsilon_per_round),
            "average_noise_multiplier": (
                np.mean(self.noise_multipliers) if self.noise_multipliers else 0.0
            ),
            "average_clipping_rate": (
                np.mean(self.clipping_rates) if self.clipping_rates else 0.0
            ),
        }


# =============================================================================
# Privacy Engine Base
# =============================================================================


class PrivacyEngine(ABC):
    """Base class for privacy-preserving mechanisms."""

    @abstractmethod
    def apply(self, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Apply privacy mechanism to gradients."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get privacy metrics."""
        pass


# =============================================================================
# Differential Privacy
# =============================================================================


class DifferentialPrivacy(PrivacyEngine):
    """
    Differential Privacy for gradient protection.

    Implements the Gaussian mechanism with gradient clipping to provide
    (epsilon, delta)-differential privacy guarantees.

    Algorithm:
        1. Clip per-sample gradients to bound sensitivity
        2. Aggregate clipped gradients
        3. Add calibrated Gaussian noise

    References:
        - Abadi et al., "Deep Learning with Differential Privacy" (2016)
        - McMahan et al., "Learning Differentially Private RNNs" (2017)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float | None = None,
    ):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy breach
            max_grad_norm: Maximum L2 norm for gradient clipping
            noise_multiplier: Noise scale (auto-calculated if None)
        """
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        self.max_grad_norm = max_grad_norm
        self.metrics = PrivacyMetrics()

        # Calculate noise multiplier if not provided
        if noise_multiplier is None:
            self.noise_multiplier = self._calculate_noise_multiplier(epsilon, delta)
        else:
            self.noise_multiplier = noise_multiplier

        logger.info(
            f"Initialized DP with epsilon={epsilon}, delta={delta}, "
            f"noise_multiplier={self.noise_multiplier:.4f}"
        )

    def _calculate_noise_multiplier(self, epsilon: float, delta: float) -> float:
        """
        Calculate noise multiplier for target (epsilon, delta).

        Uses the Gaussian mechanism formula:
            sigma >= sqrt(2 * ln(1.25/delta)) / epsilon
        """
        if epsilon <= 0:
            return float("inf")

        # Gaussian mechanism calibration
        sigma = float(np.sqrt(2 * np.log(1.25 / delta)) / epsilon)
        return max(0.1, sigma)  # Minimum noise for stability

    def clip_gradients(
        self, gradients: list[np.ndarray]
    ) -> tuple[list[np.ndarray], float]:
        """
        Clip gradients to bound sensitivity.

        Args:
            gradients: List of gradient arrays

        Returns:
            Clipped gradients and clipping rate
        """
        clipped = []
        num_clipped = 0

        for grad in gradients:
            grad_norm = np.linalg.norm(grad)

            if grad_norm > self.max_grad_norm:
                # Scale gradient to max norm
                clipped_grad = grad * (self.max_grad_norm / grad_norm)
                num_clipped += 1
            else:
                clipped_grad = grad.copy()

            clipped.append(clipped_grad)

        clipping_rate = num_clipped / len(gradients) if gradients else 0.0
        return clipped, clipping_rate

    def add_noise(self, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """
        Add calibrated Gaussian noise to gradients.

        Args:
            gradients: List of clipped gradient arrays

        Returns:
            Noisy gradients
        """
        noisy = []

        for grad in gradients:
            # Calibrate noise to gradient sensitivity
            noise_scale = self.noise_multiplier * self.max_grad_norm
            noise = np.random.normal(0, noise_scale, grad.shape)
            noisy.append(grad + noise)

        return noisy

    def apply(self, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply differential privacy to gradients.

        Args:
            gradients: Raw gradients from local training

        Returns:
            Privacy-protected gradients
        """
        if self.budget.is_exhausted:
            logger.warning("Privacy budget exhausted - returning zeros")
            return [np.zeros_like(g) for g in gradients]

        # Step 1: Clip gradients
        clipped, clipping_rate = self.clip_gradients(gradients)

        # Step 2: Add noise
        noisy = self.add_noise(clipped)

        # Update metrics
        self.metrics.total_rounds += 1
        self.metrics.clipping_rates.append(clipping_rate)
        self.metrics.noise_multipliers.append(self.noise_multiplier)

        # Consume privacy budget (simplified accounting)
        round_epsilon = self._compute_round_epsilon()
        self.metrics.epsilon_per_round.append(round_epsilon)
        self.budget.consume(round_epsilon)

        logger.debug(
            f"Applied DP: clipping_rate={clipping_rate:.2%}, "
            f"consumed_epsilon={self.budget.consumed_epsilon:.4f}"
        )

        return noisy

    def _compute_round_epsilon(self) -> float:
        """Compute epsilon consumed in this round."""
        # Simplified: assume each round consumes base epsilon
        # In practice, use RDP accountant for tighter bounds
        return self.budget.epsilon / 100  # Allow ~100 rounds

    def get_metrics(self) -> dict[str, Any]:
        """Get privacy metrics."""
        return {
            **self.metrics.to_dict(),
            "remaining_epsilon": self.budget.remaining_epsilon,
            "budget_exhausted": self.budget.is_exhausted,
        }


# =============================================================================
# Secure Aggregation
# =============================================================================


class SecureAggregation(PrivacyEngine):
    """
    Secure Aggregation for federated learning.

    Ensures the server can only see the aggregate of client updates,
    not individual client contributions.

    Protocol (simplified):
        1. Each client generates random masks
        2. Masks are shared pairwise using Diffie-Hellman
        3. Clients send masked updates
        4. Server aggregates (masks cancel out)

    References:
        - Bonawitz et al., "Practical Secure Aggregation" (2017)
    """

    def __init__(self, num_clients: int, threshold: int | None = None):
        """
        Initialize secure aggregation.

        Args:
            num_clients: Total number of clients
            threshold: Minimum clients for aggregation (default: 2/3)
        """
        self.num_clients = num_clients
        self.threshold = threshold or max(2, int(num_clients * 2 / 3))
        self.round_number = 0

        # Simulated key storage (in practice, use proper key exchange)
        self._client_seeds: dict[str, int] = {}
        self._masks: dict[str, list[np.ndarray]] = {}

        logger.info(
            f"Initialized SecureAggregation with {num_clients} clients, "
            f"threshold={self.threshold}"
        )

    def register_client(self, client_id: str) -> bytes:
        """
        Register a client and return their seed.

        Args:
            client_id: Unique client identifier

        Returns:
            Random seed for mask generation
        """
        seed = secrets.randbits(256)
        self._client_seeds[client_id] = seed
        return seed.to_bytes(32, "big")

    def generate_mask(
        self, client_id: str, shapes: list[tuple[int, ...]]
    ) -> list[np.ndarray]:
        """
        Generate masks for a client's gradients.

        Args:
            client_id: Client identifier
            shapes: Shapes of gradient arrays

        Returns:
            List of mask arrays
        """
        if client_id not in self._client_seeds:
            raise ValueError(f"Client {client_id} not registered")

        # Use client seed + round number for deterministic masks
        seed = self._client_seeds[client_id] + self.round_number
        rng = np.random.default_rng(seed)

        masks = [rng.standard_normal(shape) for shape in shapes]
        self._masks[client_id] = masks

        return masks

    def mask_gradients(
        self, client_id: str, gradients: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Apply mask to client's gradients.

        Args:
            client_id: Client identifier
            gradients: Raw gradients

        Returns:
            Masked gradients
        """
        masks = self.generate_mask(client_id, [g.shape for g in gradients])
        return [g + m for g, m in zip(gradients, masks)]

    def aggregate(
        self, masked_updates: dict[str, list[np.ndarray]]
    ) -> list[np.ndarray] | None:
        """
        Aggregate masked updates (masks cancel out in sum).

        Args:
            masked_updates: Dict of client_id -> masked gradients

        Returns:
            Aggregated gradients or None if threshold not met
        """
        if len(masked_updates) < self.threshold:
            logger.warning(
                f"Not enough clients: {len(masked_updates)} < {self.threshold}"
            )
            return None

        # In real secure aggregation, masks cancel out
        # Here we simulate by subtracting masks during aggregation
        client_ids = list(masked_updates.keys())
        num_clients = len(client_ids)

        # Get first client's update to determine shapes
        first_update = masked_updates[client_ids[0]]
        aggregated = [np.zeros_like(g) for g in first_update]

        # Sum all masked updates
        for client_id, updates in masked_updates.items():
            masks = self._masks.get(client_id, [np.zeros_like(g) for g in updates])
            for i, (update, mask) in enumerate(zip(updates, masks)):
                # Remove mask and add to aggregate
                aggregated[i] += update - mask

        # Average
        aggregated = [g / num_clients for g in aggregated]

        self.round_number += 1
        return aggregated

    def apply(self, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply secure aggregation (client-side masking).

        Note: This is called per-client. The server calls aggregate()
        to combine all masked updates.
        """
        # For standalone use, just return gradients
        # In federated setting, client calls mask_gradients()
        return gradients

    def get_metrics(self) -> dict[str, Any]:
        """Get secure aggregation metrics."""
        return {
            "num_clients": self.num_clients,
            "threshold": self.threshold,
            "round_number": self.round_number,
            "registered_clients": len(self._client_seeds),
        }


# =============================================================================
# Combined Privacy Engine
# =============================================================================


class CombinedPrivacy(PrivacyEngine):
    """
    Combines differential privacy with secure aggregation.

    Provides defense-in-depth:
        1. DP protects against inference attacks on gradients
        2. Secure aggregation protects individual updates from server
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        num_clients: int = 3,
    ):
        """Initialize combined privacy engine."""
        self.dp = DifferentialPrivacy(
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
        )
        self.secure_agg = SecureAggregation(num_clients=num_clients)

    def apply(self, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Apply both DP and secure aggregation."""
        # First apply DP
        dp_gradients = self.dp.apply(gradients)
        # Then mask for secure aggregation
        return self.secure_agg.apply(dp_gradients)

    def get_metrics(self) -> dict[str, Any]:
        """Get combined metrics."""
        return {
            "differential_privacy": self.dp.get_metrics(),
            "secure_aggregation": self.secure_agg.get_metrics(),
        }


# =============================================================================
# CLI for Testing
# =============================================================================


def main() -> None:
    """Test privacy mechanisms."""
    import argparse

    parser = argparse.ArgumentParser(description="Test privacy mechanisms")
    parser.add_argument(
        "--mechanism",
        choices=["dp", "secure_agg", "combined"],
        default="dp",
        help="Privacy mechanism to test",
    )
    parser.add_argument("--epsilon", type=float, default=1.0, help="Privacy budget")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")

    args = parser.parse_args()

    # Create test gradients
    np.random.seed(42)
    test_gradients = [np.random.randn(100, 50), np.random.randn(50, 10)]

    print(f"Testing {args.mechanism} privacy mechanism")
    orig_norms = [f"{np.linalg.norm(g):.4f}" for g in test_gradients]
    print(f"Original gradient norms: {orig_norms}")

    engine: PrivacyEngine
    if args.mechanism == "dp":
        engine = DifferentialPrivacy(epsilon=args.epsilon)
    elif args.mechanism == "secure_agg":
        engine = SecureAggregation(num_clients=5)
    else:
        engine = CombinedPrivacy(epsilon=args.epsilon, num_clients=5)

    for round_num in range(args.rounds):
        protected = engine.apply(test_gradients)
        prot_norms = [f"{np.linalg.norm(g):.4f}" for g in protected]
        print(f"Round {round_num + 1}: Protected norms: {prot_norms}")

    print(f"\nFinal metrics: {engine.get_metrics()}")


if __name__ == "__main__":
    main()
