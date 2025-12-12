"""
Federated Learning Module for Multi-Site Medical Device Security

Enables privacy-preserving collaborative model training across multiple
medical institutions without sharing raw patient data.

Components:
    - FederatedServer: Coordinates federated training rounds
    - FederatedClient: Performs local training on device data
    - ModelAggregator: Implements aggregation strategies (FedAvg, FedProx)
    - PrivacyEngine: Differential privacy and secure aggregation

Usage:
    # Server (coordinator)
    from medtech_ai_security.federated import FederatedServer

    server = FederatedServer(
        model_architecture="anomaly_detector",
        min_clients=3,
        rounds=10
    )
    server.start()

    # Client (medical institution)
    from medtech_ai_security.federated import FederatedClient

    client = FederatedClient(
        server_address="coordinator.hospital.org:50051",
        client_id="hospital_a",
        local_data_path="data/local_traffic.csv"
    )
    client.join_federation()

FDA Compliance:
    - HIPAA-compliant: No raw PHI leaves institution
    - Audit trail for all model updates
    - Differential privacy for gradient protection
"""

from medtech_ai_security.federated.server import FederatedServer
from medtech_ai_security.federated.client import FederatedClient
from medtech_ai_security.federated.aggregator import (
    ModelAggregator,
    FedAvgAggregator,
    FedProxAggregator,
)
from medtech_ai_security.federated.privacy import (
    PrivacyEngine,
    DifferentialPrivacy,
    SecureAggregation,
)

__all__ = [
    "FederatedServer",
    "FederatedClient",
    "ModelAggregator",
    "FedAvgAggregator",
    "FedProxAggregator",
    "PrivacyEngine",
    "DifferentialPrivacy",
    "SecureAggregation",
]
