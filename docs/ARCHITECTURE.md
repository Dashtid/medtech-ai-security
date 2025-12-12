# Architecture Documentation

MedTech AI Security Platform - System Architecture

## Overview

MedTech AI Security is a modular AI-powered cybersecurity platform for medical devices. The system provides threat intelligence, vulnerability detection, anomaly detection, adversarial ML testing, and supply chain risk analysis.

## System Architecture

```
+------------------------------------------------------------------+
|                      External Systems                              |
+------------------------------------------------------------------+
|  NVD API  |  CISA ICS-CERT  |  DefectDojo  |  Medical Devices    |
+------------------------------------------------------------------+
      |            |                |                  |
      v            v                v                  v
+------------------------------------------------------------------+
|                         API Gateway                               |
|                    (FastAPI / Ingress)                            |
+------------------------------------------------------------------+
      |            |                |                  |
      v            v                v                  v
+------------------------------------------------------------------+
|                      Core Services                                 |
+------------------------------------------------------------------+
| Threat Intel | SBOM Analyzer | Anomaly Detector | Adversarial ML |
+------------------------------------------------------------------+
      |            |                |                  |
      v            v                v                  v
+------------------------------------------------------------------+
|                      ML Models Layer                              |
+------------------------------------------------------------------+
|  NLP Enrichment | GNN Risk Scorer | Autoencoder | Attack/Defense |
+------------------------------------------------------------------+
      |            |                |                  |
      v            v                v                  v
+------------------------------------------------------------------+
|                      Data Layer                                   |
+------------------------------------------------------------------+
|   CVE Database  |  SBOM Storage  |  Traffic Logs  |  Model Store |
+------------------------------------------------------------------+
```

## Component Details

### 1. Threat Intelligence Module

**Purpose**: Collect, enrich, and analyze vulnerability data for medical devices.

**Components**:
- `nvd_scraper.py` - NVD API client for CVE collection
- `cisa_scraper.py` - CISA ICS-CERT advisory parser
- `claude_processor.py` - AI-powered vulnerability enrichment

**Data Flow**:
```
NVD/CISA APIs --> Scrapers --> Raw CVE Data --> Claude Enrichment --> Enriched CVEs
```

**Key Features**:
- Rate-limited API access (6s for NVD, 2s for CISA)
- Medical device filtering by keywords
- Clinical impact assessment
- Device type classification

### 2. ML Risk Scoring Module

**Purpose**: Predict vulnerability risk priority using machine learning.

**Components**:
- `risk_scorer.py` - Ensemble classifier (Naive Bayes + KNN)

**Model Architecture**:
```
Input Features (12):
  - CVSS Base Score (normalized)
  - Attack Vector (one-hot)
  - Attack Complexity
  - Privileges Required
  - User Interaction
  - Scope
  - Confidentiality Impact
  - Integrity Impact
  - Availability Impact
  - Device Type (encoded)
  - CWE Category
  - Clinical Impact Score

Ensemble:
  - Naive Bayes (probability estimation)
  - K-Nearest Neighbors (pattern matching)
  - Weighted voting

Output:
  - Risk Priority: Critical / High / Medium / Low
  - Confidence Score: 0-100%
```

**Performance**: 75% accuracy on medical device CVE dataset

### 3. Anomaly Detection Module

**Purpose**: Detect malicious network traffic in DICOM/HL7 protocols.

**Components**:
- `traffic_generator.py` - Synthetic traffic generator
- `detector.py` - Autoencoder-based anomaly detector

**Model Architecture**:
```
Autoencoder:
  Encoder:
    - Input: 16 features
    - Dense(64, ReLU)
    - Dense(32, ReLU)
    - Dense(16, ReLU) [latent space]

  Decoder:
    - Dense(32, ReLU)
    - Dense(64, ReLU)
    - Dense(16, linear) [reconstruction]

  Loss: MSE (reconstruction error)
  Threshold: Dynamic (percentile-based)
```

**Features Extracted**:
- Timing features (inter-arrival time, session duration)
- Network features (bytes transferred, packet counts)
- Protocol features (DICOM commands, HL7 message types)
- Statistical features (entropy, variance)

**Attack Types Detected**:
1. Data exfiltration
2. Message injection
3. Ransomware patterns
4. DoS attacks
5. Protocol violations
6. Unauthorized access
7. Replay attacks
8. Man-in-the-middle
9. Firmware tampering
10. Credential theft

**Performance**: 92.5% accuracy, 0.86 AUC

### 4. Adversarial ML Module

**Purpose**: Test robustness of medical AI models against adversarial attacks.

**Components**:
- `attacks.py` - Attack implementations (FGSM, PGD, C&W)
- `defenses.py` - Defense mechanisms
- `evaluator.py` - Robustness evaluation framework

**Attack Methods**:

| Method | Type | Description |
|--------|------|-------------|
| FGSM | White-box | Fast Gradient Sign Method - single-step gradient attack |
| PGD | White-box | Projected Gradient Descent - iterative attack |
| C&W | White-box | Carlini & Wagner - optimization-based minimal perturbation |

**Defense Methods**:

| Method | Type | Description |
|--------|------|-------------|
| JPEG Compression | Preprocessing | Remove high-frequency perturbations |
| Gaussian Blur | Preprocessing | Smooth adversarial noise |
| Feature Squeezing | Preprocessing | Reduce color depth |
| Adversarial Training | Model-based | Train with adversarial examples |

**Clinical Impact Assessment**:
- False Negative (missed detection) = CRITICAL for diagnosis
- False Positive (false alarm) = HIGH for treatment planning
- Robustness score weighted by clinical context

### 5. SBOM Analysis Module

**Purpose**: Analyze software supply chain risks using Graph Neural Networks.

**Components**:
- `parser.py` - SBOM format parsers (CycloneDX, SPDX)
- `graph_builder.py` - Dependency graph construction
- `gnn_model.py` - Graph Neural Network for risk propagation
- `risk_scorer.py` - Multi-factor risk scoring
- `analyzer.py` - Main analysis orchestrator

**GNN Architecture**:
```
Input:
  - Node features: 88-dimensional
    - Package name embedding (64-dim)
    - Version encoding (8-dim)
    - Package type one-hot (8-dim)
    - CVSS scores (4-dim)
    - Dependency depth (2-dim)
    - License risk (2-dim)

  - Edge features: dependency relationships
    - Direct dependency
    - Dev dependency
    - Optional dependency

Graph Layers:
  1. GCNConv(88, 64) + ReLU + Dropout(0.3)
  2. GATConv(64, 64, heads=4) + ReLU + Dropout(0.3)
  3. GCNConv(256, 64) + ReLU

Readout:
  - Global mean pooling
  - Dense(64, 32, ReLU)
  - Dense(32, 3, Softmax) [clean/direct_vuln/transitive_vuln]
```

**Risk Factors**:
1. Vulnerability score (weighted by CVSS)
2. License compliance risk
3. Dependency depth (transitive risk)
4. Package centrality (graph position)
5. Maintenance status (last update)

**FDA Compliance**:
- SBOM component enumeration
- Vulnerability disclosure requirements
- 510(k) submission formatting

### 6. Live Traffic Capture Module

**Purpose**: Real-time capture and analysis of DICOM and HL7 network traffic.

**Components**:
- `dicom_capture.py` - DICOM protocol packet capture and parsing
- `hl7_capture.py` - HL7 v2.x MLLP message capture and parsing
- `traffic_analyzer.py` - Unified analyzer with anomaly detection integration

**Protocol Support**:

| Protocol | Standard Ports | Features |
|----------|---------------|----------|
| DICOM | 104, 11112 | PDU parsing, command extraction, AE title tracking |
| HL7 v2.x | 2575, 5000 | MLLP framing, MSH segment parsing, message type classification |

**Attack Detection**:
- Large data transfers (potential exfiltration)
- Rapid connection attempts (DoS detection)
- Unknown Application Entity titles
- After-hours activity monitoring
- PHI exfiltration patterns

**Capture Modes**:
1. **Live capture**: Real-time packet capture using scapy (requires root)
2. **PCAP replay**: Analysis of captured traffic files
3. **Simulation**: Synthetic traffic generation for testing

### 7. Federated Learning Module

**Purpose**: Privacy-preserving collaborative model training across multiple medical institutions.

**Components**:
- `server.py` - Federated learning coordinator
- `client.py` - Institution-side training client
- `aggregator.py` - Model aggregation strategies
- `privacy.py` - Differential privacy and secure aggregation

**Architecture**:
```
                    +-------------------+
                    | Federated Server  |
                    | (Coordinator)     |
                    +-------------------+
                           |
          +----------------+----------------+
          |                |                |
    +----------+     +----------+     +----------+
    | Client A |     | Client B |     | Client C |
    | Hospital |     | Clinic   |     | Lab      |
    +----------+     +----------+     +----------+
         |                |                |
    [Local Data]    [Local Data]    [Local Data]
    (Never shared)  (Never shared)  (Never shared)
```

**Aggregation Strategies**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| FedAvg | Weighted average by sample count | Homogeneous data |
| FedProx | Proximal regularization | Heterogeneous systems |
| FedNova | Normalized averaging | Variable local epochs |
| Robust | Coordinate-wise median | Byzantine fault tolerance |

**Privacy Mechanisms**:

1. **Differential Privacy**:
   - Gradient clipping (bound sensitivity)
   - Gaussian noise injection
   - Privacy budget accounting (epsilon, delta)

2. **Secure Aggregation**:
   - Client mask generation
   - Pairwise key exchange
   - Server sees only aggregate

**HIPAA Compliance**:
- Raw PHI never leaves institution
- Model updates protected with DP noise
- Audit logging for all operations
- Configurable privacy budget

**Training Flow**:
```
1. Server broadcasts global model
2. Clients train locally on institution data
3. Clients apply differential privacy to gradients
4. Clients send protected updates to server
5. Server aggregates updates (FedAvg/FedProx)
6. Repeat for N rounds
```

## API Architecture

**Framework**: FastAPI with automatic OpenAPI documentation

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/ready` | GET | Kubernetes readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/sbom/analyze` | POST | Analyze SBOM for risks |
| `/api/v1/anomaly/detect` | POST | Detect traffic anomalies |
| `/api/v1/cve/{id}` | GET | Lookup CVE details |
| `/api/v1/adversarial/test` | POST | Test model robustness |

**Authentication**: API key via `X-API-Key` header

**Rate Limiting**: 10 requests/second per client

## Deployment Architecture

### Kubernetes Deployment

```
Namespace: medtech-security
+------------------------------------------------------------------+
|  Ingress (NGINX)                                                  |
|  - TLS termination                                                |
|  - Rate limiting                                                  |
|  - Path-based routing                                             |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
|  Services                                                         |
+------------------------------------------------------------------+
|  sbom-analyzer     | anomaly-detector   | threat-intel-scanner   |
|  (Deployment)      | (Deployment)       | (CronJob)              |
|  Replicas: 2-10    | Replicas: 2-8      | Schedule: Daily        |
|  HPA: CPU 70%      | HPA: CPU 70%       |                        |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
|  Persistent Storage                                               |
+------------------------------------------------------------------+
|  PVC: medtech-security-data    | PVC: medtech-security-models    |
|  Size: 10Gi (RWX)              | Size: 5Gi (ROX)                  |
+------------------------------------------------------------------+
```

### Security Configuration

- **RBAC**: Minimal permissions (read configmaps/secrets, create events)
- **Network Policy**: Restrict ingress/egress to required services
- **Pod Security**: Non-root user, read-only filesystem, no privilege escalation
- **Secrets**: External secrets management recommended (Vault, AWS SM)

## Data Flow Diagrams

### SBOM Analysis Flow

```
User/CI Pipeline
      |
      v
[SBOM Upload] --> [Parser] --> [Graph Builder] --> [GNN Model]
                                                         |
                                                         v
                                               [Risk Scorer] --> [Report]
                                                         |
                                                         v
                                               [DefectDojo Import]
```

### Anomaly Detection Flow

```
Medical Device Network
      |
      v
[Traffic Capture] --> [Feature Extraction] --> [Autoencoder]
                                                     |
                                             [Reconstruction Error]
                                                     |
                                                     v
                              [Threshold Check] --> [Alert/Log]
                                                     |
                                                     v
                                               [Dashboard]
```

### Threat Intelligence Flow

```
NVD API              CISA ICS-CERT
    |                      |
    v                      v
[NVD Scraper]       [CISA Scraper]
    |                      |
    +----------+-----------+
               |
               v
      [CVE Aggregation]
               |
               v
      [Claude Enrichment]
               |
               v
      [Risk Scoring ML]
               |
               v
      [DefectDojo Import]
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| ML Framework | TensorFlow 2.13+, Keras 3.x |
| API Framework | FastAPI |
| Package Manager | UV |
| Containerization | Docker |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Grafana |
| Vulnerability Management | DefectDojo |

## Performance Characteristics

| Component | Latency | Throughput |
|-----------|---------|------------|
| SBOM Analysis | 2-10s | 100 SBOMs/hour |
| Anomaly Detection | 10ms | 10K records/second |
| CVE Lookup | 100ms | 1000 requests/second |
| Adversarial Testing | 30s-5min | 10 models/hour |

## Scalability Considerations

1. **Horizontal Scaling**: All services support horizontal pod autoscaling
2. **Caching**: Consider Redis for CVE lookup caching
3. **Async Processing**: Background jobs for long-running analyses
4. **Model Serving**: TensorFlow Serving for high-throughput inference
5. **Database**: PostgreSQL for persistent CVE storage (future)

## Security Considerations

1. **Input Validation**: All API inputs validated via Pydantic
2. **Rate Limiting**: Prevent abuse of external API calls
3. **Secrets Management**: No hardcoded credentials
4. **Audit Logging**: All security events logged
5. **TLS**: All external communications encrypted
6. **Container Security**: Minimal base images, no root

## Future Architecture Enhancements

1. **Event-Driven Architecture**: Kafka/RabbitMQ for async processing
2. **Service Mesh**: Istio for advanced traffic management
3. **Federated Learning**: Multi-site model training
4. **Real-time Dashboard**: WebSocket-based live monitoring
5. **DICOM/HL7 Capture**: pcap integration for live traffic
