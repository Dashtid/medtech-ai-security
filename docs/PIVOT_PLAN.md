# Strategic Pivot: Medical Device Cybersecurity AI

**Date:** 2025-11-06
**Status:** Transition initiated
**Repository:** biomedical-ai -> medtech-ai-security

---

## Executive Summary

This document outlines the strategic transition from pure biomedical imaging AI to **AI-powered medical device cybersecurity**. The pivot leverages completed multi-task learning foundation while aligning with real-world expertise in medical device security, vulnerability management, and healthcare compliance.

**Core Rationale:**
- Unique intersection of medical domain knowledge + AI/ML + cybersecurity
- Addresses critical healthcare security challenges (FDA, EU MDR, ICS-CERT)
- Leverages existing production infrastructure (K3s, DefectDojo, RAG backend)
- Aligns with role as medical device cybersecurity engineer at Hermes Medical Solutions

---

## Foundation Established

### Completed Biomedical AI Work

**Multi-Task PET/CT Learning System:**
- Architecture: U-Net with 31.6M parameters (shared encoder + dual decoders)
- Training: 30 epochs completed on synthetic PET/CT data
- Performance: DICE 0.340, C-index 0.669 (training)
- Model size: 363 MB FP32
- Documentation: 1,500+ lines (portfolio-ready)

**Transferable Skills Demonstrated:**
1. **Multi-task learning**: Simultaneous optimization of multiple objectives
2. **Uncertainty quantification**: Monte Carlo Dropout for Bayesian predictions
3. **Loss function design**: Focal Tversky for severe class imbalance
4. **Production optimization**: INT8 quantization, model pruning, TFLite conversion
5. **Medical domain**: PET/CT imaging, survival analysis, clinical metrics
6. **Professional engineering**: Type hints, testing, CI/CD, documentation

---

## New Direction: Medical Device Cybersecurity AI

### Problem Space

**Healthcare Security Challenges:**
- Medical devices have 1,000+ known vulnerabilities (NVD database)
- FDA publishes 50+ medical device cybersecurity advisories annually
- ICS-CERT issues critical alerts for medical systems
- Manual vulnerability assessment is time-consuming and error-prone
- AI systems in medical devices are vulnerable to adversarial attacks
- SBOM analysis requires domain expertise and automation

**Why AI/ML?**
- NLP can extract structured threat intelligence from unstructured advisories
- ML can identify vulnerability patterns across device types
- Anomaly detection can flag suspicious medical device network traffic
- Graph neural networks can analyze complex SBOM dependencies
- Adversarial ML can test robustness of medical AI models

### Target Applications

#### 1. NLP-Based Threat Intelligence Extraction

**Objective:** Automatically extract and classify medical device vulnerabilities from CVE/ICS-CERT advisories

**Approach:**
- Fine-tune LLM (Llama 3.1 via Ollama) for medical device context
- Extract: CVE ID, affected devices, vulnerability type, CVSS score, remediation
- Classify: Device type (imaging, infusion, monitoring), criticality, exploitability
- Entity recognition: Manufacturer, device model, software version, protocol
- Leverage existing RAG backend infrastructure for document processing

**Data Sources:**
- NVD (National Vulnerability Database) API
- ICS-CERT medical device advisories
- FDA cybersecurity communications
- Manufacturer security bulletins

**Output:**
- Structured JSON with extracted vulnerability intelligence
- Automatic DefectDojo finding creation
- Risk scoring based on device context and clinical impact

#### 2. ML-Powered Vulnerability Detection

**Objective:** Predict vulnerability likelihood for medical devices based on features

**Approach:**
- Feature engineering: Device type, software stack, communication protocols, regulatory class
- Training data: Historical CVE database + SBOM analysis
- Model: Random Forest or Gradient Boosting for interpretability
- Output: Vulnerability risk score + explainable factors

**Integration:**
- PMS-scanner (existing tool) provides SBOM via Grype
- ML model analyzes SBOM + device metadata
- High-risk predictions trigger manual security review
- Findings pushed to DefectDojo for tracking

#### 3. Anomaly Detection for Medical Device Traffic

**Objective:** Identify suspicious network behavior in medical device protocols

**Approach:**
- Protocol analysis: DICOM, HL7, Modbus, proprietary protocols
- Baseline: Normal communication patterns per device type
- Model: Autoencoder for unsupervised anomaly detection
- Features: Packet size, timing, protocol fields, connection patterns

**Use Cases:**
- Detect unauthorized DICOM queries (data exfiltration)
- Identify HL7 message injection attacks
- Flag unusual device communication patterns
- Alert on protocol violations

**Data Collection:**
- Lab environment with medical device simulators
- Synthetic attack traffic generation
- Real-world anonymized packet captures (with permission)

#### 4. Adversarial ML for Medical AI Models

**Objective:** Test robustness of medical AI models against adversarial attacks

**Approach:**
- Attack techniques: FGSM, PGD, C&W for image-based models
- Defense techniques: Adversarial training, certified defenses
- Evaluation: Robustness metrics, attack success rate, CVSS scoring
- Documentation: Vulnerability reports for AI model developers

**Target Models:**
- Medical imaging classifiers (X-ray, CT, MRI)
- Diagnostic decision support systems
- Drug dosing AI algorithms
- Our own biomedical AI model as test case

**Output:**
- Adversarial robustness testing framework
- Vulnerability reports with proof-of-concept attacks
- Defense recommendations for medical AI developers

#### 5. SBOM Analysis with Graph Neural Networks

**Objective:** Analyze software supply chain risk in medical devices

**Approach:**
- Graph representation: Dependencies as nodes, relationships as edges
- Node features: Package name, version, known CVEs, license
- Edge features: Dependency type (direct, transitive), version constraints
- GNN model: Predict vulnerability propagation and supply chain risk

**Integration:**
- Grype provides SBOM data (existing PMS-scanner capability)
- GNN analyzes dependency graph
- Risk scoring considers transitive vulnerabilities
- Visualization of high-risk dependency paths

---

## Technical Roadmap

### Phase 1: NLP Threat Intelligence (2-3 weeks)

**Goal:** Automated CVE extraction from advisories

**Tasks:**
1. Scrape NVD API for medical device CVEs (filter by CPE keywords)
2. Collect ICS-CERT medical advisories (web scraping)
3. Fine-tune Llama 3.1 for vulnerability extraction using existing RAG backend
4. Build JSON schema for structured threat intelligence
5. Create DefectDojo API integration for finding creation
6. Implement automated daily scraping pipeline

**Success Criteria:**
- Extract 100+ CVEs with 90% accuracy
- Automatic DefectDojo finding creation
- Daily threat intelligence updates

**Infrastructure:**
- Use existing Ollama deployment on lab server
- Leverage RAG backend for document ingestion
- Deploy on K3s cluster for production

### Phase 2: ML Vulnerability Detection (2-3 weeks)

**Goal:** Risk scoring for medical devices based on SBOM

**Tasks:**
1. Build training dataset from historical CVEs + SBOM data
2. Feature engineering: Device type, software stack, protocol, regulatory class
3. Train Random Forest model for vulnerability prediction
4. Integrate with PMS-scanner SBOM output
5. Create explainable risk reports
6. Deploy model as microservice on K3s

**Success Criteria:**
- Predict vulnerability risk with 75% accuracy
- Explainable factor attribution
- Integration with existing PMS-scanner workflow

**Infrastructure:**
- Python microservice (FastAPI)
- PostgreSQL for training data storage
- Prometheus metrics for model performance

### Phase 3: Anomaly Detection (4-6 weeks)

**Goal:** Network traffic anomaly detection for medical devices

**Tasks:**
1. Set up lab environment with medical device simulators
2. Capture baseline DICOM/HL7 traffic
3. Generate synthetic attack traffic
4. Train autoencoder for anomaly detection
5. Build real-time monitoring dashboard
6. Deploy on K3s with alerting to DefectDojo

**Success Criteria:**
- Detect 90% of attack traffic with <5% false positive rate
- Real-time monitoring with <1 second latency
- Integration with existing Prometheus/Grafana stack

**Infrastructure:**
- Network tap or mirror port for traffic capture
- Redis for real-time data buffering
- Grafana dashboard for visualization

### Phase 4: Adversarial ML (3-4 weeks)

**Goal:** Robustness testing framework for medical AI

**Tasks:**
1. Implement FGSM, PGD, C&W attacks
2. Test attacks on our biomedical AI model
3. Implement adversarial training defenses
4. Create vulnerability assessment methodology
5. Build automated robustness testing pipeline
6. Document findings and defense recommendations

**Success Criteria:**
- Successful attacks on 5+ medical AI models
- Robustness improvement via adversarial training
- Vulnerability report template for medical AI

**Infrastructure:**
- GPU resources for adversarial training (lab server)
- TensorFlow/PyTorch for attack implementation
- Jupyter notebooks for exploration and reporting

### Phase 5: SBOM Graph Analysis (4-6 weeks)

**Goal:** Supply chain risk analysis with GNNs

**Tasks:**
1. Parse SBOM into graph representation
2. Collect vulnerability data for dependencies
3. Implement Graph Neural Network (PyTorch Geometric)
4. Train model for vulnerability propagation prediction
5. Build dependency visualization tool
6. Integrate with PMS-scanner and DefectDojo

**Success Criteria:**
- Identify transitive vulnerability risks with 80% accuracy
- Visual dependency graphs with risk highlighting
- Automated supply chain risk reports

**Infrastructure:**
- Neo4j or NetworkX for graph storage
- PyTorch Geometric for GNN implementation
- D3.js or Cytoscape.js for visualization

---

## Infrastructure Leverage

### Existing Production Systems

**Lab Server (10.143.31.18):**
- K3s cluster for container orchestration
- Ollama for LLM inference (Llama 3.1)
- GPU: NVIDIA Quadro RTX 5000 (16GB VRAM)
- Docker for containerized services

**Deployed Services:**
- **DefectDojo**: Vulnerability management (PostgreSQL backend)
- **PMS-scanner**: Medical device vulnerability scanning (Grype, DefectDojo API)
- **RAG backend**: Document processing and LLM integration
- **Gitea**: Git repository and CI/CD
- **Prometheus/Grafana**: Monitoring and visualization

**Advantages:**
- No new infrastructure needed for Phase 1-2
- Existing CI/CD pipelines for automated testing
- Production-ready monitoring and alerting
- Secure environment for sensitive medical data

---

## Skills Transfer Matrix

| Biomedical AI Skill | Medical Device Security Application |
|---------------------|-------------------------------------|
| Multi-task learning | Simultaneous vulnerability detection + risk scoring |
| Uncertainty quantification | Confidence scores for ML predictions (reduce false positives) |
| Loss function design | Class imbalance in vulnerability datasets (rare critical CVEs) |
| Model optimization | Deploy lightweight models on edge medical devices |
| Medical domain knowledge | Understand clinical impact of security vulnerabilities |
| Data preprocessing | Clean and normalize vulnerability databases, packet captures |
| Evaluation metrics | Security-specific metrics (precision, recall, F1 for threat detection) |
| Production deployment | K8s deployment of ML microservices |
| Documentation | Security reports, vulnerability disclosures, compliance documentation |

---

## Compliance and Regulatory Considerations

**FDA Cybersecurity Guidance:**
- Premarket cybersecurity submissions require threat modeling
- SBOM requirements for medical devices (2023 guidance)
- Vulnerability disclosure and patching requirements

**EU MDR (Medical Device Regulation):**
- Cybersecurity is part of essential requirements
- Risk management per ISO 14971
- Post-market surveillance including cybersecurity monitoring

**IEC 62304 (Medical Device Software Lifecycle):**
- Security by design principles
- Software maintenance and updates
- Risk-based development processes

**How AI/ML Helps:**
- Automated threat modeling for premarket submissions
- SBOM analysis for regulatory compliance
- Continuous post-market surveillance with anomaly detection
- Documentation generation for regulatory filings

---

## Portfolio and Career Impact

### Resume Talking Points

**Unique Value Proposition:**
- "Combined medical device cybersecurity engineering with AI/ML to automate threat intelligence extraction, reducing manual analysis time by 80%"
- "Built NLP system for CVE extraction from 500+ medical device advisories, automatically creating DefectDojo findings"
- "Developed ML-powered vulnerability risk scoring for medical device SBOM analysis"
- "Created adversarial robustness testing framework for medical AI models, identifying exploitable weaknesses"

**Differentiators:**
- Medical domain knowledge (Hermes Medical Solutions, QMS, technical files)
- Production AI/ML deployment (K3s, Docker, CI/CD)
- Cybersecurity expertise (PMS-scanner, DefectDojo, vulnerability management)
- Regulatory awareness (FDA, EU MDR, IEC 62304)

### Interview Preparation

**Technical Questions:**
1. "How does NLP help with vulnerability management?"
   - Answer: Extracts structured data from unstructured advisories, automates finding creation, scales analysis beyond manual capacity

2. "What's the challenge with medical device ML models?"
   - Answer: Safety-critical applications, adversarial robustness, data privacy, regulatory compliance, high false positive cost

3. "How do you handle class imbalance in vulnerability datasets?"
   - Answer: Focal Tversky loss, SMOTE, cost-sensitive learning, ensemble methods (demonstrated in biomedical AI work)

4. "What's the risk of adversarial attacks on medical AI?"
   - Answer: Misdiagnosis, inappropriate treatment, patient harm. Need certified defenses and robustness testing.

### Publications and Presentations

**Potential Conference Submissions:**
- **IEEE EMBC** (Engineering in Medicine and Biology Conference): "AI-Powered Medical Device Cybersecurity"
- **Black Hat Medical Device Security Village**: Adversarial attacks on medical AI
- **DEF CON Bio Hacking Village**: Medical device network anomaly detection
- **USENIX Security**: SBOM supply chain risk analysis with graph neural networks

**Blog Posts:**
- "From Medical Imaging AI to Medical Device Security AI"
- "How Multi-Task Learning Applies to Cybersecurity"
- "Building an NLP System for CVE Extraction"
- "Adversarial Robustness Testing for Medical AI"

---

## Risk Mitigation

### Potential Challenges

**Challenge 1: Limited labeled data for medical device vulnerabilities**
- Mitigation: Transfer learning from general cybersecurity datasets, semi-supervised learning, synthetic data generation

**Challenge 2: Medical device simulators are expensive**
- Mitigation: Start with publicly documented protocols (DICOM, HL7), use open-source simulators, partner with academic institutions

**Challenge 3: Regulatory compliance for AI in medical devices**
- Mitigation: Focus on tools for security teams (not embedded in devices), follow FDA AI/ML guidance, maintain documentation

**Challenge 4: Adversarial ML research is sensitive**
- Mitigation: Responsible disclosure process, focus on defense recommendations, partner with device manufacturers

**Challenge 5: Graph neural networks are complex**
- Mitigation: Start with simpler models (Random Forest), incremental complexity, leverage existing GNN libraries (PyTorch Geometric)

---

## Success Metrics

### Technical Metrics

**Phase 1 (NLP Threat Intelligence):**
- Extraction accuracy: 90% for CVE ID, affected devices, CVSS score
- Coverage: 100+ medical device CVEs extracted
- Automation: Daily updates with zero manual intervention

**Phase 2 (ML Vulnerability Detection):**
- Prediction accuracy: 75% for vulnerability presence
- Explainability: Feature importance scores for all predictions
- Integration: PMS-scanner workflow with <5 minute analysis time

**Phase 3 (Anomaly Detection):**
- Detection rate: 90% of attack traffic
- False positive rate: <5%
- Latency: <1 second for real-time detection

**Phase 4 (Adversarial ML):**
- Attack success rate: >80% on undefended models
- Defense effectiveness: 50% reduction in attack success after adversarial training
- Documentation: 5+ vulnerability reports for medical AI models

**Phase 5 (SBOM GNN):**
- Transitive vulnerability detection: 80% accuracy
- Supply chain risk score: Actionable for 95% of SBOMs
- Visualization: Interactive dependency graphs for all devices

### Career Metrics

- GitHub repository with 100+ stars (demonstrates impact)
- Conference presentation or paper acceptance
- Blog post with 1,000+ views
- Job offers or promotions referencing this work
- Industry recognition (invited talks, consulting requests)

---

## Timeline

**Estimated Total: 16-24 weeks (4-6 months)**

| Phase | Duration | Completion Date |
|-------|----------|-----------------|
| Phase 1: NLP Threat Intelligence | 2-3 weeks | Week 3 |
| Phase 2: ML Vulnerability Detection | 2-3 weeks | Week 6 |
| Phase 3: Anomaly Detection | 4-6 weeks | Week 12 |
| Phase 4: Adversarial ML | 3-4 weeks | Week 16 |
| Phase 5: SBOM GNN | 4-6 weeks | Week 22 |

**Parallel Work:**
- Documentation continuous throughout
- Blog posts after each phase
- Conference submissions at 6-month mark

---

## Next Immediate Steps

1. **Rename repository** to "medtech-ai-security" (Git rename + remote update)
2. **Commit transition documents** (this PIVOT_PLAN.md + updated README.md)
3. **Set up NLP development environment** (verify Ollama deployment, RAG backend access)
4. **Start Phase 1 - Task 1**: Scrape NVD API for medical device CVEs (create script with filters)
5. **Research fine-tuning approaches** for Llama 3.1 (LoRA, QLoRA, full fine-tuning comparison)

---

## Conclusion

This strategic pivot leverages completed biomedical AI work while shifting focus to the unique intersection of medical device cybersecurity and AI/ML. The transition:

- **Aligns with career**: Medical device security engineer at Hermes Medical Solutions
- **Leverages infrastructure**: K3s, DefectDojo, PMS-scanner, RAG backend, Ollama
- **Demonstrates unique skills**: Medical domain + AI/ML + cybersecurity
- **Addresses real problems**: FDA guidance, EU MDR compliance, patient safety
- **Builds portfolio**: Publications, presentations, open-source tools

The biomedical AI foundation provides credibility in medical AI, multi-task learning, and production ML deployment, while the new direction focuses on critical healthcare security challenges that few practitioners are equipped to solve.

---

**Status:** Ready to begin Phase 1 (NLP Threat Intelligence)
**Next Milestone:** First 100 CVEs extracted and pushed to DefectDojo (2-3 weeks)
**Documentation:** This PIVOT_PLAN.md, updated README.md, commit history

**Let's build AI-powered medical device cybersecurity.**
