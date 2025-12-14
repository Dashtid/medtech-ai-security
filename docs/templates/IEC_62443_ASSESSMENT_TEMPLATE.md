# IEC 62443 Security Assessment Template

This template provides a structured approach for conducting security assessments
per IEC 62443 standards, recognized by FDA as a consensus standard for medical
device cybersecurity.

## Document Control

| Field | Value |
|-------|-------|
| Document ID | [ASSESSMENT-YYYY-NNN] |
| System Name | [System Name] |
| Version | 1.0 |
| Date | [YYYY-MM-DD] |
| Author | [Name] |
| Reviewer | [Name] |
| Approver | [Name] |

---

## 1. Executive Summary

### 1.1 Assessment Scope

[Describe the scope of the security assessment, including systems, zones, and
components covered.]

### 1.2 Target Security Level

| Parameter | Value |
|-----------|-------|
| Target Security Level | SL-[1-4] |
| Achieved Security Level | SL-[0-4] |
| Gap Status | [Compliant/Gap Identified] |

### 1.3 Key Findings

| Finding ID | Description | Severity | Status |
|------------|-------------|----------|--------|
| F-001 | [Description] | [High/Medium/Low] | [Open/Closed] |
| F-002 | [Description] | [High/Medium/Low] | [Open/Closed] |

---

## 2. System Description

### 2.1 System Overview

[Provide a high-level description of the medical device system, its purpose,
and primary functions.]

### 2.2 System Architecture

[Include or reference system architecture diagram]

### 2.3 Components Inventory

| Component ID | Name | Type | Version | Criticality |
|--------------|------|------|---------|-------------|
| C-001 | [Name] | [Hardware/Software/Firmware] | [Version] | [High/Medium/Low] |
| C-002 | [Name] | [Hardware/Software/Firmware] | [Version] | [High/Medium/Low] |

### 2.4 Data Flows

[Describe critical data flows including patient data, control commands, and
configuration data.]

---

## 3. Security Level Determination

### 3.1 Threat Analysis

Reference: IEC 62443-3-2

| Threat Category | Threat Description | Likelihood | Impact | Risk Level |
|-----------------|-------------------|------------|--------|------------|
| Unauthorized access | [Description] | [H/M/L] | [H/M/L] | [H/M/L] |
| Data tampering | [Description] | [H/M/L] | [H/M/L] | [H/M/L] |
| Denial of service | [Description] | [H/M/L] | [H/M/L] | [H/M/L] |

### 3.2 Security Level Target Justification

| Security Level | Description | Justification |
|----------------|-------------|---------------|
| SL-1 | Casual/coincidental violation | [Justify if applicable] |
| SL-2 | Intentional violation, simple means | [Justify if applicable] |
| SL-3 | Sophisticated attack, moderate resources | [Justify if applicable] |
| SL-4 | State-sponsored attack, extensive resources | [Justify if applicable] |

**Selected Target SL**: SL-[X]

**Rationale**: [Provide detailed justification for the selected security level]

---

## 4. Zone and Conduit Model

### 4.1 Zone Definitions

| Zone ID | Name | Description | Target SL | Assets |
|---------|------|-------------|-----------|--------|
| Z-01 | [Zone Name] | [Description] | SL-[X] | [Asset list] |
| Z-02 | [Zone Name] | [Description] | SL-[X] | [Asset list] |

### 4.2 Zone Diagrams

[Include zone and conduit diagram showing all zones and their interconnections]

### 4.3 Conduit Definitions

| Conduit ID | Name | Source Zone | Dest Zone | Protocol | SL |
|------------|------|-------------|-----------|----------|-----|
| CD-01 | [Name] | Z-01 | Z-02 | [Protocol] | SL-[X] |
| CD-02 | [Name] | Z-02 | Z-03 | [Protocol] | SL-[X] |

### 4.4 Conduit Security Controls

| Conduit ID | Security Controls | Implementation Status |
|------------|-------------------|----------------------|
| CD-01 | [Control list] | [Implemented/Planned/Gap] |
| CD-02 | [Control list] | [Implemented/Planned/Gap] |

---

## 5. Foundational Requirements Assessment

Reference: IEC 62443-3-3

### 5.1 FR 1: Identification and Authentication Control (IAC)

| SR ID | Requirement | SL-1 | SL-2 | SL-3 | SL-4 | Status | Evidence |
|-------|-------------|------|------|------|------|--------|----------|
| SR 1.1 | Human user identification and authentication | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 1.2 | Software process identification | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 1.3 | Account management | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 1.4 | Identifier management | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 1.5 | Authenticator management | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |

**FR 1 Assessment Summary**: [Compliant/Partial/Non-Compliant]

### 5.2 FR 2: Use Control (UC)

| SR ID | Requirement | SL-1 | SL-2 | SL-3 | SL-4 | Status | Evidence |
|-------|-------------|------|------|------|------|--------|----------|
| SR 2.1 | Authorization enforcement | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 2.2 | Wireless use control | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 2.3 | Use control for portable devices | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |

**FR 2 Assessment Summary**: [Compliant/Partial/Non-Compliant]

### 5.3 FR 3: System Integrity (SI)

| SR ID | Requirement | SL-1 | SL-2 | SL-3 | SL-4 | Status | Evidence |
|-------|-------------|------|------|------|------|--------|----------|
| SR 3.1 | Communication integrity | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 3.2 | Malicious code protection | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 3.3 | Security functionality verification | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 3.4 | Software and information integrity | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |

**FR 3 Assessment Summary**: [Compliant/Partial/Non-Compliant]

### 5.4 FR 4: Data Confidentiality (DC)

| SR ID | Requirement | SL-1 | SL-2 | SL-3 | SL-4 | Status | Evidence |
|-------|-------------|------|------|------|------|--------|----------|
| SR 4.1 | Information confidentiality | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 4.2 | Information persistence | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |

**FR 4 Assessment Summary**: [Compliant/Partial/Non-Compliant]

### 5.5 FR 5: Restricted Data Flow (RDF)

| SR ID | Requirement | SL-1 | SL-2 | SL-3 | SL-4 | Status | Evidence |
|-------|-------------|------|------|------|------|--------|----------|
| SR 5.1 | Network segmentation | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 5.2 | Zone boundary protection | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |

**FR 5 Assessment Summary**: [Compliant/Partial/Non-Compliant]

### 5.6 FR 6: Timely Response to Events (TRE)

| SR ID | Requirement | SL-1 | SL-2 | SL-3 | SL-4 | Status | Evidence |
|-------|-------------|------|------|------|------|--------|----------|
| SR 6.1 | Audit log accessibility | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 6.2 | Continuous monitoring | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |

**FR 6 Assessment Summary**: [Compliant/Partial/Non-Compliant]

### 5.7 FR 7: Resource Availability (RA)

| SR ID | Requirement | SL-1 | SL-2 | SL-3 | SL-4 | Status | Evidence |
|-------|-------------|------|------|------|------|--------|----------|
| SR 7.1 | Denial of service protection | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 7.2 | Resource management | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 7.3 | Control system backup | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |
| SR 7.4 | Control system recovery | [ ] | [ ] | [ ] | [ ] | [Status] | [Evidence ref] |

**FR 7 Assessment Summary**: [Compliant/Partial/Non-Compliant]

---

## 6. SOUP Inventory (IEC 62304)

Reference: IEC 62304 Medical device software life cycle processes

### 6.1 SOUP Component List

| ID | Component | Version | Vendor | Purpose | Risk Class | SBOM Ref |
|----|-----------|---------|--------|---------|------------|----------|
| SOUP-001 | [Name] | [Version] | [Vendor] | [Purpose] | [A/B/C] | [Ref] |
| SOUP-002 | [Name] | [Version] | [Vendor] | [Purpose] | [A/B/C] | [Ref] |

### 6.2 SOUP Risk Assessment

| SOUP ID | Known Vulnerabilities | Risk Score | Mitigation |
|---------|----------------------|------------|------------|
| SOUP-001 | [CVE list or None] | [H/M/L] | [Mitigation strategy] |
| SOUP-002 | [CVE list or None] | [H/M/L] | [Mitigation strategy] |

### 6.3 SOUP Update Management

| SOUP ID | Update Mechanism | Update Frequency | Last Updated |
|---------|-----------------|------------------|--------------|
| SOUP-001 | [Mechanism] | [Frequency] | [Date] |
| SOUP-002 | [Mechanism] | [Frequency] | [Date] |

---

## 7. Gap Analysis

### 7.1 Identified Gaps

| Gap ID | Requirement | Current State | Required State | Severity |
|--------|-------------|---------------|----------------|----------|
| GAP-001 | [SR X.X] | [Current] | [Required] | [H/M/L] |
| GAP-002 | [SR X.X] | [Current] | [Required] | [H/M/L] |

### 7.2 Gap Remediation Plan

| Gap ID | Remediation Action | Owner | Target Date | Status |
|--------|-------------------|-------|-------------|--------|
| GAP-001 | [Action] | [Owner] | [Date] | [Status] |
| GAP-002 | [Action] | [Owner] | [Date] | [Status] |

---

## 8. Recommendations

### 8.1 High Priority Recommendations

1. [Recommendation 1]
2. [Recommendation 2]

### 8.2 Medium Priority Recommendations

1. [Recommendation 1]
2. [Recommendation 2]

### 8.3 Low Priority Recommendations

1. [Recommendation 1]
2. [Recommendation 2]

---

## 9. FDA Compliance Mapping

### 9.1 FDA Premarket Cybersecurity Guidance Alignment

| FDA Requirement | IEC 62443 Mapping | Status |
|-----------------|-------------------|--------|
| Threat modeling | IEC 62443-3-2 | [Status] |
| Security risk assessment | IEC 62443-3-2 | [Status] |
| Security controls | IEC 62443-3-3 | [Status] |
| SBOM | IEC 62443-2-4, 62304 | [Status] |
| Vulnerability management | IEC 62443-2-3 | [Status] |
| Security testing | IEC 62443-4-1 | [Status] |

### 9.2 Regulatory References

- FDA Guidance: Content of Premarket Submissions for Management of Cybersecurity
  in Medical Devices (2023)
- IEC 62443-3-3: System security requirements and security levels
- IEC 62443-4-2: Technical security requirements for IACS components
- IEC 81001-5-1: Security for health software and health IT systems
- IEC 62304: Medical device software life cycle processes

---

## 10. Appendices

### Appendix A: Evidence Documentation

| Evidence ID | Description | Location | Date Collected |
|-------------|-------------|----------|----------------|
| E-001 | [Description] | [Location/Path] | [Date] |
| E-002 | [Description] | [Location/Path] | [Date] |

### Appendix B: Assessment Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| [Tool name] | [Version] | [Purpose] |

### Appendix C: Acronyms and Definitions

| Term | Definition |
|------|------------|
| FR | Foundational Requirement |
| IAC | Identification and Authentication Control |
| SL | Security Level |
| SOUP | Software of Unknown Provenance |
| SR | System Requirement |
| UC | Use Control |

### Appendix D: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | [Date] | [Author] | Initial release |

---

## Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Assessor | [Name] | __________ | [Date] |
| Reviewer | [Name] | __________ | [Date] |
| Approver | [Name] | __________ | [Date] |

---

*This template aligns with IEC 62443 standards and FDA cybersecurity guidance for
medical devices. Customize sections as needed for your specific system and
regulatory requirements.*
