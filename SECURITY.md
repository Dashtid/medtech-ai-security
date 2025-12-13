# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of MedTech AI Security seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

- **Email**: david.dashti@hermesmedical.com
- **Subject**: [SECURITY] MedTech AI Security Vulnerability Report

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., SQL injection, XSS, buffer overflow, etc.)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 7 days with an assessment of the vulnerability
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### Safe Harbor

We support responsible disclosure. If you make a good faith effort to comply with this policy during your security research, we will:

- Consider your research to be authorized
- Work with you to understand and resolve the issue quickly
- Not pursue legal action related to your research
- Credit you in our security acknowledgments (if desired)

### Scope

This policy applies to the following:

- Source code in this repository
- Dependencies managed by this project
- Documentation and configuration files

The following are **out of scope**:

- Third-party services or dependencies (report to the respective maintainers)
- Social engineering attacks
- Physical attacks
- Denial of service attacks

## Security Best Practices

When using MedTech AI Security in production:

1. **Environment Variables**: Never commit API keys, tokens, or credentials
2. **Network Security**: Deploy behind a firewall with proper access controls
3. **Data Protection**: Ensure HIPAA/GDPR compliance when processing medical data
4. **Updates**: Keep dependencies updated using Dependabot alerts
5. **Monitoring**: Enable security scanning in your CI/CD pipeline

## Security Features

This project includes several security measures:

- **Static Analysis**: Bandit security linting in CI pipeline
- **Dependency Scanning**: Safety and Dependabot for vulnerability detection
- **Type Checking**: Mypy for catching potential runtime errors
- **Code Quality**: Ruff and Black for consistent, secure code patterns

## Acknowledgments

We would like to thank the following security researchers for their responsible disclosure:

*No vulnerabilities have been reported yet.*

---

Thank you for helping keep MedTech AI Security and its users safe!
