# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it through GitHub's Private Vulnerability Reporting (PVR) feature:

1. Go to the [Security tab](https://github.com/BioStructBenchmark/BioStructBenchmark/security) of this repository
2. Click "Report a vulnerability"
3. Provide a detailed description of the vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

## What to Include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Any suggested fixes (optional)

## Response Timeline

- We will acknowledge receipt within 48 hours
- We will provide an initial assessment within 7 days
- We will work with you to understand and resolve the issue

## Security Measures

This project uses several security tools as part of our development process:

- **Bandit**: Static analysis for common security issues in Python code
- **pip-audit**: Checks dependencies for known vulnerabilities
- **zizmor**: Security scanning for GitHub Actions workflows
- **Pre-commit hooks**: Automated security checks before each commit

## Dependency Updates

We regularly update dependencies to address known vulnerabilities. Security updates are prioritized and released as soon as possible.
