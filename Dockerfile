# MedTech AI Security - Production Docker Image
# Multi-stage build for optimized image size
#
# Build: docker build -t medtech-ai-security:latest .
# Run:   docker run -it medtech-ai-security:latest medsec-sbom --help

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.14-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /build

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache .

# =============================================================================
# Stage 2: Runtime - Production image
# =============================================================================
FROM python:3.14-slim as runtime

# Labels
LABEL org.opencontainers.image.title="MedTech AI Security"
LABEL org.opencontainers.image.description="AI-powered medical device cybersecurity platform"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.vendor="David Dashti"
LABEL org.opencontainers.image.source="https://github.com/Dashtid/medtech-ai-security"
LABEL org.opencontainers.image.licenses="MIT"
LABEL maintainer="David Dashti <david.dashti@hermesmedical.com>"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 medtech \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home medtech

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories with proper ownership
RUN mkdir -p /app/data /app/models /app/logs /app/reports \
    && chown -R medtech:medtech /app

# Copy application code
COPY --chown=medtech:medtech src/ ./src/
COPY --chown=medtech:medtech scripts/ ./scripts/

# Switch to non-root user
USER medtech

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    TF_CPP_MIN_LOG_LEVEL=2 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import medtech_ai_security; print('OK')" || exit 1

# Volume mounts for data persistence
VOLUME ["/app/data", "/app/models", "/app/reports"]

# Expose ports for services
EXPOSE 8080 8081 50051

# Default command - show available CLI tools
CMD ["python", "-c", "print('MedTech AI Security CLI Tools:\\n  medsec-nvd\\n  medsec-cisa\\n  medsec-enrich\\n  medsec-risk\\n  medsec-traffic-gen\\n  medsec-anomaly\\n  medsec-adversarial\\n  medsec-sbom\\n  medsec-defectdojo\\n\\nRun with --help for usage.')"]

# =============================================================================
# Stage 3: API Server - FastAPI-based services
# =============================================================================
FROM runtime as api-server

# Install additional API dependencies
USER root
RUN pip install --no-cache-dir fastapi uvicorn[standard] prometheus-client
USER medtech

# Copy API modules
COPY --chown=medtech:medtech src/medtech_ai_security/api/ ./src/medtech_ai_security/api/

# Override health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose API port
EXPOSE 8080

# Run API server
CMD ["uvicorn", "medtech_ai_security.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# Stage 4: Development - Additional dev tools
# =============================================================================
FROM runtime as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install dev Python packages
RUN pip install --no-cache-dir pytest pytest-cov black ruff mypy ipython

USER medtech

# Disable health check in development
HEALTHCHECK NONE

# Development shell
CMD ["bash"]
