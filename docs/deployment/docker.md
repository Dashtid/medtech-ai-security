# Docker Deployment

Guide for deploying MedTech AI Security using Docker containers.

## Prerequisites

- Docker 20.10 or later
- Docker Compose 2.0 or later (optional)
- 4GB RAM minimum (8GB recommended for ML models)

## Quick Start

### Pull and Run

```bash
# Pull the latest image
docker pull ghcr.io/dashtid/medtech-ai-security:latest

# Run the container
docker run -d \
  --name medsec \
  -p 8000:8000 \
  -v medsec-data:/app/data \
  -e ANTHROPIC_API_KEY=your-key \
  ghcr.io/dashtid/medtech-ai-security:latest
```

### Build Locally

```bash
# Clone the repository
git clone https://github.com/Dashtid/medtech-ai-security.git
cd medtech-ai-security

# Build the image
docker build -t medsec:local .

# Run locally built image
docker run -d --name medsec -p 8000:8000 medsec:local
```

## Docker Compose

For full deployment with all services:

```yaml
# docker-compose.yml
version: '3.8'

services:
  medsec:
    image: ghcr.io/dashtid/medtech-ai-security:latest
    container_name: medsec
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - NVD_API_KEY=${NVD_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - medsec-data:/app/data
      - medsec-models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: medsec-redis
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  medsec-data:
  medsec-models:
  redis-data:
```

Start the stack:

```bash
# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=your-anthropic-key
NVD_API_KEY=your-nvd-key
EOF

# Start services
docker-compose up -d

# View logs
docker-compose logs -f medsec

# Stop services
docker-compose down
```

## Image Tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest stable release |
| `v0.2.0` | Specific version |
| `main` | Latest from main branch (unstable) |
| `sha-abc123` | Specific commit |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | No | - | API key for Claude AI enrichment |
| `NVD_API_KEY` | No | - | NVD API key for higher rate limits |
| `LOG_LEVEL` | No | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `API_HOST` | No | 0.0.0.0 | API server bind address |
| `API_PORT` | No | 8000 | API server port |
| `WORKERS` | No | 4 | Number of Uvicorn workers |

## Volume Mounts

| Mount Point | Purpose |
|-------------|---------|
| `/app/data` | Persistent data (CVE cache, analysis results) |
| `/app/models` | ML model files |
| `/app/logs` | Application logs |

## Resource Limits

Recommended resource constraints:

```yaml
services:
  medsec:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## GPU Support

For faster ML inference with GPU:

```yaml
services:
  medsec:
    image: ghcr.io/dashtid/medtech-ai-security:latest-gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Ensure NVIDIA Container Toolkit is installed:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Health Checks

The container includes a health check endpoint:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' medsec

# Manual health check
curl http://localhost:8000/api/v1/health
```

## Logging

View container logs:

```bash
# Follow logs
docker logs -f medsec

# Last 100 lines
docker logs --tail 100 medsec

# With timestamps
docker logs -t medsec
```

Configure log rotation in Docker:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

## Security

### Running as Non-Root

The container runs as a non-root user by default:

```dockerfile
USER medsec
```

### Read-Only Filesystem

For enhanced security:

```yaml
services:
  medsec:
    read_only: true
    tmpfs:
      - /tmp
      - /app/cache
    volumes:
      - medsec-data:/app/data
```

### Network Isolation

```yaml
services:
  medsec:
    networks:
      - frontend
      - backend

networks:
  frontend:
  backend:
    internal: true
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs medsec

# Verify environment variables
docker inspect medsec | jq '.[0].Config.Env'

# Check resource availability
docker stats --no-stream
```

### Out of Memory

```bash
# Increase memory limit
docker update --memory 8g medsec

# Or in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

### Permission Issues

```bash
# Fix volume permissions
docker run --rm -v medsec-data:/data alpine chown -R 1000:1000 /data
```

## Upgrading

```bash
# Pull new image
docker pull ghcr.io/dashtid/medtech-ai-security:latest

# Stop and remove old container
docker stop medsec && docker rm medsec

# Start with new image
docker run -d --name medsec \
  -p 8000:8000 \
  -v medsec-data:/app/data \
  ghcr.io/dashtid/medtech-ai-security:latest
```

Or with Docker Compose:

```bash
docker-compose pull
docker-compose up -d
```
