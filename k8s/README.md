# Kubernetes Deployment

Kubernetes manifests for deploying MedTech AI Security to a K8s cluster.

## Quick Start

```bash
# Create namespace and deploy all resources
kubectl apply -k k8s/

# Or apply individually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml  # Edit with real values first
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/deployment-threat-intel.yaml
kubectl apply -f k8s/deployment-sbom-analyzer.yaml
kubectl apply -f k8s/deployment-anomaly-detector.yaml
kubectl apply -f k8s/ingress.yaml
```

## Prerequisites

- Kubernetes 1.25+
- kubectl configured for your cluster
- Storage class with ReadWriteMany support (or adjust PVC specs)
- NGINX Ingress Controller (optional, for external access)
- cert-manager (optional, for TLS)

## Components

| Component | Type | Description |
|-----------|------|-------------|
| threat-intel-scanner | Deployment | Runs threat intelligence collection |
| nvd-scanner | CronJob | Daily NVD CVE scanning (6 AM UTC) |
| cisa-scanner | CronJob | Daily CISA advisory scanning (7 AM UTC) |
| sbom-analyzer | Deployment + Service | REST API for SBOM analysis |
| anomaly-detector | Deployment + Service | Real-time traffic anomaly detection |

## Configuration

### Secrets

Edit `secret.yaml` before deploying:

```yaml
stringData:
  NVD_API_KEY: "your-nvd-api-key"
  DEFECTDOJO_URL: "https://your-defectdojo.example.com"
  DEFECTDOJO_API_KEY: "your-defectdojo-api-key"
  CLAUDE_API_KEY: "your-claude-api-key"
```

For production, use external secrets management:
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- Sealed Secrets

### ConfigMap

Adjust `configmap.yaml` for environment-specific settings:

```yaml
data:
  NVD_DAYS_BACK: "30"      # CVEs from last N days
  LOG_LEVEL: "INFO"         # DEBUG, INFO, WARNING, ERROR
  ANOMALY_THRESHOLD: "0.5"  # Detection sensitivity
```

### Storage

Default PVCs request:
- `medtech-security-data`: 10Gi (scan results, reports)
- `medtech-security-models`: 5Gi (trained ML models)

Adjust `pvc.yaml` for your storage requirements.

## Scaling

### Horizontal Pod Autoscaler

SBOM Analyzer and Anomaly Detector include HPA configurations:

```yaml
# SBOM Analyzer: 2-10 replicas based on CPU/memory
# Anomaly Detector: 2-8 replicas based on CPU
```

### Manual Scaling

```bash
kubectl scale deployment sbom-analyzer --replicas=5 -n medtech-security
```

## Monitoring

### Prometheus Metrics

Services expose metrics at `/metrics`:

```bash
# Port-forward to access metrics
kubectl port-forward svc/sbom-analyzer 8080:80 -n medtech-security
curl http://localhost:8080/metrics
```

### Health Checks

```bash
# Check pod status
kubectl get pods -n medtech-security

# Check service endpoints
kubectl get endpoints -n medtech-security

# View logs
kubectl logs -l app.kubernetes.io/name=sbom-analyzer -n medtech-security
```

## Security

### Network Policy

Default policy restricts traffic:
- Ingress: Same namespace + ingress-nginx namespace only
- Egress: DNS (UDP 53) + HTTPS (TCP 443) only

### Pod Security

All pods run as:
- Non-root user (UID 1000)
- Read-only root filesystem
- No privilege escalation
- All capabilities dropped

### RBAC

Service account has minimal permissions:
- Read configmaps/secrets in namespace
- Read pods for health checks
- Create events for logging

## Troubleshooting

### Pod not starting

```bash
kubectl describe pod <pod-name> -n medtech-security
kubectl logs <pod-name> -n medtech-security
```

### CronJob not running

```bash
kubectl get cronjobs -n medtech-security
kubectl get jobs -n medtech-security
kubectl logs job/<job-name> -n medtech-security
```

### Storage issues

```bash
kubectl get pvc -n medtech-security
kubectl describe pvc medtech-security-data -n medtech-security
```

## Cleanup

```bash
# Delete all resources
kubectl delete -k k8s/

# Or delete namespace (removes everything)
kubectl delete namespace medtech-security
```

## Production Checklist

- [ ] Update secret.yaml with real credentials
- [ ] Configure external secrets management
- [ ] Adjust resource limits based on load testing
- [ ] Configure TLS certificates (cert-manager)
- [ ] Set up Prometheus/Grafana monitoring
- [ ] Configure alerting rules
- [ ] Test disaster recovery procedures
- [ ] Document backup/restore process
