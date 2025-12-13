# Kubernetes Deployment

Guide for deploying MedTech AI Security on Kubernetes clusters.

## Prerequisites

- Kubernetes 1.25 or later
- kubectl configured
- Helm 3.0 or later (optional)
- Container registry access (GHCR)

## Quick Start with Kustomize

```bash
# Clone the repository
git clone https://github.com/Dashtid/medtech-ai-security.git
cd medtech-ai-security

# Deploy to default namespace
kubectl apply -k k8s/overlays/production/

# Check deployment status
kubectl get pods -l app=medsec
```

## Directory Structure

```
k8s/
  base/
    deployment.yaml
    service.yaml
    configmap.yaml
    kustomization.yaml
  overlays/
    development/
      kustomization.yaml
    production/
      kustomization.yaml
      ingress.yaml
```

## Base Resources

### Deployment

```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medsec
  labels:
    app: medsec
spec:
  replicas: 2
  selector:
    matchLabels:
      app: medsec
  template:
    metadata:
      labels:
        app: medsec
    spec:
      containers:
        - name: medsec
          image: ghcr.io/dashtid/medtech-ai-security:latest
          ports:
            - containerPort: 8000
          env:
            - name: LOG_LEVEL
              valueFrom:
                configMapKeyRef:
                  name: medsec-config
                  key: log-level
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: medsec-secrets
                  key: anthropic-api-key
          resources:
            requests:
              memory: "2Gi"
              cpu: "500m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /api/v1/health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /api/v1/health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: data
              mountPath: /app/data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: medsec-data
```

### Service

```yaml
# k8s/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: medsec
spec:
  selector:
    app: medsec
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

### ConfigMap

```yaml
# k8s/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medsec-config
data:
  log-level: "INFO"
  workers: "4"
```

## Secrets Management

Create secrets for API keys:

```bash
# Create secret from literals
kubectl create secret generic medsec-secrets \
  --from-literal=anthropic-api-key=sk-ant-... \
  --from-literal=nvd-api-key=...

# Or from file
kubectl create secret generic medsec-secrets \
  --from-file=anthropic-api-key=./anthropic-key.txt
```

For production, use a secrets manager:

```yaml
# External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: medsec-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: medsec-secrets
  data:
    - secretKey: anthropic-api-key
      remoteRef:
        key: medsec/api-keys
        property: anthropic
```

## Ingress Configuration

### NGINX Ingress

```yaml
# k8s/overlays/production/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: medsec
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - medsec.example.com
      secretName: medsec-tls
  rules:
    - host: medsec.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: medsec
                port:
                  number: 8000
```

### Traefik Ingress

```yaml
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: medsec
spec:
  entryPoints:
    - websecure
  routes:
    - match: Host(`medsec.example.com`)
      kind: Rule
      services:
        - name: medsec
          port: 8000
  tls:
    certResolver: letsencrypt
```

## Persistent Storage

```yaml
# k8s/base/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: medsec-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

## Horizontal Pod Autoscaler

```yaml
# k8s/overlays/production/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: medsec
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medsec
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

## Network Policies

```yaml
# k8s/base/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: medsec
spec:
  podSelector:
    matchLabels:
      app: medsec
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8000
  egress:
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - port: 443
        - port: 80
```

## Pod Disruption Budget

```yaml
# k8s/overlays/production/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: medsec
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: medsec
```

## Helm Chart

For Helm-based deployment:

```bash
# Add repository (if published)
helm repo add medsec https://dashtid.github.io/medtech-ai-security/charts

# Install
helm install medsec medsec/medtech-ai-security \
  --set apiKeys.anthropic=sk-ant-... \
  --set ingress.enabled=true \
  --set ingress.host=medsec.example.com

# Upgrade
helm upgrade medsec medsec/medtech-ai-security \
  --set image.tag=v0.2.1
```

## Monitoring

### ServiceMonitor (Prometheus)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: medsec
  labels:
    app: medsec
spec:
  selector:
    matchLabels:
      app: medsec
  endpoints:
    - port: http
      path: /api/v1/metrics
      interval: 30s
```

### Grafana Dashboard

Import the provided dashboard:

```bash
kubectl apply -f k8s/monitoring/grafana-dashboard.yaml
```

## Troubleshooting

### Check Pod Status

```bash
# Get pods
kubectl get pods -l app=medsec

# Describe pod
kubectl describe pod medsec-xxx

# View logs
kubectl logs -f medsec-xxx

# Execute into pod
kubectl exec -it medsec-xxx -- /bin/sh
```

### Common Issues

**ImagePullBackOff:**

```bash
# Check secret for registry
kubectl get secret regcred -o yaml

# Create registry secret
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=username \
  --docker-password=token
```

**CrashLoopBackOff:**

```bash
# Check logs
kubectl logs medsec-xxx --previous

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

## Cleanup

```bash
# Delete all resources
kubectl delete -k k8s/overlays/production/

# Or specific resources
kubectl delete deployment medsec
kubectl delete service medsec
kubectl delete pvc medsec-data
```
