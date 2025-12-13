# Production Deployment

Best practices and considerations for production deployments.

## Pre-Deployment Checklist

- [ ] Security audit completed
- [ ] SSL/TLS certificates configured
- [ ] API keys stored in secrets manager
- [ ] Logging and monitoring configured
- [ ] Backup strategy implemented
- [ ] Rate limiting configured
- [ ] Network policies defined
- [ ] Resource limits set
- [ ] Health checks configured
- [ ] Disaster recovery plan documented

## Architecture Overview

```
                                    +------------------+
                                    |   Load Balancer  |
                                    |   (TLS Term.)    |
                                    +--------+---------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
           +--------v--------+     +---------v-------+     +---------v-------+
           |  MedSec API #1  |     |  MedSec API #2  |     |  MedSec API #3  |
           |   (Container)   |     |   (Container)   |     |   (Container)   |
           +--------+--------+     +---------+-------+     +---------+-------+
                    |                        |                        |
                    +------------------------+------------------------+
                                             |
                              +--------------+-------------+
                              |                            |
                    +---------v---------+       +----------v---------+
                    |      Redis        |       |   Object Storage   |
                    |  (Cache/Queue)    |       |   (ML Models)      |
                    +-------------------+       +--------------------+
```

## Infrastructure Requirements

### Compute

| Component | Min CPU | Min Memory | Recommended |
|-----------|---------|------------|-------------|
| API Server | 2 cores | 4 GB | 4 cores, 8 GB |
| ML Worker | 4 cores | 8 GB | 8 cores, 16 GB |
| Redis | 1 core | 1 GB | 2 cores, 2 GB |

### Storage

| Data Type | Size Estimate | Growth Rate |
|-----------|---------------|-------------|
| CVE Database | 2 GB | 100 MB/month |
| ML Models | 500 MB | Per release |
| Logs | Variable | 1 GB/day |
| Analysis Results | Variable | 500 MB/day |

## Security Configuration

### TLS/SSL

Always use TLS 1.3 in production:

```yaml
# NGINX configuration
ssl_protocols TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers on;
ssl_session_timeout 1d;
ssl_session_cache shared:SSL:50m;
```

### API Security Headers

```yaml
# Security headers
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### Rate Limiting

Configure per-client rate limits:

```yaml
# NGINX rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req zone=api burst=20 nodelay;
```

### Authentication

For production API access:

1. Use API keys with expiration
2. Implement IP allowlisting
3. Enable request signing for sensitive operations
4. Rotate credentials regularly

## High Availability

### Multi-Region Deployment

```
Region: US-East                    Region: EU-West
+-------------------+              +-------------------+
|  Load Balancer    |<------------>|  Load Balancer    |
+--------+----------+    DNS       +--------+----------+
         |                                  |
+--------v----------+              +--------v----------+
|  MedSec Cluster   |              |  MedSec Cluster   |
|  (3 replicas)     |              |  (3 replicas)     |
+--------+----------+              +--------+----------+
         |                                  |
+--------v----------+              +--------v----------+
|  Redis Sentinel   |<------------>|  Redis Sentinel   |
+-------------------+   Repl.      +-------------------+
```

### Database Considerations

For persistent data:

- Use managed database services (AWS RDS, GCP Cloud SQL)
- Configure automatic backups
- Enable point-in-time recovery
- Set up read replicas for scaling

## Monitoring and Alerting

### Metrics to Monitor

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| API Latency (p99) | > 500ms | > 2s | Scale or optimize |
| Error Rate | > 1% | > 5% | Investigate logs |
| CPU Usage | > 70% | > 90% | Scale horizontally |
| Memory Usage | > 75% | > 90% | Increase limits |
| Disk Usage | > 70% | > 85% | Expand storage |

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: medsec
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected

      - alert: APILatencyHigh
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: API latency above threshold
```

### Log Aggregation

Configure centralized logging:

```yaml
# Fluent Bit configuration
[OUTPUT]
    Name  es
    Match *
    Host  elasticsearch.logging.svc
    Port  9200
    Index medsec-logs
    Type  _doc
```

## Backup and Recovery

### Backup Strategy

| Data | Frequency | Retention | Method |
|------|-----------|-----------|--------|
| CVE Database | Daily | 30 days | pg_dump |
| ML Models | Per release | Indefinite | Object storage |
| Configuration | On change | 90 days | Git |
| Logs | Real-time | 30 days | Log aggregation |

### Backup Script

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/medsec"

# Database backup
pg_dump medsec | gzip > "${BACKUP_DIR}/db_${DATE}.sql.gz"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/db_${DATE}.sql.gz" s3://medsec-backups/

# Cleanup old backups
find "${BACKUP_DIR}" -name "*.gz" -mtime +30 -delete
```

### Recovery Procedures

1. **Database Recovery:**
   ```bash
   gunzip -c backup.sql.gz | psql medsec
   ```

2. **Configuration Recovery:**
   ```bash
   git checkout v0.2.0
   kubectl apply -k k8s/overlays/production/
   ```

## Performance Optimization

### Caching Strategy

```python
# Redis caching for CVE lookups
CACHE_CONFIG = {
    "cvss_scores": {
        "ttl": 3600,  # 1 hour
        "prefix": "cvss:"
    },
    "enrichment_results": {
        "ttl": 86400,  # 24 hours
        "prefix": "enrich:"
    },
    "sbom_analysis": {
        "ttl": 7200,  # 2 hours
        "prefix": "sbom:"
    }
}
```

### Connection Pooling

```python
# Database connection pool
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 1800
}
```

## Compliance Considerations

### HIPAA

If processing PHI:

- Enable encryption at rest
- Configure audit logging
- Implement access controls
- Sign BAA with cloud provider

### FDA 21 CFR Part 11

For regulated environments:

- Electronic signatures
- Audit trails
- Data integrity controls
- Validation documentation

### SOC 2

Controls to implement:

- Access management
- Change management
- Incident response
- Business continuity

## Disaster Recovery

### RTO and RPO

| Scenario | RTO | RPO |
|----------|-----|-----|
| Single Pod Failure | < 1 min | 0 |
| Node Failure | < 5 min | 0 |
| Availability Zone | < 15 min | < 1 min |
| Region Failure | < 1 hour | < 5 min |

### Failover Procedures

1. **Automatic Failover:**
   - Kubernetes handles pod rescheduling
   - Load balancer health checks remove unhealthy backends

2. **Manual Failover:**
   ```bash
   # Switch DNS to backup region
   aws route53 change-resource-record-sets \
     --hosted-zone-id ZONE_ID \
     --change-batch file://failover.json
   ```

## Maintenance Windows

### Update Procedures

1. Deploy to staging environment
2. Run integration tests
3. Perform canary deployment (10% traffic)
4. Monitor for 30 minutes
5. Roll out to remaining pods
6. Keep previous version for rollback

### Zero-Downtime Deployment

```yaml
# Rolling update strategy
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

## Cost Optimization

### Resource Right-Sizing

- Monitor actual resource usage
- Adjust requests/limits quarterly
- Use spot instances for non-critical workloads

### Auto-Scaling

```yaml
# Scale down during off-hours
spec:
  minReplicas: 1
  maxReplicas: 10
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
```
