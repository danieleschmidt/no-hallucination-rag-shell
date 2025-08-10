# Production Deployment Guide

## Overview
This guide covers deploying the No-Hallucination RAG System to production environments.

## Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   RAG System     │    │     Cache       │
│   (Ingress)     ├────┤   (3+ replicas)  ├────┤   (Redis)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                       ┌───────────────────┐
                       │    Monitoring     │
                       │ (Prometheus +     │
                       │    Grafana)       │
                       └───────────────────┘
```

## Prerequisites

### Infrastructure
- Kubernetes cluster (v1.25+)
- 16GB+ RAM per node
- 100GB+ storage
- Load balancer with SSL termination

### Software
- Docker (for building images)
- kubectl (configured for your cluster)
- Helm (optional, for easier management)

## Quick Start

### CI/CD Setup (Optional)
If you want to set up automated CI/CD with GitHub Actions:
1. Copy `deployment/templates/github-workflow-template.yml` to `.github/workflows/deploy.yml`
2. Ensure your GitHub repository has `workflows` permission enabled
3. Configure required secrets: `KUBECONFIG` for Kubernetes access

### Manual Deployment

1. **Build and Push Docker Image**
   ```bash
   docker build -t terragon/no-hallucination-rag:latest .
   docker push terragon/no-hallucination-rag:latest
   ```

2. **Deploy to Kubernetes**
   ```bash
   ./deployment/scripts/deploy.sh
   ```

3. **Verify Deployment**
   ```bash
   kubectl get pods -l app=rag-system
   curl -f https://your-domain.com/health
   ```

## Configuration

### Environment Variables
- `ENVIRONMENT`: Set to "production"
- `LOG_LEVEL`: Set to "INFO" or "WARNING"
- `CACHE_SIZE`: Number of cached items (default: 10000)
- `REDIS_URL`: Redis connection string
- `PROMETHEUS_ENABLED`: Enable metrics (default: true)

### Resource Limits
- **Memory**: 1GB limit, 512MB request
- **CPU**: 500m limit, 200m request
- **Storage**: 10GB persistent volume

### Auto-scaling
- **Min replicas**: 3
- **Max replicas**: 20
- **CPU threshold**: 70%
- **Memory threshold**: 80%

## Monitoring

### Metrics Endpoints
- `/metrics` - Prometheus metrics
- `/health` - Health check
- `/ready` - Readiness probe

### Key Metrics
- `http_requests_total` - Request count
- `http_request_duration_seconds` - Response times
- `cache_hits_total` - Cache hit rate
- `factuality_check_score` - Quality metrics

### Alerts
Configure alerts for:
- High error rate (>5%)
- Slow response times (>2s p95)
- Low cache hit rate (<80%)
- Pod restart loops

## Security

### Best Practices
- Non-root container user
- Read-only root filesystem
- Resource limits enforced
- Network policies applied
- Regular security scans

### TLS/SSL
- Certificate auto-renewal with cert-manager
- HTTPS enforcement
- Strong cipher suites only

## Backup & Recovery

### Data Backup
```bash
kubectl create job backup-$(date +%Y%m%d) \
    --from=cronjob/backup-job
```

### Disaster Recovery
1. Restore from backup
2. Redeploy application
3. Verify functionality
4. Update DNS if needed

## Troubleshooting

### Common Issues
1. **Pod stuck in Pending**: Check resource limits
2. **ImagePullBackOff**: Verify image exists and credentials
3. **CrashLoopBackOff**: Check logs with `kubectl logs`
4. **Service unavailable**: Check ingress and service configs

### Debug Commands
```bash
# Check pod status
kubectl get pods -l app=rag-system

# View logs
kubectl logs -l app=rag-system --tail=100

# Describe failing pod
kubectl describe pod <pod-name>

# Check resource usage
kubectl top pods -l app=rag-system
```

## Maintenance

### Regular Tasks
- Monitor resource usage
- Review security scans
- Update dependencies
- Performance testing
- Backup verification

### Updates
1. Test in staging environment
2. Deploy during maintenance window
3. Monitor for issues
4. Rollback if problems occur

## Support
For issues or questions:
- Check monitoring dashboards
- Review application logs
- Contact: ops@terragon.ai
