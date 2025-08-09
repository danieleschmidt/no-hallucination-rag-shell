# Terragon RAG System - Production Deployment Guide

## Overview

This guide covers deploying the Terragon quantum-enhanced RAG system to production environments with high availability, monitoring, and security.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  Terragon RAG   │────│     Redis       │
│   (nginx/ALB)   │    │   Application   │    │     Cache       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│   Prometheus    │──────────────┘
                        │   Monitoring    │
                        └─────────────────┘
```

## Quick Start

### Docker Compose Deployment

1. **Clone and prepare**:
```bash
git clone https://github.com/terragon-labs/rag-system.git
cd rag-system
```

2. **Start services**:
```bash
docker-compose up -d
```

3. **Verify deployment**:
```bash
curl http://localhost:8080/health
```

### Kubernetes Deployment

1. **Apply configurations**:
```bash
kubectl apply -f deploy/kubernetes.yaml
```

2. **Check status**:
```bash
kubectl get pods -n terragon
kubectl logs -f deployment/terragon-rag -n terragon
```

## Environment Configuration

### Required Environment Variables

```bash
# Core Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here

# Cache Configuration
CACHE_SIZE_L1=1000
CACHE_SIZE_L2=10000
CACHE_SIZE_L3=100000
REDIS_URL=redis://redis:6379

# Performance Tuning
MAX_WORKERS=4
QUANTUM_COHERENCE_TIME=300
CIRCUIT_BREAKER_THRESHOLD=10
AUTO_SCALE_CPU_THRESHOLD=70

# Security
JWT_SECRET=your-jwt-secret
API_KEY_REQUIRED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Monitoring
METRICS_ENABLED=true
TRACING_ENABLED=true
PROMETHEUS_PORT=8081
```

### Optional Environment Variables

```bash
# Database (if using persistent storage)
DATABASE_URL=postgresql://user:pass@db:5432/terragon

# External Services
ML_MODEL_ENDPOINT=https://api.example.com/models
VECTOR_DB_URL=https://pinecone.example.com

# Cloud Storage
AWS_S3_BUCKET=terragon-cache
GOOGLE_CLOUD_PROJECT=terragon-prod
```

## Security Configuration

### 1. Network Security

- Use HTTPS in production (included in Kubernetes config)
- Configure firewall rules to restrict access
- Use private networks for internal communication

### 2. Authentication & Authorization

```python
# Environment variables for security
JWT_SECRET=your-256-bit-secret-key
API_KEY_REQUIRED=true
ALLOWED_ORIGINS=https://your-domain.com
IP_WHITELIST=10.0.0.0/8,192.168.0.0/16
```

### 3. Data Protection

- Enable encryption at rest for persistent volumes
- Use TLS for all inter-service communication
- Implement proper secret management (Vault, K8s Secrets)

## Performance Optimization

### 1. Cache Configuration

```yaml
# Optimal cache settings for production
CACHE_SIZE_L1: 2000      # Hot cache - SSD storage
CACHE_SIZE_L2: 20000     # Warm cache - fast disk
CACHE_SIZE_L3: 200000    # Cold cache - standard disk
CACHE_EVICTION_POLICY: adaptive
```

### 2. Auto-scaling Rules

```yaml
# HPA configuration
minReplicas: 3
maxReplicas: 20
targetCPUUtilization: 70%
targetMemoryUtilization: 80%
```

### 3. Resource Limits

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi" 
    cpu: "2000m"
```

## Monitoring & Observability

### 1. Metrics Collection

Key metrics monitored:

- **Performance**: Response time, throughput, cache hit rates
- **System**: CPU, memory, disk usage
- **Business**: Task completion rates, quantum coherence levels
- **Errors**: Error rates, circuit breaker states

### 2. Alerting Rules

```yaml
# Critical alerts
- High error rate (>5% for 5 minutes)
- High response time (>2s for 5 minutes)
- Low cache hit rate (<60% for 10 minutes)
- Circuit breaker open (immediate)
- High memory usage (>90% for 5 minutes)
```

### 3. Dashboard Setup

Access Prometheus at: `http://localhost:9090`

Key dashboard panels:
- Request rate and latency
- Cache performance by level (L1/L2/L3)
- Quantum task processing metrics
- Auto-scaling events
- Error rates by type

## Backup & Recovery

### 1. Data Backup

```bash
# Daily cache backup (for warm start)
docker exec terragon-redis redis-cli SAVE
docker cp terragon-redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d).rdb

# Configuration backup
kubectl get configmaps -n terragon -o yaml > backups/configmaps-$(date +%Y%m%d).yaml
```

### 2. Disaster Recovery

1. **Application Recovery**:
```bash
# Redeploy from backup
kubectl apply -f backups/kubernetes-backup.yaml
```

2. **Cache Recovery**:
```bash
# Restore Redis data
docker cp backups/redis-latest.rdb terragon-redis:/data/dump.rdb
docker restart terragon-redis
```

## Maintenance

### 1. Rolling Updates

```bash
# Update application
docker build -t terragon/rag:v1.1.0 .
kubectl set image deployment/terragon-rag terragon-rag=terragon/rag:v1.1.0 -n terragon
kubectl rollout status deployment/terragon-rag -n terragon
```

### 2. Health Checks

Regular health check endpoints:
- `/health` - Overall system health
- `/ready` - Readiness for traffic
- `/metrics` - Prometheus metrics
- `/info` - System information

### 3. Log Management

```bash
# View application logs
kubectl logs -f deployment/terragon-rag -n terragon

# View Redis logs  
kubectl logs -f deployment/terragon-redis -n terragon

# Aggregate logs (if using log aggregation)
stern terragon-rag -n terragon
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**:
```bash
# Check cache sizes and adjust
kubectl set env deployment/terragon-rag CACHE_SIZE_L3=50000 -n terragon
```

2. **Circuit Breaker Open**:
```bash
# Check error logs and upstream services
kubectl logs deployment/terragon-rag -n terragon | grep "circuit_breaker"
```

3. **Cache Miss Rate High**:
```bash
# Monitor cache performance
curl http://localhost:8080/metrics | grep cache_hit_rate
```

4. **Auto-scaling Issues**:
```bash
# Check HPA status
kubectl get hpa -n terragon
kubectl describe hpa terragon-rag-hpa -n terragon
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
kubectl set env deployment/terragon-rag LOG_LEVEL=DEBUG -n terragon
```

## Performance Benchmarking

### Load Testing

```bash
# Install k6 for load testing
brew install k6  # or apt-get install k6

# Run load test
k6 run --vus 50 --duration 5m scripts/load-test.js
```

### Benchmarks

Expected performance (3-replica setup):
- **Throughput**: 1000+ requests/second
- **Latency P95**: <500ms
- **Cache Hit Rate**: >80%
- **Memory Usage**: <2GB per pod
- **CPU Usage**: <70% per pod

## Security Checklist

- [ ] HTTPS enabled with valid certificates
- [ ] API keys configured and rotated regularly
- [ ] Network policies restrict pod-to-pod communication
- [ ] Secrets stored in Kubernetes secrets (not env vars)
- [ ] Container runs as non-root user
- [ ] Resource limits prevent resource exhaustion
- [ ] Security scanning enabled in CI/CD
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested

## Support

For production support:
- GitHub Issues: https://github.com/terragon-labs/rag-system/issues
- Documentation: https://docs.terragon-labs.com
- Enterprise Support: enterprise@terragon-labs.com

---

Generated by Terragon SDLC v4.0 - Autonomous Production Deployment System