# Deployment Guide

This guide covers deploying the No-Hallucination RAG Shell to production environments.

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+ (for local development)
- 4GB+ RAM recommended
- 10GB+ disk space

### Production Deployment

```bash
# Clone repository
git clone https://github.com/danieleschmidt/no-hallucination-rag-shell.git
cd no-hallucination-rag-shell

# Run automated deployment
./scripts/deploy.sh
```

The deployment script will:
1. ✅ Validate prerequisites
2. 🔒 Run security checks
3. 🧪 Execute test suite
4. 🐳 Build Docker image
5. 🚀 Deploy with Docker Compose
6. 🏥 Perform health checks
7. ⚡ Run performance tests

## Configuration

### Environment Files

- `configs/production.yaml` - Production settings (strict governance, high security)
- `configs/development.yaml` - Development settings (relaxed for testing)

### Key Configuration Options

```yaml
rag_system:
  factuality_threshold: 0.95    # Minimum factuality score
  governance_mode: "strict"     # Governance compliance level
  max_sources: 15              # Maximum sources per query
  
security:
  rate_limits:
    requests_per_minute: 30    # Rate limiting
    requests_per_hour: 500
    
caching:
  query_ttl: 1800             # Cache TTL in seconds
  max_memory_mb: 512          # Max cache memory
```

## Services

### Core Service

- **No-Hallucination RAG Shell**: Main application
  - Port: 8000
  - Health check: `/health` endpoint
  - Metrics: Prometheus format

### Optional Services

- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Monitoring dashboards (port 3000, admin/admin)
- **Redis**: Advanced caching backend (port 6379)

## Monitoring

### Health Checks

```bash
# Application health
docker-compose exec no-hallucination-rag python -c "
from no_hallucination_rag import FactualRAG
rag = FactualRAG()
print(rag.get_system_health())
"

# Container health
docker-compose ps
```

### Logs

```bash
# Application logs
docker-compose logs -f no-hallucination-rag

# All services
docker-compose logs -f
```

### Metrics

Access Prometheus at http://localhost:9090 for:
- Query performance metrics
- Factuality score distributions  
- Cache hit rates
- Error rates
- Resource usage

## Security

### Default Security Features

- ✅ Input validation and sanitization
- ✅ Rate limiting per client
- ✅ Malicious pattern detection
- ✅ IP filtering capabilities
- ✅ Governance compliance checking
- ✅ Audit logging

### Security Configuration

```yaml
security:
  enable_ip_filtering: true
  rate_limits:
    requests_per_minute: 30
    requests_per_hour: 500
    requests_per_day: 5000
```

### API Keys (Optional)

```python
from no_hallucination_rag.security.security_manager import SecurityManager

security = SecurityManager()
api_key, key_id = security.create_api_key(
    user_id="production_user",
    expires_days=30
)
print(f"API Key: {api_key}")
```

## Performance Tuning

### Auto-Optimization

The system automatically optimizes performance based on usage patterns:

```yaml
optimization:
  auto_optimize: true
  monitoring_window_minutes: 60
  optimization_interval_minutes: 30
```

### Manual Optimization

```python
from no_hallucination_rag import FactualRAG

rag = FactualRAG()

# Force optimization
result = rag.optimize_performance()
print(result)

# Get performance stats
stats = rag.get_performance_stats()
print(stats)
```

### Scaling Recommendations

| Load Level | Configuration |
|------------|---------------|
| Light (<100 qps) | Default settings |
| Medium (100-500 qps) | Increase `max_concurrent_queries: 50` |
| Heavy (500+ qps) | Scale horizontally, use Redis caching |

## Troubleshooting

### Common Issues

#### 1. High Memory Usage

```bash
# Check cache usage
docker-compose exec no-hallucination-rag python -c "
from no_hallucination_rag import FactualRAG
rag = FactualRAG()
stats = rag.get_performance_stats()
print('Cache usage:', stats['cache']['memory_usage'])
"

# Clear caches
docker-compose exec no-hallucination-rag python -c "
from no_hallucination_rag import FactualRAG
rag = FactualRAG()
rag.invalidate_caches()
"
```

#### 2. Slow Response Times

```bash
# Check performance metrics
docker-compose exec no-hallucination-rag python -c "
from no_hallucination_rag import FactualRAG
rag = FactualRAG()
health = rag.get_system_health()
print('Avg response time:', health.get('metrics', {}).get('avg_response_time'))
"

# Force optimization
docker-compose exec no-hallucination-rag python -c "
from no_hallucination_rag import FactualRAG
rag = FactualRAG()
rag.optimize_performance()
"
```

#### 3. Rate Limiting Issues

```bash
# Check security stats
docker-compose exec no-hallucination-rag python -c "
from no_hallucination_rag import FactualRAG
rag = FactualRAG()
health = rag.get_system_health()
print('Security stats:', health.get('security'))
"
```

### Log Analysis

```bash
# Error analysis
docker-compose logs no-hallucination-rag | grep ERROR

# Performance analysis  
docker-compose logs no-hallucination-rag | grep "response_time"

# Security events
docker-compose logs no-hallucination-rag | grep "Security event"
```

## Backup and Recovery

### Data Backup

```bash
# Backup data volumes
docker run --rm -v no-hallucination-rag_rag_data:/data \
  -v $(pwd)/backup:/backup alpine \
  tar czf /backup/rag_data_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
```

### Configuration Backup

```bash
# Backup configurations
tar czf backup/configs_$(date +%Y%m%d_%H%M%S).tar.gz configs/
```

## Updates

### Rolling Update

```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
docker-compose build no-hallucination-rag
docker-compose up -d no-hallucination-rag

# Verify deployment
./scripts/deploy.sh
```

### Zero-Downtime Update

For production environments requiring zero downtime:

1. Deploy to staging environment
2. Run full test suite
3. Use blue-green deployment strategy
4. Gradually shift traffic

## Support

### Health Monitoring

Set up alerts for:
- High error rates (>5%)
- Slow response times (>10s)
- Low factuality scores (<0.9)
- High memory usage (>80%)
- Security violations

### Contact

- GitHub Issues: https://github.com/danieleschmidt/no-hallucination-rag-shell/issues
- Documentation: README.md
- Security Issues: security@terragonlabs.com