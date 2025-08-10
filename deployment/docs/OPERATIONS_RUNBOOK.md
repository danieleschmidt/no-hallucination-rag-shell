# Operations Runbook

## Emergency Contacts
- On-call Engineer: +1-XXX-XXX-XXXX
- DevOps Team: devops@terragon.ai
- Engineering Manager: eng-mgr@terragon.ai

## Service Level Objectives (SLOs)
- Availability: 99.9% uptime
- Response Time: <500ms p95
- Error Rate: <0.1%
- Cache Hit Rate: >85%

## Incident Response

### Severity Levels
- **P0/Critical**: Service down, data loss risk
- **P1/High**: Major functionality impaired
- **P2/Medium**: Minor functionality issues
- **P3/Low**: Cosmetic or enhancement requests

### Response Times
- P0: 15 minutes
- P1: 1 hour
- P2: 4 hours
- P3: Next business day

## Common Scenarios

### High CPU Usage
1. Check current load: `kubectl top pods`
2. Scale up replicas: `kubectl scale deployment rag-system --replicas=10`
3. Monitor auto-scaler triggers
4. Investigate root cause in metrics

### Memory Leaks
1. Identify affected pods: `kubectl top pods --sort-by=memory`
2. Restart affected pods: `kubectl delete pod <pod-name>`
3. Monitor memory usage trends
4. Review application logs for errors

### Cache Issues
1. Check Redis connectivity: `kubectl exec -it <pod> -- redis-cli ping`
2. Monitor cache hit rates in Grafana
3. Clear cache if corrupted: `kubectl exec <redis-pod> -- redis-cli FLUSHALL`
4. Restart cache pods if needed

### Database Connectivity
1. Test connection from pod: `kubectl exec -it <pod> -- python -c "import psycopg2; ..."`
2. Check database server status
3. Verify connection secrets: `kubectl get secrets`
4. Review network policies

## Maintenance Procedures

### Planned Deployments
1. Announce maintenance window
2. Scale up replicas for zero-downtime
3. Deploy new version
4. Monitor for 30 minutes
5. Scale back to normal

### Database Maintenance
1. Enable read-only mode
2. Create database backup
3. Perform maintenance
4. Verify integrity
5. Restore normal operations

### Security Updates
1. Scan for vulnerabilities
2. Test updates in staging
3. Apply during maintenance window
4. Verify all functionality
5. Update documentation

## Monitoring Playbook

### Key Dashboards
- **System Overview**: Overall health and performance
- **Application Metrics**: RAG-specific metrics
- **Infrastructure**: Kubernetes cluster health
- **Security**: Attack patterns and anomalies

### Alert Rules
```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High error rate detected"

- alert: SlowResponseTime
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
  for: 10m
  labels:
    severity: critical
  annotations:
    summary: "Response time is too slow"
```

## Performance Tuning

### Optimization Checklist
- [ ] Cache hit rate >85%
- [ ] Response time <500ms p95
- [ ] CPU usage <70% average
- [ ] Memory usage <80%
- [ ] Error rate <0.1%

### Scaling Guidelines
- Scale up when CPU >70% for 5+ minutes
- Scale down when CPU <30% for 10+ minutes
- Always maintain minimum 3 replicas
- Max 20 replicas unless approved

## Data Management

### Backup Schedule
- **Daily**: Application data backup
- **Weekly**: Full system backup
- **Monthly**: Offsite backup verification

### Retention Policies
- Logs: 30 days
- Metrics: 90 days
- Backups: 1 year
- Audit logs: 7 years

## Compliance & Security

### Regular Reviews
- Security scan results
- Access control lists
- SSL certificate expiry
- Dependency vulnerabilities

### Audit Requirements
- Log all administrative actions
- Monitor privileged access
- Track data access patterns
- Maintain compliance reports
