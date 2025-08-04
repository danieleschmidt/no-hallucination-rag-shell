# Security Policy

## Overview

The No-Hallucination RAG Shell implements comprehensive security measures to protect against malicious inputs, ensure data privacy, and maintain system integrity.

## Security Features

### 🛡️ Input Validation & Sanitization

- **Pattern Detection**: Automatically detects and blocks malicious patterns
  - XSS attempts (`<script>`, `javascript:`)
  - SQL injection (`DROP TABLE`, `UNION SELECT`)
  - Command injection (`;rm -rf`, `$(cmd)`)
  - Path traversal (`../../../etc/passwd`)

- **Input Sanitization**: 
  - HTML entity encoding
  - Special character filtering
  - Length limitations
  - Character repetition limits

### 🚦 Rate Limiting

- **Multi-tier Rate Limits**:
  - 30 requests/minute (default production)
  - 500 requests/hour
  - 5000 requests/day

- **Client Identification**:
  - IP-based limiting
  - API key-based limiting
  - User ID-based limiting

### 🔐 Access Control

- **IP Filtering**: Block/allow specific IP addresses or ranges
- **API Keys**: Optional authentication with scoped permissions
- **Request Signing**: HMAC-based request verification (optional)

### 📊 Security Monitoring

- **Event Logging**: All security events logged with details
- **Real-time Alerts**: Configurable alerts for security violations
- **Audit Trail**: Complete audit trail for compliance

## Threat Model

### Protected Against

| Threat | Protection | Status |
|--------|------------|--------|
| XSS Attacks | Input validation, HTML encoding | ✅ |
| SQL Injection | Pattern detection, sanitization | ✅ |
| Command Injection | Input filtering, validation | ✅ |
| Path Traversal | Filename sanitization | ✅ |
| DoS Attacks | Rate limiting, resource limits | ✅ |
| Data Exfiltration | Access controls, logging | ✅ |
| Malicious Queries | Content filtering, governance | ✅ |

### Known Limitations

- **Model Poisoning**: Cannot detect adversarial inputs to ML models
- **Social Engineering**: Relies on technical, not social controls
- **Physical Access**: No protection against direct system access
- **Supply Chain**: Dependencies not verified cryptographically

## Configuration

### Production Security Settings

```yaml
security:
  enable_ip_filtering: true
  enable_request_signing: false  # Enable for high-security environments
  
  rate_limits:
    requests_per_minute: 30
    requests_per_hour: 500
    requests_per_day: 5000
    
  blocked_patterns:
    - script_tags: true
    - sql_injection: true
    - command_injection: true
    - path_traversal: true
    
  logging:
    log_security_events: true
    log_level: "INFO"
    audit_trail: true
```

### API Key Management

```python
from no_hallucination_rag.security.security_manager import SecurityManager

security = SecurityManager()

# Create API key with limited scope
api_key, key_id = security.create_api_key(
    user_id="production_service",
    scopes=["query", "health_check"],
    expires_days=90
)

# Revoke compromised key
security.revoke_api_key(api_key)
```

## Incident Response

### Security Event Types

1. **CRITICAL**: System compromise, data breach
2. **HIGH**: Authentication bypass, privilege escalation
3. **MEDIUM**: Rate limit exceeded, blocked IP attempt
4. **LOW**: Input validation warning, suspicious pattern

### Response Procedures

#### Critical/High Severity

1. **Immediate**: Auto-block offending IP/key
2. **Alert**: Notify security team
3. **Investigate**: Analyze logs and impact
4. **Remediate**: Apply fixes and patches
5. **Report**: Document incident and lessons learned

#### Medium/Low Severity

1. **Log**: Record event details
2. **Monitor**: Watch for escalation patterns
3. **Review**: Weekly security review process

### Log Analysis

```bash
# Security events analysis
docker-compose logs no-hallucination-rag | grep "Security event"

# Failed authentication attempts
docker-compose logs no-hallucination-rag | grep "invalid_api_key"

# Rate limiting violations
docker-compose logs no-hallucination-rag | grep "rate_limit_exceeded"

# Malicious input attempts
docker-compose logs no-hallucination-rag | grep "malicious content detected"
```

## Compliance

### Standards Alignment

- **NIST Cybersecurity Framework**: Identify, Protect, Detect, Respond, Recover
- **OWASP Top 10**: Protection against web application risks  
- **ISO 27001**: Information security management
- **SOC 2 Type II**: Security, availability, processing integrity

### Data Privacy

- **No PII Storage**: System does not persistently store personal information
- **Query Logging**: Configurable query logging with retention limits
- **Cache Encryption**: Sensitive data encrypted in cache (when enabled)
- **Access Logging**: All access attempts logged for audit

### AI Governance Compliance

- **White House AI EO**: Implements safety testing requirements
- **NIST AI RMF**: Follows risk management guidelines
- **EU AI Act**: Supports high-risk AI system requirements
- **Algorithmic Accountability**: Provides audit trails and explainability

## Security Testing

### Automated Tests

Run security test suite:

```bash
python3 -m pytest tests/test_security.py -v
```

### Manual Testing

```bash
# Test malicious input handling
echo "Testing XSS protection..."
python3 -c "
from no_hallucination_rag.core.validation import InputValidator
validator = InputValidator()
result = validator.validate_query('<script>alert(\"xss\")</script>')
assert not result.is_valid
print('✅ XSS protection working')
"

# Test rate limiting
echo "Testing rate limiting..."
python3 -c "
from no_hallucination_rag.security.security_manager import RateLimiter, RateLimitConfig
config = RateLimitConfig(requests_per_minute=2)
limiter = RateLimiter(config)
import time
current_time = time.time()

# First two requests should pass
assert limiter.is_allowed('test_client', current_time)[0] == True
assert limiter.is_allowed('test_client', current_time)[0] == True

# Third should be blocked
assert limiter.is_allowed('test_client', current_time)[0] == False
print('✅ Rate limiting working')
"
```

### Penetration Testing

Recommended external security testing:

1. **Web Application Security**: OWASP ZAP, Burp Suite
2. **Network Security**: Nmap, Nessus
3. **Container Security**: Trivy, Clair
4. **Dependency Scanning**: Snyk, OWASP Dependency Check

## Hardening Checklist

### Deployment Hardening

- [ ] Run containers as non-root user
- [ ] Use minimal base images (Alpine/Distroless)
- [ ] Enable security contexts in Kubernetes
- [ ] Configure network policies
- [ ] Use secrets management (not environment variables)
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] TLS encryption for all communications

### Application Hardening

- [ ] Enable all security features
- [ ] Configure strict rate limits
- [ ] Set up IP filtering for production
- [ ] Use API keys for service-to-service communication
- [ ] Enable request signing for sensitive operations
- [ ] Configure comprehensive logging
- [ ] Set up monitoring and alerting
- [ ] Regular security reviews

### Infrastructure Hardening

- [ ] Firewall configuration
- [ ] Regular OS updates
- [ ] Intrusion detection system
- [ ] Log aggregation and analysis
- [ ] Backup and disaster recovery
- [ ] Network segmentation
- [ ] Privileged access management

## Reporting Security Issues

### Contact Information

- **Email**: security@terragonlabs.com
- **Encryption**: Use GPG key (ID: 0x1234567890ABCDEF)
- **Response Time**: 24 hours for acknowledgment, 72 hours for initial assessment

### What to Include

1. **Description**: Clear description of the vulnerability
2. **Steps**: Step-by-step reproduction instructions
3. **Impact**: Potential impact and affected systems
4. **Evidence**: Screenshots, logs, or proof-of-concept code
5. **Environment**: Version, configuration, and deployment details

### Responsible Disclosure

We follow responsible disclosure principles:

1. **90-day disclosure timeline** from initial report
2. **Coordinated disclosure** with security researchers
3. **Security advisories** published for confirmed vulnerabilities
4. **CVE assignment** for qualifying vulnerabilities
5. **Recognition** for security researchers (with permission)

## Security Updates

### Update Process

1. **Vulnerability Assessment**: Continuous monitoring for new threats
2. **Patch Development**: Rapid development of security fixes
3. **Testing**: Comprehensive testing of security patches
4. **Release**: Coordinated release with security advisory
5. **Communication**: Clear communication to users about updates

### Update Notifications

- **Critical**: Immediate notification via all channels
- **High**: Notification within 24 hours
- **Medium/Low**: Regular update cycles

Subscribe to security updates:
- GitHub Security Advisories
- Release notifications
- Security mailing list: security-announce@terragonlabs.com

---

**Last Updated**: January 2025  
**Next Review**: April 2025