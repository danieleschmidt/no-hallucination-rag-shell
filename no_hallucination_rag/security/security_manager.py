"""
Enhanced security manager for RAG system.
Generation 2: Comprehensive security with threat detection and prevention.
"""

import logging
import time
import re
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class SecurityValidationResult:
    """Result of security validation."""
    is_valid: bool
    risk_score: float
    threats: List[str]
    recommended_action: str


@dataclass
class InjectionCheckResult:
    """Result of injection attempt detection."""
    is_injection: bool
    confidence: float
    detected_patterns: List[str]
    recommended_action: str


class RateLimiter:
    """Rate limiter with sliding window."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed for client - compatibility method."""
        user_context = {"user_id": client_id}
        allowed = self.allow_request(user_context)
        identifier = self._get_identifier(user_context)
        return allowed, {
            "requests_remaining": max(0, self.max_requests - len(self.requests[identifier])),
            "minute_count": len(self.requests[identifier]),
            "window_seconds": self.window_seconds
        }
        
    def allow_request(self, user_context: Optional[dict] = None) -> bool:
        """Check if request is allowed based on rate limits."""
        identifier = self._get_identifier(user_context)
        current_time = time.time()
        
        # Clean old requests
        user_requests = self.requests[identifier]
        while user_requests and current_time - user_requests[0] > self.window_seconds:
            user_requests.popleft()
            
        # Check rate limit
        if len(user_requests) >= self.max_requests:
            return False
            
        user_requests.append(current_time)
        return True
        
    def _get_identifier(self, user_context: Optional[dict]) -> str:
        """Get identifier for rate limiting."""
        if not user_context:
            return "anonymous"
        return user_context.get("user_id", user_context.get("ip_address", "anonymous"))


class ContentFilter:
    """Content filtering for malicious content."""
    
    def __init__(self):
        self.blocked_categories = ["explicit", "harmful", "spam", "malicious"]
        
    def scan_content(self, content: str) -> "ContentFilterResult":
        """Scan content for violations."""
        violations = []
        risk_score = 0.0
        
        # Check for explicit content markers
        explicit_patterns = [
            r'\b(password|secret|token|key)\s*[:=]\s*[^\s]+',
            r'\b(admin|root|sudo)\s+(password|access)',
            r'\b(credit\s+card|social\s+security)\b'
        ]
        
        for pattern in explicit_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append("Sensitive information detected")
                risk_score += 20.0
                
        return ContentFilterResult(
            is_safe=risk_score < 30.0,
            violations=violations,
            risk_score=risk_score
        )


@dataclass
class ContentFilterResult:
    """Result of content filtering."""
    is_safe: bool
    violations: List[str]
    risk_score: float


class SecurityManager:
    """Enhanced security manager with threat detection and prevention."""
    
    def __init__(self, enable_rate_limiting: bool = True, enable_content_filtering: bool = True):
        self.logger = logging.getLogger(__name__)
        self.blocked_queries = set()
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.content_filter = ContentFilter() if enable_content_filtering else None
        self.security_events = []
        self.threat_level = "LOW"
        
    def _load_suspicious_patterns(self) -> list:
        """Load patterns that indicate suspicious or malicious queries."""
        return [
            r'(?i)\b(inject|script|alert|eval|exec)\b',
            r'(?i)<[^>]*script[^>]*>',
            r'(?i)\b(union|select|drop|delete|insert)\s+',
            r'(?i)\b(password|token|key|secret)\s*[:=]',
            r'(?i)\b(admin|root|sudo)\b.*\b(access|login|credential)\b'
        ]
        
    def validate_request(self, client_ip: str = None, user_id: str = None, 
                        api_key: str = None) -> Tuple[bool, str]:
        """Enhanced request validation with comprehensive security checks."""
        user_context = {"ip_address": client_ip, "user_id": user_id, "api_key": api_key}
        
        # Rate limiting check
        if self.rate_limiter and not self.rate_limiter.allow_request(user_context):
            self._log_security_event("Rate limit exceeded", [], 40.0, user_context)
            return False, "Rate limit exceeded"
            
        # API key validation (if provided)
        if api_key and not self._validate_api_key(api_key):
            self._log_security_event("Invalid API key", [], 60.0, user_context)
            return False, "Invalid API key"
            
        return True, "Request validated"
        
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and authenticity."""
        # Basic format validation
        if not api_key or len(api_key) < 20:
            return False
        # In production, this would check against a secure store
        return True
        
    def validate_query(self, query: str, user_context: Optional[dict] = None) -> SecurityValidationResult:
        """Comprehensive query validation with threat assessment."""
        threats = []
        risk_score = 0.0
        
        # Rate limiting check
        if self.rate_limiter and not self.rate_limiter.allow_request(user_context):
            threats.append("Rate limit exceeded")
            risk_score += 30.0
            
        # Check blocked queries
        if query in self.blocked_queries:
            self.logger.warning(f"Blocked query detected: {query[:50]}...")
            threats.append("Query in blocklist")
            risk_score += 50.0
        
        # Pattern-based threat detection
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query):
                threats.append("Suspicious pattern detected")
                risk_score += 25.0
        
        # Length validation
        if len(query) > 2000:
            threats.append("Query exceeds maximum length")
            risk_score += 15.0
            
        # Content filtering
        if self.content_filter:
            filter_result = self.content_filter.scan_content(query)
            if not filter_result.is_safe:
                threats.extend(filter_result.violations)
                risk_score += filter_result.risk_score
        
        # Log security events
        if threats:
            self._log_security_event(query, threats, risk_score, user_context)
            
        is_valid = risk_score < 50.0
        return SecurityValidationResult(
            is_valid=is_valid,
            risk_score=risk_score,
            threats=threats,
            recommended_action="BLOCK" if risk_score >= 75.0 else "ALLOW" if is_valid else "REVIEW"
        )
        
    def _log_security_event(self, query: str, threats: list, risk_score: float, context: dict):
        """Log security event for monitoring."""
        event = {
            'timestamp': time.time(),
            'query': query[:100],  # Truncated for logging
            'threats': threats,
            'risk_score': risk_score,
            'context': context or {},
            'threat_level': self._calculate_threat_level(risk_score)
        }
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]
            
        self.logger.warning(f"Security event: {event}")
        
    def _calculate_threat_level(self, risk_score: float) -> str:
        """Calculate threat level based on risk score."""
        if risk_score >= 75.0:
            return "HIGH"
        elif risk_score >= 50.0:
            return "MEDIUM"
        else:
            return "LOW"
        
    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        recent_events = [e for e in self.security_events if time.time() - e['timestamp'] < 3600]
        
        threat_levels = {}
        for event in recent_events:
            level = event.get('threat_level', 'LOW')
            threat_levels[level] = threat_levels.get(level, 0) + 1
        
        return {
            "blocked_queries_count": len(self.blocked_queries),
            "total_security_events": len(self.security_events),
            "recent_events_1h": len(recent_events),
            "current_threat_level": self.threat_level,
            "threat_level_breakdown": threat_levels,
            "rate_limiter_active": self.rate_limiter is not None,
            "content_filter_active": self.content_filter is not None
        }