"""
Security management for RAG system including rate limiting, authentication, and access control.
"""

import logging
import time
import hashlib
import hmac
import secrets
import re
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import ipaddress
from enum import Enum


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of detected attacks."""
    INJECTION = "injection"
    XSS = "xss"
    BRUTE_FORCE = "brute_force"
    ENUMERATION = "enumeration"
    SCRAPING = "scraping"
    DDOS = "ddos"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


@dataclass
class ThreatDetection:
    """Threat detection result."""
    attack_type: AttackType
    threat_level: ThreatLevel
    confidence: float
    indicators: List[str]
    recommended_action: str
    details: Dict[str, Any]


@dataclass
class SecurityEvent:
    """Security event for logging and monitoring."""
    event_type: str
    severity: str
    source_ip: Optional[str]
    user_id: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 10


class RateLimiter:
    """Token bucket rate limiter with multiple time windows."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.client_requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(
        self,
        client_id: str,
        current_time: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Unique client identifier
            current_time: Current timestamp (for testing)
            
        Returns:
            Tuple of (allowed, details)
        """
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            # Clean old requests
            self._cleanup_old_requests(client_id, current_time)
            
            requests = self.client_requests[client_id]
            
            # Check rate limits
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            day_ago = current_time - 86400
            
            minute_count = sum(1 for req_time in requests if req_time >= minute_ago)
            hour_count = sum(1 for req_time in requests if req_time >= hour_ago)
            day_count = sum(1 for req_time in requests if req_time >= day_ago)
            
            # Check limits
            if minute_count >= self.config.requests_per_minute:
                return False, {
                    "reason": "minute_limit_exceeded",
                    "limit": self.config.requests_per_minute,
                    "current": minute_count,
                    "reset_time": minute_ago + 60
                }
            
            if hour_count >= self.config.requests_per_hour:
                return False, {
                    "reason": "hour_limit_exceeded", 
                    "limit": self.config.requests_per_hour,
                    "current": hour_count,
                    "reset_time": hour_ago + 3600
                }
            
            if day_count >= self.config.requests_per_day:
                return False, {
                    "reason": "day_limit_exceeded",
                    "limit": self.config.requests_per_day, 
                    "current": day_count,
                    "reset_time": day_ago + 86400
                }
            
            # Record request
            requests.append(current_time)
            
            return True, {
                "minute_count": minute_count + 1,
                "hour_count": hour_count + 1,
                "day_count": day_count + 1,
                "limits": {
                    "minute": self.config.requests_per_minute,
                    "hour": self.config.requests_per_hour,
                    "day": self.config.requests_per_day
                }
            }
    
    def _cleanup_old_requests(self, client_id: str, current_time: float) -> None:
        """Remove old request records."""
        cutoff_time = current_time - 86400  # Keep 24 hours
        requests = self.client_requests[client_id]
        self.client_requests[client_id] = [
            req_time for req_time in requests if req_time >= cutoff_time
        ]
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get rate limiting statistics for a client."""
        current_time = time.time()
        with self._lock:
            self._cleanup_old_requests(client_id, current_time)
            requests = self.client_requests[client_id]
            
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            day_ago = current_time - 86400
            
            return {
                "requests_last_minute": sum(1 for req_time in requests if req_time >= minute_ago),
                "requests_last_hour": sum(1 for req_time in requests if req_time >= hour_ago),
                "requests_last_day": sum(1 for req_time in requests if req_time >= day_ago),
                "limits": {
                    "minute": self.config.requests_per_minute,
                    "hour": self.config.requests_per_hour,
                    "day": self.config.requests_per_day
                }
            }


class SecurityManager:
    """Comprehensive security management for RAG system."""
    
    def __init__(
        self,
        rate_limit_config: Optional[RateLimitConfig] = None,
        enable_ip_filtering: bool = True,
        enable_request_signing: bool = False
    ):
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.enable_ip_filtering = enable_ip_filtering
        self.enable_request_signing = enable_request_signing
        self.logger = logging.getLogger(__name__)
        
        # Security state
        self.blocked_ips: set = set()
        self.allowed_ip_ranges: List[ipaddress.IPv4Network] = []
        self.security_events: List[SecurityEvent] = []
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default allowed IP ranges (can be configured)
        self._initialize_default_ip_ranges()
    
    def validate_request(
        self,
        client_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        signature: Optional[str] = None,
        payload: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive request validation.
        
        Args:
            client_ip: Client IP address
            user_id: User identifier
            api_key: API key for authentication
            signature: Request signature for verification
            payload: Request payload for signature verification
            
        Returns:
            Tuple of (is_valid, details)
        """
        details = {"checks": []}
        
        try:
            # IP filtering check
            if self.enable_ip_filtering and client_ip:
                ip_valid, ip_details = self._validate_ip(client_ip)
                details["checks"].append({"type": "ip_validation", "result": ip_valid, "details": ip_details})
                if not ip_valid:
                    self._log_security_event("ip_blocked", "medium", client_ip, user_id, ip_details)
                    return False, details
            
            # Rate limiting check
            client_id = self._get_client_id(client_ip, user_id, api_key)
            rate_limit_valid, rate_details = self.rate_limiter.is_allowed(client_id)
            details["checks"].append({"type": "rate_limit", "result": rate_limit_valid, "details": rate_details})
            
            if not rate_limit_valid:
                self._log_security_event("rate_limit_exceeded", "medium", client_ip, user_id, rate_details)
                return False, details
            
            # API key validation
            if api_key:
                key_valid, key_details = self._validate_api_key(api_key)
                details["checks"].append({"type": "api_key", "result": key_valid, "details": key_details})
                if not key_valid:
                    self._log_security_event("invalid_api_key", "high", client_ip, user_id, key_details)
                    return False, details
            
            # Request signature validation
            if self.enable_request_signing and signature and payload:
                sig_valid, sig_details = self._validate_signature(api_key, payload, signature)
                details["checks"].append({"type": "signature", "result": sig_valid, "details": sig_details})
                if not sig_valid:
                    self._log_security_event("invalid_signature", "high", client_ip, user_id, sig_details)
                    return False, details
            
            # All checks passed
            details["client_id"] = client_id
            return True, details
            
        except Exception as e:
            self.logger.error(f"Security validation error: {e}")
            self._log_security_event("validation_error", "critical", client_ip, user_id, {"error": str(e)})
            return False, {"error": "Security validation failed"}
    
    def _validate_ip(self, client_ip: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate client IP address."""
        try:
            ip_addr = ipaddress.ip_address(client_ip)
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                return False, {"reason": "ip_blocked", "ip": client_ip}
            
            # Check if IP is in allowed ranges (if configured)
            if self.allowed_ip_ranges:
                ip_allowed = any(ip_addr in network for network in self.allowed_ip_ranges)
                if not ip_allowed:
                    return False, {"reason": "ip_not_in_allowed_range", "ip": client_ip}
            
            # Check for private/local IP addresses in production
            if ip_addr.is_private and not self._is_development_mode():
                return False, {"reason": "private_ip_not_allowed", "ip": client_ip}
            
            return True, {"ip": client_ip, "valid": True}
            
        except ValueError:
            return False, {"reason": "invalid_ip_format", "ip": client_ip}
    
    def _validate_api_key(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate API key."""
        if not api_key:
            return False, {"reason": "api_key_missing"}
        
        # Hash the API key for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash not in self.api_keys:
            return False, {"reason": "api_key_invalid"}
        
        key_info = self.api_keys[key_hash]
        
        # Check if key is expired
        if key_info.get("expires"):
            expires = datetime.fromisoformat(key_info["expires"])
            if datetime.utcnow() > expires:
                return False, {"reason": "api_key_expired", "expired_at": key_info["expires"]}
        
        # Check if key is active
        if not key_info.get("active", True):
            return False, {"reason": "api_key_inactive"}
        
        return True, {
            "key_id": key_info.get("key_id"),
            "user_id": key_info.get("user_id"),
            "scopes": key_info.get("scopes", [])
        }
    
    def _validate_signature(self, api_key: Optional[str], payload: str, signature: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate request signature using HMAC."""
        if not api_key:
            return False, {"reason": "api_key_required_for_signature"}
        
        try:
            # Get secret key for API key
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            key_info = self.api_keys.get(key_hash)
            
            if not key_info or "secret" not in key_info:
                return False, {"reason": "secret_key_not_found"}
            
            # Calculate expected signature
            secret = key_info["secret"].encode()
            expected_signature = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
            
            # Compare signatures securely
            if not hmac.compare_digest(signature, expected_signature):
                return False, {"reason": "signature_mismatch"}
            
            return True, {"signature_valid": True}
            
        except Exception as e:
            return False, {"reason": "signature_validation_error", "error": str(e)}
    
    def _get_client_id(
        self,
        client_ip: Optional[str],
        user_id: Optional[str],
        api_key: Optional[str]
    ) -> str:
        """Generate unique client ID for rate limiting."""
        if user_id:
            return f"user:{user_id}"
        elif api_key:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"key:{key_hash}"
        elif client_ip:
            return f"ip:{client_ip}"
        else:
            return "anonymous"
    
    def _log_security_event(
        self,
        event_type: str,
        severity: str,
        source_ip: Optional[str],
        user_id: Optional[str],
        details: Dict[str, Any]
    ) -> None:
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            details=details,
            timestamp=datetime.utcnow()
        )
        
        with self._lock:
            self.security_events.append(event)
            
            # Keep only recent events
            cutoff = datetime.utcnow() - timedelta(hours=24)
            self.security_events = [
                e for e in self.security_events if e.timestamp >= cutoff
            ]
        
        # Log to system logger
        log_message = f"Security event: {event_type} from {source_ip or 'unknown'}"
        if user_id:
            log_message += f" (user: {user_id})"
        log_message += f" - {details}"
        
        if severity == "critical":
            self.logger.critical(log_message)
        elif severity == "high":
            self.logger.error(log_message)
        elif severity == "medium":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _initialize_default_ip_ranges(self) -> None:
        """Initialize default allowed IP ranges."""
        # Allow common private networks for development
        if self._is_development_mode():
            self.allowed_ip_ranges.extend([
                ipaddress.IPv4Network('127.0.0.0/8'),    # Localhost
                ipaddress.IPv4Network('10.0.0.0/8'),     # Private A
                ipaddress.IPv4Network('172.16.0.0/12'),  # Private B
                ipaddress.IPv4Network('192.168.0.0/16'), # Private C
            ])
    
    def _is_development_mode(self) -> bool:
        """Check if running in development mode."""
        import os
        return os.getenv('ENVIRONMENT', 'production').lower() in ['development', 'dev', 'local']
    
    def block_ip(self, ip: str, reason: str = "manual_block") -> None:
        """Block an IP address."""
        with self._lock:
            self.blocked_ips.add(ip)
        
        self._log_security_event("ip_blocked", "medium", ip, None, {"reason": reason})
        self.logger.warning(f"Blocked IP address: {ip} (reason: {reason})")
    
    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address."""
        with self._lock:
            self.blocked_ips.discard(ip)
        
        self.logger.info(f"Unblocked IP address: {ip}")
    
    def create_api_key(
        self,
        user_id: str,
        scopes: Optional[List[str]] = None,
        expires_days: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Create new API key.
        
        Args:
            user_id: User identifier
            scopes: API key scopes/permissions
            expires_days: Expiration in days (None for no expiration)
            
        Returns:
            Tuple of (api_key, key_id)
        """
        # Generate API key and secret
        api_key = secrets.token_urlsafe(32)
        secret = secrets.token_urlsafe(32)
        key_id = secrets.token_hex(8)
        
        # Calculate expiration
        expires = None
        if expires_days:
            expires = (datetime.utcnow() + timedelta(days=expires_days)).isoformat()
        
        # Store key info
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.api_keys[key_hash] = {
            "key_id": key_id,
            "user_id": user_id,
            "scopes": scopes or [],
            "secret": secret,
            "created": datetime.utcnow().isoformat(),
            "expires": expires,
            "active": True
        }
        
        self.logger.info(f"Created API key {key_id} for user {user_id}")
        return api_key, key_id
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            self.api_keys[key_hash]["active"] = False
            key_id = self.api_keys[key_hash]["key_id"]
            self.logger.info(f"Revoked API key {key_id}")
            return True
        
        return False
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        with self._lock:
            recent_events = [
                e for e in self.security_events 
                if e.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ]
            
            event_counts = defaultdict(int)
            for event in recent_events:
                event_counts[event.event_type] += 1
        
        return {
            "blocked_ips": len(self.blocked_ips),
            "active_api_keys": sum(1 for key_info in self.api_keys.values() if key_info.get("active")),
            "recent_security_events": len(recent_events),
            "event_types_last_hour": dict(event_counts),
            "rate_limit_config": {
                "requests_per_minute": self.rate_limiter.config.requests_per_minute,
                "requests_per_hour": self.rate_limiter.config.requests_per_hour,
                "requests_per_day": self.rate_limiter.config.requests_per_day
            }
        }


class ThreatDetectionEngine:
    """Advanced threat detection and monitoring engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Pattern databases for threat detection
        self.injection_patterns = [
            r"(?i)(union\s+select|select\s+.*\s+from|drop\s+table|delete\s+from)",
            r"(?i)(insert\s+into|update\s+.*\s+set|alter\s+table)",
            r"(?i)(\'\s*or\s+\d+\s*=\s*\d+|--|\#|\/\*|\*\/)",
            r"(?i)(exec\s*\(|eval\s*\(|system\s*\(|shell_exec)",
            r"(?i)(script\s*>|javascript:|data:text/html)"
        ]
        
        self.xss_patterns = [
            r"(?i)(<script[^>]*>.*?</script>|<script[^>]*>)",
            r"(?i)(javascript:|data:text/html|vbscript:)",
            r"(?i)(on\w+\s*=|<iframe|<object|<embed)",
            r"(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()",
            r"(?i)(document\.cookie|document\.domain|window\.location)"
        ]
        
        self.suspicious_patterns = [
            r"(?i)(password|passwd|pwd|secret|token|key)\s*[:=]",
            r"(?i)(admin|administrator|root|superuser)",
            r"(?i)(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",  # Path traversal
            r"(?i)(\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b)",  # Credit card
            r"(?i)(\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b)",  # SSN
            r"(?i)([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"  # Email
        ]
        
        # Behavioral analysis tracking
        self.client_behavior: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "request_count": 0,
            "error_count": 0,
            "unique_queries": set(),
            "suspicious_queries": 0,
            "last_seen": time.time(),
            "request_intervals": deque(maxlen=50),
            "user_agents": set(),
            "threat_score": 0.0
        })
        
        # Threat intelligence feeds (simplified)
        self.known_malicious_ips: Set[str] = set()
        self.suspicious_patterns_cache: Dict[str, ThreatDetection] = {}
        
        # Load threat intelligence (in production, would be from external feeds)
        self._load_threat_intelligence()
    
    def analyze_request(
        self,
        query: str,
        client_ip: str,
        user_agent: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ThreatDetection]:
        """
        Analyze incoming request for threats.
        
        Args:
            query: User query to analyze
            client_ip: Client IP address
            user_agent: User agent string
            user_id: User identifier
            metadata: Additional request metadata
            
        Returns:
            List of detected threats
        """
        threats = []
        current_time = time.time()
        
        with self._lock:
            # Update behavioral tracking
            self._update_client_behavior(client_ip, query, user_agent, current_time)
            
            # Analyze query content
            content_threats = self._analyze_content(query)
            threats.extend(content_threats)
            
            # Analyze behavioral patterns
            behavioral_threats = self._analyze_behavior(client_ip, current_time)
            threats.extend(behavioral_threats)
            
            # Check against threat intelligence
            intel_threats = self._check_threat_intelligence(client_ip, query)
            threats.extend(intel_threats)
            
            # Update threat score
            if threats:
                behavior = self.client_behavior[client_ip]
                for threat in threats:
                    behavior["threat_score"] += threat.confidence * 0.1
                
                # Cap threat score
                behavior["threat_score"] = min(behavior["threat_score"], 10.0)
        
        return threats
    
    def _analyze_content(self, query: str) -> List[ThreatDetection]:
        """Analyze query content for malicious patterns."""
        threats = []
        
        # Check for SQL injection
        for pattern in self.injection_patterns:
            if re.search(pattern, query):
                threats.append(ThreatDetection(
                    attack_type=AttackType.INJECTION,
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.8,
                    indicators=[f"SQL injection pattern detected: {pattern}"],
                    recommended_action="block_request",
                    details={"matched_pattern": pattern, "query_excerpt": query[:100]}
                ))
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, query):
                threats.append(ThreatDetection(
                    attack_type=AttackType.XSS,
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.7,
                    indicators=[f"XSS pattern detected: {pattern}"],
                    recommended_action="sanitize_and_block",
                    details={"matched_pattern": pattern, "query_excerpt": query[:100]}
                ))
        
        # Check for suspicious patterns
        suspicious_count = 0
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query):
                suspicious_count += 1
        
        if suspicious_count >= 2:
            threats.append(ThreatDetection(
                attack_type=AttackType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.6,
                indicators=[f"Multiple suspicious patterns detected: {suspicious_count}"],
                recommended_action="monitor_closely",
                details={"pattern_count": suspicious_count, "query_excerpt": query[:100]}
            ))
        
        return threats
    
    def _analyze_behavior(self, client_ip: str, current_time: float) -> List[ThreatDetection]:
        """Analyze client behavioral patterns."""
        threats = []
        behavior = self.client_behavior[client_ip]
        
        # Check request frequency
        if len(behavior["request_intervals"]) >= 10:
            avg_interval = sum(behavior["request_intervals"][-10:]) / 10
            if avg_interval < 1.0:  # Less than 1 second between requests
                threats.append(ThreatDetection(
                    attack_type=AttackType.SCRAPING,
                    threat_level=ThreatLevel.MEDIUM,
                    confidence=0.7,
                    indicators=[f"High frequency requests: {avg_interval:.2f}s average interval"],
                    recommended_action="rate_limit",
                    details={"avg_interval": avg_interval, "recent_requests": len(behavior["request_intervals"])}
                ))
        
        # Check for brute force patterns
        if behavior["error_count"] > 10 and behavior["request_count"] > 20:
            error_rate = behavior["error_count"] / behavior["request_count"]
            if error_rate > 0.5:
                threats.append(ThreatDetection(
                    attack_type=AttackType.BRUTE_FORCE,
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.8,
                    indicators=[f"High error rate: {error_rate:.1%}"],
                    recommended_action="temporary_block",
                    details={"error_rate": error_rate, "total_requests": behavior["request_count"]}
                ))
        
        # Check for enumeration attempts
        if len(behavior["unique_queries"]) > 50 and behavior["request_count"] > 100:
            uniqueness_ratio = len(behavior["unique_queries"]) / behavior["request_count"]
            if uniqueness_ratio > 0.8:  # Very diverse queries
                threats.append(ThreatDetection(
                    attack_type=AttackType.ENUMERATION,
                    threat_level=ThreatLevel.MEDIUM,
                    confidence=0.6,
                    indicators=[f"High query diversity: {uniqueness_ratio:.1%}"],
                    recommended_action="monitor_closely",
                    details={"uniqueness_ratio": uniqueness_ratio, "unique_queries": len(behavior["unique_queries"])}
                ))
        
        # Check overall threat score
        if behavior["threat_score"] > 5.0:
            threats.append(ThreatDetection(
                attack_type=AttackType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.HIGH,
                confidence=min(behavior["threat_score"] / 10.0, 0.95),
                indicators=[f"High cumulative threat score: {behavior['threat_score']:.1f}"],
                recommended_action="block_client",
                details={"threat_score": behavior["threat_score"]}
            ))
        
        return threats
    
    def _check_threat_intelligence(self, client_ip: str, query: str) -> List[ThreatDetection]:
        """Check against threat intelligence feeds."""
        threats = []
        
        # Check known malicious IPs
        if client_ip in self.known_malicious_ips:
            threats.append(ThreatDetection(
                attack_type=AttackType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.CRITICAL,
                confidence=0.95,
                indicators=["IP address in threat intelligence database"],
                recommended_action="block_immediately",
                details={"source": "threat_intelligence", "ip": client_ip}
            ))
        
        # Check for common attack vectors in query
        attack_keywords = [
            "admin", "password", "login", "token", "secret", "key",
            "config", "database", "backup", "test", "debug"
        ]
        
        query_lower = query.lower()
        found_keywords = [kw for kw in attack_keywords if kw in query_lower]
        
        if len(found_keywords) >= 3:
            threats.append(ThreatDetection(
                attack_type=AttackType.ENUMERATION,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.5,
                indicators=[f"Multiple attack keywords: {', '.join(found_keywords)}"],
                recommended_action="monitor_closely",
                details={"keywords": found_keywords}
            ))
        
        return threats
    
    def _update_client_behavior(
        self, 
        client_ip: str, 
        query: str, 
        user_agent: Optional[str], 
        current_time: float
    ) -> None:
        """Update client behavioral tracking."""
        behavior = self.client_behavior[client_ip]
        
        # Update counters
        behavior["request_count"] += 1
        
        # Track request intervals
        if behavior["last_seen"]:
            interval = current_time - behavior["last_seen"]
            behavior["request_intervals"].append(interval)
        
        behavior["last_seen"] = current_time
        
        # Track unique queries (hash for privacy)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        behavior["unique_queries"].add(query_hash)
        
        # Track user agents
        if user_agent:
            behavior["user_agents"].add(user_agent)
        
        # Check for suspicious query patterns
        if any(re.search(pattern, query) for pattern in self.suspicious_patterns):
            behavior["suspicious_queries"] += 1
    
    def _load_threat_intelligence(self) -> None:
        """Load threat intelligence data."""
        # In production, this would load from external threat feeds
        # For now, adding some known bad IPs as examples
        self.known_malicious_ips.update([
            "192.168.1.100",  # Example malicious IP
            "10.0.0.5"        # Example malicious IP
        ])
    
    def get_client_threat_score(self, client_ip: str) -> float:
        """Get current threat score for client."""
        return self.client_behavior[client_ip]["threat_score"]
    
    def reset_client_threats(self, client_ip: str) -> None:
        """Reset threat tracking for client."""
        if client_ip in self.client_behavior:
            self.client_behavior[client_ip]["threat_score"] = 0.0
            self.client_behavior[client_ip]["error_count"] = 0
            self.client_behavior[client_ip]["suspicious_queries"] = 0
    
    def get_threat_stats(self) -> Dict[str, Any]:
        """Get threat detection statistics."""
        with self._lock:
            total_clients = len(self.client_behavior)
            high_threat_clients = sum(
                1 for behavior in self.client_behavior.values()
                if behavior["threat_score"] > 3.0
            )
            
            total_requests = sum(
                behavior["request_count"]
                for behavior in self.client_behavior.values()
            )
            
            suspicious_requests = sum(
                behavior["suspicious_queries"]
                for behavior in self.client_behavior.values()
            )
            
            return {
                "total_clients_tracked": total_clients,
                "high_threat_clients": high_threat_clients,
                "total_requests_analyzed": total_requests,
                "suspicious_requests_detected": suspicious_requests,
                "threat_intel_ips": len(self.known_malicious_ips),
                "detection_patterns": {
                    "injection_patterns": len(self.injection_patterns),
                    "xss_patterns": len(self.xss_patterns),
                    "suspicious_patterns": len(self.suspicious_patterns)
                }
            }