
"""
Comprehensive security system with authentication, authorization, and protection.
"""

import hmac
import hashlib
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import ipaddress
import re


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    timestamp: datetime
    event_type: str
    severity: str  # low, medium, high, critical
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    max_requests: int
    time_window: int  # seconds
    block_duration: int = 300  # 5 minutes default
    

class RateLimiter:
    """Advanced rate limiting system."""
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_clients: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, rule: RateLimitRule):
        """Add rate limiting rule."""
        self.rules[rule.name] = rule
        self.logger.info(f"Added rate limit rule: {rule.name}")
    
    def check_rate_limit(self, client_id: str, rule_name: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if client is within rate limits."""
        if rule_name not in self.rules:
            return True, {"error": "Rule not found"}
        
        rule = self.rules[rule_name]
        now = datetime.utcnow()
        key = f"{rule_name}:{client_id}"
        
        # Check if client is blocked
        if key in self.blocked_clients:
            if now < self.blocked_clients[key]:
                remaining_block = (self.blocked_clients[key] - now).total_seconds()
                return False, {
                    "blocked": True,
                    "remaining_block_time": remaining_block,
                    "reason": "Rate limit exceeded"
                }
            else:
                # Block expired
                del self.blocked_clients[key]
        
        # Clean old requests
        history = self.request_history[key]
        cutoff_time = now - timedelta(seconds=rule.time_window)
        while history and history[0] < cutoff_time:
            history.popleft()
        
        # Check current request count
        if len(history) >= rule.max_requests:
            # Rate limit exceeded - block client
            self.blocked_clients[key] = now + timedelta(seconds=rule.block_duration)
            self.logger.warning(f"Rate limit exceeded for {client_id} on rule {rule_name}")
            
            return False, {
                "blocked": True,
                "reason": "Rate limit exceeded",
                "max_requests": rule.max_requests,
                "time_window": rule.time_window,
                "block_duration": rule.block_duration
            }
        
        # Add current request
        history.append(now)
        
        remaining_requests = rule.max_requests - len(history)
        return True, {
            "allowed": True,
            "remaining_requests": remaining_requests,
            "reset_time": (now + timedelta(seconds=rule.time_window)).isoformat()
        }


class IPWhitelist:
    """IP address whitelist/blacklist management."""
    
    def __init__(self):
        self.whitelist: List[ipaddress.IPv4Network] = []
        self.blacklist: List[ipaddress.IPv4Network] = []
        self.logger = logging.getLogger(__name__)
    
    def add_to_whitelist(self, ip_or_network: str):
        """Add IP or network to whitelist."""
        try:
            network = ipaddress.IPv4Network(ip_or_network, strict=False)
            self.whitelist.append(network)
            self.logger.info(f"Added to whitelist: {ip_or_network}")
        except ValueError as e:
            self.logger.error(f"Invalid IP/network for whitelist: {ip_or_network} - {e}")
    
    def add_to_blacklist(self, ip_or_network: str):
        """Add IP or network to blacklist."""
        try:
            network = ipaddress.IPv4Network(ip_or_network, strict=False)
            self.blacklist.append(network)
            self.logger.info(f"Added to blacklist: {ip_or_network}")
        except ValueError as e:
            self.logger.error(f"Invalid IP/network for blacklist: {ip_or_network} - {e}")
    
    def is_allowed(self, ip_address: str) -> Tuple[bool, str]:
        """Check if IP address is allowed."""
        try:
            ip = ipaddress.IPv4Address(ip_address)
            
            # Check blacklist first
            for network in self.blacklist:
                if ip in network:
                    return False, f"IP {ip_address} is blacklisted"
            
            # If whitelist is empty, allow all (except blacklisted)
            if not self.whitelist:
                return True, "No whitelist configured"
            
            # Check whitelist
            for network in self.whitelist:
                if ip in network:
                    return True, f"IP {ip_address} is whitelisted"
            
            return False, f"IP {ip_address} not in whitelist"
            
        except ValueError as e:
            return False, f"Invalid IP address: {ip_address} - {e}"


class APIKeyManager:
    """API key management and validation."""
    
    def __init__(self):
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.key_usage: Dict[str, List[datetime]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def generate_key(self, user_id: str, permissions: List[str] = None, expires_in_days: int = 30) -> str:
        """Generate new API key."""
        api_key = secrets.token_urlsafe(32)
        
        self.keys[api_key] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=expires_in_days),
            "permissions": permissions or ["read"],
            "active": True,
            "usage_count": 0
        }
        
        self.logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def validate_key(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate API key."""
        if api_key not in self.keys:
            return False, {"error": "Invalid API key"}
        
        key_data = self.keys[api_key]
        
        # Check if key is active
        if not key_data["active"]:
            return False, {"error": "API key is disabled"}
        
        # Check expiration
        if datetime.utcnow() > key_data["expires_at"]:
            return False, {"error": "API key has expired"}
        
        # Update usage
        key_data["usage_count"] += 1
        self.key_usage[api_key].append(datetime.utcnow())
        
        return True, {
            "user_id": key_data["user_id"],
            "permissions": key_data["permissions"],
            "usage_count": key_data["usage_count"]
        }
    
    def revoke_key(self, api_key: str):
        """Revoke API key."""
        if api_key in self.keys:
            self.keys[api_key]["active"] = False
            self.logger.info(f"Revoked API key for user {self.keys[api_key]['user_id']}")
    
    def get_key_stats(self, api_key: str) -> Dict[str, Any]:
        """Get API key usage statistics."""
        if api_key not in self.keys:
            return {"error": "API key not found"}
        
        key_data = self.keys[api_key]
        usage_history = self.key_usage[api_key]
        
        # Calculate usage stats
        now = datetime.utcnow()
        last_24h = sum(1 for usage in usage_history if (now - usage).total_seconds() < 86400)
        last_7d = sum(1 for usage in usage_history if (now - usage).total_seconds() < 604800)
        
        return {
            "user_id": key_data["user_id"],
            "created_at": key_data["created_at"].isoformat(),
            "expires_at": key_data["expires_at"].isoformat(),
            "total_usage": key_data["usage_count"],
            "usage_last_24h": last_24h,
            "usage_last_7d": last_7d,
            "active": key_data["active"]
        }


class SecurityAuditor:
    """Security event logging and analysis."""
    
    def __init__(self, max_events: int = 10000):
        self.events: deque = deque(maxlen=max_events)
        self.alert_thresholds = {
            "failed_auth_burst": {"count": 5, "time_window": 300},  # 5 failures in 5 minutes
            "suspicious_queries": {"count": 10, "time_window": 600},  # 10 suspicious queries in 10 minutes
            "rate_limit_violations": {"count": 3, "time_window": 300}  # 3 rate limit violations in 5 minutes
        }
        self.logger = logging.getLogger(__name__)
    
    def log_event(self, event: SecurityEvent):
        """Log security event."""
        self.events.append(event)
        
        # Log to standard logger
        log_level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(event.severity, logging.INFO)
        
        self.logger.log(log_level, f"[SECURITY] {event.event_type}: {event.description}")
        
        # Check for alert conditions
        self._check_alerts(event)
    
    def log_auth_failure(self, source_ip: str, user_id: str = None, reason: str = ""):
        """Log authentication failure."""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="auth_failure",
            severity="medium",
            source_ip=source_ip,
            user_id=user_id,
            description=f"Authentication failed: {reason}",
            metadata={"reason": reason}
        )
        self.log_event(event)
    
    def log_suspicious_query(self, source_ip: str, query: str, user_id: str = None):
        """Log suspicious query."""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="suspicious_query",
            severity="high",
            source_ip=source_ip,
            user_id=user_id,
            description=f"Suspicious query detected: {query[:100]}...",
            metadata={"query_length": len(query)}
        )
        self.log_event(event)
    
    def log_rate_limit_violation(self, source_ip: str, rule_name: str, user_id: str = None):
        """Log rate limit violation."""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="rate_limit_violation",
            severity="medium",
            source_ip=source_ip,
            user_id=user_id,
            description=f"Rate limit violated: {rule_name}",
            metadata={"rule": rule_name}
        )
        self.log_event(event)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security events summary."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]
        
        # Count by event type
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        source_ips = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
            if event.source_ip:
                source_ips[event.source_ip] += 1
        
        # Top IPs by event count
        top_ips = sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "time_range_hours": hours,
            "total_events": len(recent_events),
            "events_by_type": dict(event_counts),
            "events_by_severity": dict(severity_counts),
            "top_source_ips": top_ips,
            "latest_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type,
                    "severity": e.severity,
                    "description": e.description
                }
                for e in list(recent_events)[-10:]
            ]
        }
    
    def _check_alerts(self, event: SecurityEvent):
        """Check if event triggers security alerts."""
        now = datetime.utcnow()
        
        # Check for auth failure bursts
        if event.event_type == "auth_failure":
            recent_failures = [e for e in self.events 
                             if e.event_type == "auth_failure" 
                             and e.source_ip == event.source_ip
                             and (now - e.timestamp).total_seconds() < 300]
            
            if len(recent_failures) >= 5:
                self.logger.critical(f"ALERT: Authentication failure burst from {event.source_ip}")
        
        # Check for suspicious query patterns
        elif event.event_type == "suspicious_query":
            recent_suspicious = [e for e in self.events
                               if e.event_type == "suspicious_query"
                               and (now - e.timestamp).total_seconds() < 600]
            
            if len(recent_suspicious) >= 10:
                self.logger.critical("ALERT: High volume of suspicious queries detected")


class ComprehensiveSecurityManager:
    """Integrated security management system."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.ip_whitelist = IPWhitelist()
        self.api_key_manager = APIKeyManager()
        self.auditor = SecurityAuditor()
        self.logger = logging.getLogger(__name__)
        
        # Setup default rate limits
        self._setup_default_rules()
    
    def validate_request(self, 
                        client_ip: str = None,
                        api_key: str = None,
                        user_id: str = None,
                        query: str = None) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive request validation."""
        
        # IP whitelist check
        if client_ip:
            ip_allowed, ip_message = self.ip_whitelist.is_allowed(client_ip)
            if not ip_allowed:
                self.auditor.log_event(SecurityEvent(
                    timestamp=datetime.utcnow(),
                    event_type="ip_blocked",
                    severity="high",
                    source_ip=client_ip,
                    description=ip_message
                ))
                return False, {"error": "IP address not allowed", "details": ip_message}
        
        # API key validation
        if api_key:
            key_valid, key_data = self.api_key_manager.validate_key(api_key)
            if not key_valid:
                if client_ip:
                    self.auditor.log_auth_failure(client_ip, user_id, key_data.get("error", "Invalid key"))
                return False, {"error": "Invalid API key", "details": key_data}
            
            user_id = key_data.get("user_id", user_id)
        
        # Rate limiting
        if client_ip:
            rate_ok, rate_info = self.rate_limiter.check_rate_limit(client_ip, "general")
            if not rate_ok:
                self.auditor.log_rate_limit_violation(client_ip, "general", user_id)
                return False, {"error": "Rate limit exceeded", "details": rate_info}
        
        # Query validation (basic security)
        if query:
            if self._is_suspicious_query(query):
                self.auditor.log_suspicious_query(client_ip or "unknown", query, user_id)
                return False, {"error": "Query contains suspicious content"}
        
        return True, {"user_id": user_id, "permissions": key_data.get("permissions", []) if api_key else []}
    
    def _setup_default_rules(self):
        """Setup default security rules."""
        # General rate limit: 100 requests per minute
        self.rate_limiter.add_rule(RateLimitRule(
            name="general",
            max_requests=100,
            time_window=60,
            block_duration=300
        ))
        
        # Strict rate limit for suspicious IPs: 10 requests per minute
        self.rate_limiter.add_rule(RateLimitRule(
            name="strict",
            max_requests=10,
            time_window=60,
            block_duration=600
        ))
    
    def _is_suspicious_query(self, query: str) -> bool:
        """Basic suspicious query detection."""
        suspicious_patterns = [
            r"<script",
            r"javascript:",
            r"union\s+select",
            r"drop\s+table",
            r"exec\s*\(",
            r"\.\.\/",
            r"cmd\.exe",
            r"/bin/sh"
        ]
        
        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "rate_limiter": {
                "rules_count": len(self.rate_limiter.rules),
                "blocked_clients": len(self.rate_limiter.blocked_clients)
            },
            "ip_controls": {
                "whitelist_entries": len(self.ip_whitelist.whitelist),
                "blacklist_entries": len(self.ip_whitelist.blacklist)
            },
            "api_keys": {
                "total_keys": len(self.api_key_manager.keys),
                "active_keys": sum(1 for k in self.api_key_manager.keys.values() if k["active"])
            },
            "security_events": self.auditor.get_security_summary(24)
        }
