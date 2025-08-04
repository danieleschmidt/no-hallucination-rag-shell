"""
Security management for RAG system including rate limiting, authentication, and access control.
"""

import logging
import time
import hashlib
import hmac
import secrets
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import ipaddress


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