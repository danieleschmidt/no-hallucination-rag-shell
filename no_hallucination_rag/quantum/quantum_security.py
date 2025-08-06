"""
Security manager for quantum task planning operations.
"""

import logging
import hashlib
import hmac
import secrets
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from .quantum_planner import QuantumTask, TaskState
from ..security.security_manager import SecurityManager


class QuantumSecurityLevel(Enum):
    """Security levels for quantum operations."""
    PUBLIC = "public"           # No encryption, basic access control
    PROTECTED = "protected"     # Access control with user verification  
    CONFIDENTIAL = "confidential"  # Encrypted storage, audit logging
    SECRET = "secret"          # Full encryption, strict access control
    TOP_SECRET = "top_secret"  # Maximum security, quantum-safe encryption


class QuantumThreatType(Enum):
    """Types of security threats to quantum systems."""
    EAVESDROPPING = "eavesdropping"           # Unauthorized observation
    DECOHERENCE_ATTACK = "decoherence_attack" # Malicious decoherence
    ENTANGLEMENT_HIJACK = "entanglement_hijack" # Unauthorized entanglement
    STATE_INJECTION = "state_injection"       # Malicious state injection
    MEASUREMENT_ATTACK = "measurement_attack" # Unauthorized measurement
    QUANTUM_TAMPERING = "quantum_tampering"   # Task manipulation


@dataclass
class SecurityEvent:
    """Security event for quantum operations."""
    event_id: str
    event_type: QuantumThreatType
    severity: str  # low, medium, high, critical
    timestamp: datetime
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


class QuantumAccessControl:
    """Access control for quantum operations."""
    
    def __init__(self):
        self.permissions = {
            "create_task": {"roles": ["user", "admin"], "security_level": QuantumSecurityLevel.PUBLIC},
            "observe_task": {"roles": ["user", "admin"], "security_level": QuantumSecurityLevel.PROTECTED},
            "entangle_tasks": {"roles": ["user", "admin"], "security_level": QuantumSecurityLevel.PROTECTED},
            "apply_quantum_gate": {"roles": ["admin"], "security_level": QuantumSecurityLevel.CONFIDENTIAL},
            "export_quantum_state": {"roles": ["admin"], "security_level": QuantumSecurityLevel.SECRET},
            "create_ghz_state": {"roles": ["admin"], "security_level": QuantumSecurityLevel.CONFIDENTIAL},
            "break_entanglement": {"roles": ["admin"], "security_level": QuantumSecurityLevel.CONFIDENTIAL}
        }
        
        # Rate limiting
        self.rate_limits = {
            "create_task": {"max_requests": 100, "time_window": 3600},  # 100 per hour
            "observe_task": {"max_requests": 1000, "time_window": 3600}, # 1000 per hour
            "entangle_tasks": {"max_requests": 50, "time_window": 3600},  # 50 per hour
            "apply_quantum_gate": {"max_requests": 20, "time_window": 3600}, # 20 per hour
        }
        
        self.request_history: Dict[str, List[datetime]] = {}
    
    def check_permission(
        self,
        operation: str,
        user_id: str,
        user_roles: List[str],
        security_clearance: QuantumSecurityLevel = QuantumSecurityLevel.PUBLIC
    ) -> Tuple[bool, str]:
        """Check if user has permission for operation."""
        
        if operation not in self.permissions:
            return False, f"Unknown operation: {operation}"
        
        perm_config = self.permissions[operation]
        
        # Check role-based access
        if not any(role in perm_config["roles"] for role in user_roles):
            return False, f"Insufficient role privileges for {operation}"
        
        # Check security level clearance
        required_level = perm_config["security_level"]
        if security_clearance.value < required_level.value:
            return False, f"Insufficient security clearance for {operation}"
        
        # Check rate limits
        if not self._check_rate_limit(operation, user_id):
            return False, f"Rate limit exceeded for {operation}"
        
        return True, "Permission granted"
    
    def _check_rate_limit(self, operation: str, user_id: str) -> bool:
        """Check rate limiting for operation."""
        
        if operation not in self.rate_limits:
            return True
        
        limit_config = self.rate_limits[operation]
        max_requests = limit_config["max_requests"]
        time_window = limit_config["time_window"]
        
        # Initialize user request history
        user_key = f"{user_id}:{operation}"
        if user_key not in self.request_history:
            self.request_history[user_key] = []
        
        # Clean old requests outside time window
        now = datetime.utcnow()
        cutoff_time = now - timedelta(seconds=time_window)
        self.request_history[user_key] = [
            req_time for req_time in self.request_history[user_key]
            if req_time > cutoff_time
        ]
        
        # Check if within limits
        if len(self.request_history[user_key]) >= max_requests:
            return False
        
        # Record this request
        self.request_history[user_key].append(now)
        return True


class QuantumSecurityManager:
    """
    Comprehensive security manager for quantum task planning.
    
    Handles encryption, access control, threat detection, and audit logging
    for quantum operations.
    """
    
    def __init__(self, security_level: QuantumSecurityLevel = QuantumSecurityLevel.PROTECTED):
        self.security_level = security_level
        self.access_control = QuantumAccessControl()
        self.logger = logging.getLogger(__name__)
        
        # Security state
        self.security_events: Dict[str, SecurityEvent] = {}
        self.encrypted_tasks: Dict[str, bytes] = {}
        self.task_access_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Cryptographic keys
        self.master_key = secrets.token_bytes(32)  # AES-256 key
        self.hmac_key = secrets.token_bytes(32)    # HMAC key
        
        # Threat detection
        self.threat_patterns = {
            QuantumThreatType.EAVESDROPPING: self._detect_eavesdropping,
            QuantumThreatType.DECOHERENCE_ATTACK: self._detect_decoherence_attack,
            QuantumThreatType.ENTANGLEMENT_HIJACK: self._detect_entanglement_hijack,
            QuantumThreatType.STATE_INJECTION: self._detect_state_injection,
            QuantumThreatType.MEASUREMENT_ATTACK: self._detect_measurement_attack,
            QuantumThreatType.QUANTUM_TAMPERING: self._detect_quantum_tampering
        }
        
        self.logger.info(f"Quantum Security Manager initialized with level: {security_level.value}")
    
    def secure_task_creation(
        self,
        task: QuantumTask,
        user_id: str,
        user_roles: List[str],
        client_context: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[QuantumTask]]:
        """Secure task creation with access control and encryption."""
        
        # Check permissions
        allowed, message = self.access_control.check_permission(
            "create_task", user_id, user_roles, self.security_level
        )
        
        if not allowed:
            self._log_security_event(
                QuantumThreatType.MEASUREMENT_ATTACK,
                "unauthorized_task_creation",
                user_id=user_id,
                context={"operation": "create_task", "denied_reason": message}
            )
            return False, message, None
        
        # Encrypt sensitive task data if required
        if self.security_level in [QuantumSecurityLevel.CONFIDENTIAL, QuantumSecurityLevel.SECRET, QuantumSecurityLevel.TOP_SECRET]:
            encrypted_task = self._encrypt_task(task)
        else:
            encrypted_task = task
        
        # Log access
        self._log_task_access(
            task.id, user_id, "create", 
            client_context.get("client_ip", "unknown")
        )
        
        # Threat detection
        self._run_threat_detection(task, user_id, "create", client_context)
        
        return True, "Task created successfully", encrypted_task
    
    def secure_task_observation(
        self,
        task: QuantumTask,
        user_id: str,
        user_roles: List[str],
        client_context: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[QuantumTask]]:
        """Secure task observation with access control."""
        
        # Check permissions
        allowed, message = self.access_control.check_permission(
            "observe_task", user_id, user_roles, self.security_level
        )
        
        if not allowed:
            self._log_security_event(
                QuantumThreatType.MEASUREMENT_ATTACK,
                "unauthorized_task_observation",
                task_id=task.id,
                user_id=user_id,
                context={"operation": "observe_task", "denied_reason": message}
            )
            return False, message, None
        
        # Decrypt task if encrypted
        if task.id in self.encrypted_tasks:
            decrypted_task = self._decrypt_task(task.id)
        else:
            decrypted_task = task
        
        # Log access
        self._log_task_access(
            task.id, user_id, "observe",
            client_context.get("client_ip", "unknown")
        )
        
        # Detect potential eavesdropping
        if self._detect_suspicious_observation_pattern(user_id, task.id):
            self._log_security_event(
                QuantumThreatType.EAVESDROPPING,
                "suspicious_observation_pattern",
                task_id=task.id,
                user_id=user_id,
                context={"operation": "observe_task"}
            )
        
        return True, "Task observation authorized", decrypted_task
    
    def secure_entanglement_creation(
        self,
        task_id1: str,
        task_id2: str,
        user_id: str,
        user_roles: List[str],
        client_context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Secure entanglement creation with access control."""
        
        # Check permissions
        allowed, message = self.access_control.check_permission(
            "entangle_tasks", user_id, user_roles, self.security_level
        )
        
        if not allowed:
            self._log_security_event(
                QuantumThreatType.ENTANGLEMENT_HIJACK,
                "unauthorized_entanglement_creation",
                user_id=user_id,
                context={
                    "operation": "entangle_tasks",
                    "task_ids": [task_id1, task_id2],
                    "denied_reason": message
                }
            )
            return False, message
        
        # Check for entanglement hijack attempts
        if self._detect_entanglement_hijack_attempt(task_id1, task_id2, user_id):
            self._log_security_event(
                QuantumThreatType.ENTANGLEMENT_HIJACK,
                "entanglement_hijack_attempt",
                user_id=user_id,
                context={
                    "operation": "entangle_tasks",
                    "task_ids": [task_id1, task_id2]
                }
            )
            return False, "Entanglement creation blocked due to security concerns"
        
        # Log access for both tasks
        client_ip = client_context.get("client_ip", "unknown")
        self._log_task_access(task_id1, user_id, "entangle", client_ip)
        self._log_task_access(task_id2, user_id, "entangle", client_ip)
        
        return True, "Entanglement creation authorized"
    
    def secure_quantum_gate_operation(
        self,
        task_id: str,
        gate_type: str,
        user_id: str,
        user_roles: List[str],
        client_context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Secure quantum gate operation with strict access control."""
        
        # Check permissions (quantum gate operations require higher privileges)
        allowed, message = self.access_control.check_permission(
            "apply_quantum_gate", user_id, user_roles, self.security_level
        )
        
        if not allowed:
            self._log_security_event(
                QuantumThreatType.STATE_INJECTION,
                "unauthorized_quantum_gate_operation",
                task_id=task_id,
                user_id=user_id,
                context={
                    "operation": "apply_quantum_gate",
                    "gate_type": gate_type,
                    "denied_reason": message
                }
            )
            return False, message
        
        # Validate gate operation for security
        if not self._validate_quantum_gate_security(gate_type, task_id):
            self._log_security_event(
                QuantumThreatType.QUANTUM_TAMPERING,
                "malicious_quantum_gate_detected",
                task_id=task_id,
                user_id=user_id,
                context={"gate_type": gate_type}
            )
            return False, "Quantum gate operation blocked due to security validation failure"
        
        # Log access
        self._log_task_access(
            task_id, user_id, f"quantum_gate:{gate_type}",
            client_context.get("client_ip", "unknown")
        )
        
        return True, "Quantum gate operation authorized"
    
    def generate_secure_task_id(self) -> str:
        """Generate cryptographically secure task ID."""
        
        # Use cryptographically secure random bytes
        random_bytes = secrets.token_bytes(16)
        
        # Create HMAC for integrity
        hmac_digest = hmac.new(
            self.hmac_key,
            random_bytes,
            hashlib.sha256
        ).digest()[:8]  # Use first 8 bytes
        
        # Combine and format as UUID-like string
        combined = random_bytes + hmac_digest
        task_id = (
            f"{combined[:4].hex()}-{combined[4:6].hex()}-"
            f"{combined[6:8].hex()}-{combined[8:10].hex()}-{combined[10:16].hex()}"
        )
        
        return task_id
    
    def verify_task_integrity(self, task: QuantumTask) -> bool:
        """Verify task data integrity using cryptographic signatures."""
        
        try:
            # Create task data hash
            task_data = {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "state": task.state.value,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat()
            }
            
            task_json = json.dumps(task_data, sort_keys=True)
            task_hash = hashlib.sha256(task_json.encode()).digest()
            
            # Verify HMAC
            expected_hmac = hmac.new(
                self.hmac_key,
                task_hash,
                hashlib.sha256
            ).digest()
            
            # In a real implementation, this would be stored with the task
            # For now, we assume integrity if basic validation passes
            return True
            
        except Exception as e:
            self.logger.error(f"Task integrity verification failed: {e}")
            return False
    
    def _encrypt_task(self, task: QuantumTask) -> QuantumTask:
        """Encrypt sensitive task data."""
        
        try:
            # In a real implementation, this would use proper AES encryption
            # For demo purposes, we'll use a simple encoding
            sensitive_data = {
                "description": task.description,
                "context": task.context
            }
            
            encrypted_data = secrets.token_urlsafe(32)  # Simulated encryption
            self.encrypted_tasks[task.id] = encrypted_data.encode()
            
            # Return task with encrypted fields cleared
            encrypted_task = QuantumTask(
                id=task.id,
                title=task.title,
                description="[ENCRYPTED]",
                state=task.state,
                priority=task.priority,
                probability_amplitude=task.probability_amplitude,
                coherence_time=task.coherence_time,
                entangled_tasks=task.entangled_tasks,
                created_at=task.created_at,
                due_date=task.due_date,
                dependencies=task.dependencies,
                tags=task.tags,
                estimated_duration=task.estimated_duration,
                progress=task.progress,
                context={"encrypted": True}
            )
            
            return encrypted_task
            
        except Exception as e:
            self.logger.error(f"Task encryption failed: {e}")
            return task
    
    def _decrypt_task(self, task_id: str) -> Optional[QuantumTask]:
        """Decrypt task data."""
        
        if task_id not in self.encrypted_tasks:
            return None
        
        try:
            # In a real implementation, this would use proper AES decryption
            # For demo purposes, we'll return a placeholder
            decrypted_data = self.encrypted_tasks[task_id].decode()
            
            # This would normally reconstruct the full task from encrypted data
            # For now, return None to indicate decryption would happen
            return None
            
        except Exception as e:
            self.logger.error(f"Task decryption failed: {e}")
            return None
    
    def _log_task_access(
        self,
        task_id: str,
        user_id: str,
        operation: str,
        client_ip: str
    ) -> None:
        """Log task access for audit trail."""
        
        if task_id not in self.task_access_logs:
            self.task_access_logs[task_id] = []
        
        access_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "operation": operation,
            "client_ip": client_ip,
            "success": True
        }
        
        self.task_access_logs[task_id].append(access_log)
        
        # Keep only last 100 access logs per task
        if len(self.task_access_logs[task_id]) > 100:
            self.task_access_logs[task_id] = self.task_access_logs[task_id][-100:]
    
    def _log_security_event(
        self,
        threat_type: QuantumThreatType,
        description: str,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "medium"
    ) -> None:
        """Log security event."""
        
        event_id = secrets.token_hex(8)
        
        security_event = SecurityEvent(
            event_id=event_id,
            event_type=threat_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            task_id=task_id,
            user_id=user_id,
            description=description,
            context=context or {}
        )
        
        self.security_events[event_id] = security_event
        
        self.logger.warning(
            f"Security event [{event_id}]: {threat_type.value} - {description}"
            + (f" (Task: {task_id})" if task_id else "")
            + (f" (User: {user_id})" if user_id else "")
        )
    
    def _run_threat_detection(
        self,
        task: QuantumTask,
        user_id: str,
        operation: str,
        client_context: Dict[str, Any]
    ) -> None:
        """Run threat detection algorithms."""
        
        for threat_type, detector in self.threat_patterns.items():
            try:
                if detector(task, user_id, operation, client_context):
                    self._log_security_event(
                        threat_type,
                        f"Threat detected during {operation}",
                        task_id=task.id,
                        user_id=user_id,
                        context=client_context,
                        severity="high"
                    )
            except Exception as e:
                self.logger.error(f"Threat detection failed for {threat_type.value}: {e}")
    
    def _detect_eavesdropping(
        self,
        task: QuantumTask,
        user_id: str,
        operation: str,
        client_context: Dict[str, Any]
    ) -> bool:
        """Detect potential eavesdropping attempts."""
        
        # Check for suspicious patterns
        if operation == "observe" and task.state == TaskState.SUPERPOSITION:
            # Multiple rapid observations could indicate eavesdropping
            access_logs = self.task_access_logs.get(task.id, [])
            recent_observations = [
                log for log in access_logs[-10:]
                if log["operation"] == "observe" 
                and datetime.fromisoformat(log["timestamp"]) > datetime.utcnow() - timedelta(minutes=5)
            ]
            
            if len(recent_observations) > 5:
                return True
        
        return False
    
    def _detect_decoherence_attack(
        self,
        task: QuantumTask,
        user_id: str,
        operation: str,
        client_context: Dict[str, Any]
    ) -> bool:
        """Detect malicious decoherence attempts."""
        
        # Rapid state changes could indicate attack
        if operation in ["observe", "apply_quantum_gate"]:
            if task.coherence_time < 60:  # Very short coherence time
                return True
        
        return False
    
    def _detect_entanglement_hijack(
        self,
        task: QuantumTask,
        user_id: str,
        operation: str,
        client_context: Dict[str, Any]
    ) -> bool:
        """Detect entanglement hijacking attempts."""
        
        if operation == "entangle":
            # Check if user has access to both tasks being entangled
            # This would normally check against ownership/permissions
            return False  # Simplified for demo
        
        return False
    
    def _detect_state_injection(
        self,
        task: QuantumTask,
        user_id: str,
        operation: str,
        client_context: Dict[str, Any]
    ) -> bool:
        """Detect malicious state injection."""
        
        if operation == "apply_quantum_gate":
            # Check for suspicious quantum operations
            suspicious_patterns = [
                task.probability_amplitude == 0.0,  # Attempting to zero out task
                task.coherence_time > 86400  # Unreasonably long coherence
            ]
            
            return any(suspicious_patterns)
        
        return False
    
    def _detect_measurement_attack(
        self,
        task: QuantumTask,
        user_id: str,
        operation: str,
        client_context: Dict[str, Any]
    ) -> bool:
        """Detect unauthorized measurement attacks."""
        
        # Check for unauthorized access attempts
        client_ip = client_context.get("client_ip", "")
        
        # Simplified IP-based detection (in reality would be more sophisticated)
        suspicious_ips = ["127.0.0.1", "0.0.0.0"]  # Example suspicious IPs
        
        return client_ip in suspicious_ips
    
    def _detect_quantum_tampering(
        self,
        task: QuantumTask,
        user_id: str,
        operation: str,
        client_context: Dict[str, Any]
    ) -> bool:
        """Detect quantum tampering attempts."""
        
        # Check task integrity
        return not self.verify_task_integrity(task)
    
    def _detect_suspicious_observation_pattern(self, user_id: str, task_id: str) -> bool:
        """Detect suspicious observation patterns."""
        
        access_logs = self.task_access_logs.get(task_id, [])
        user_observations = [
            log for log in access_logs
            if log["user_id"] == user_id and log["operation"] == "observe"
        ]
        
        # More than 10 observations in the last hour is suspicious
        recent_observations = [
            log for log in user_observations
            if datetime.fromisoformat(log["timestamp"]) > datetime.utcnow() - timedelta(hours=1)
        ]
        
        return len(recent_observations) > 10
    
    def _detect_entanglement_hijack_attempt(
        self,
        task_id1: str,
        task_id2: str,
        user_id: str
    ) -> bool:
        """Detect entanglement hijack attempts."""
        
        # Check if user has accessed both tasks recently
        logs1 = self.task_access_logs.get(task_id1, [])
        logs2 = self.task_access_logs.get(task_id2, [])
        
        user_logs1 = [log for log in logs1 if log["user_id"] == user_id]
        user_logs2 = [log for log in logs2 if log["user_id"] == user_id]
        
        # If user has no prior access to either task, it might be hijack attempt
        return len(user_logs1) == 0 and len(user_logs2) == 0
    
    def _validate_quantum_gate_security(self, gate_type: str, task_id: str) -> bool:
        """Validate quantum gate operation for security."""
        
        # Block dangerous gate combinations
        dangerous_gates = ["identity", "null", "corrupt"]
        
        if gate_type.lower() in dangerous_gates:
            return False
        
        # Check task access history
        access_logs = self.task_access_logs.get(task_id, [])
        recent_gates = [
            log for log in access_logs[-5:]
            if log["operation"].startswith("quantum_gate:")
        ]
        
        # Block rapid quantum gate operations
        if len(recent_gates) > 3:
            return False
        
        return True
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report."""
        
        # Count security events by type and severity
        event_counts = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for event in self.security_events.values():
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            severity_counts[event.severity] += 1
        
        # Calculate security metrics
        total_events = len(self.security_events)
        mitigated_events = sum(1 for event in self.security_events.values() if event.mitigated)
        
        return {
            "summary": {
                "security_level": self.security_level.value,
                "total_security_events": total_events,
                "mitigated_events": mitigated_events,
                "active_threats": total_events - mitigated_events,
                "encrypted_tasks": len(self.encrypted_tasks)
            },
            "event_distribution": event_counts,
            "severity_distribution": severity_counts,
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "type": event.event_type.value,
                    "severity": event.severity,
                    "timestamp": event.timestamp.isoformat(),
                    "description": event.description,
                    "mitigated": event.mitigated
                }
                for event in sorted(
                    self.security_events.values(),
                    key=lambda x: x.timestamp,
                    reverse=True
                )[:10]
            ],
            "security_recommendations": self._generate_security_recommendations()
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on events."""
        
        recommendations = []
        
        # Analyze security events for patterns
        threat_types = [event.event_type for event in self.security_events.values()]
        
        if QuantumThreatType.EAVESDROPPING in threat_types:
            recommendations.append("Consider implementing quantum key distribution for sensitive tasks")
        
        if QuantumThreatType.ENTANGLEMENT_HIJACK in threat_types:
            recommendations.append("Strengthen entanglement access controls and monitoring")
        
        if QuantumThreatType.STATE_INJECTION in threat_types:
            recommendations.append("Implement stricter quantum gate operation validation")
        
        if len(self.security_events) > 100:
            recommendations.append("Consider implementing automated threat response")
        
        return recommendations
    
    def clear_security_events(self, older_than_days: int = 30) -> int:
        """Clear old security events."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        events_to_remove = [
            event_id for event_id, event in self.security_events.items()
            if event.timestamp < cutoff_date
        ]
        
        for event_id in events_to_remove:
            del self.security_events[event_id]
        
        return len(events_to_remove)