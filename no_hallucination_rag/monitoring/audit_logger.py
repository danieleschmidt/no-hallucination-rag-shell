"""
Comprehensive audit logging for compliance and security monitoring.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import threading
from pathlib import Path
import gzip
import os


class AuditEventType(Enum):
    """Types of audit events."""
    USER_QUERY = "user_query"
    DOCUMENT_ACCESS = "document_access"
    FACTUALITY_CHECK = "factuality_check"
    GOVERNANCE_CHECK = "governance_check"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_MODIFICATION = "data_modification"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    PERFORMANCE_EVENT = "performance_event"
    ERROR_EVENT = "error_event"
    COMPLIANCE_EVENT = "compliance_event"


class AuditLevel(Enum):
    """Audit event levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    event_type: AuditEventType
    level: AuditLevel
    timestamp: str
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    query: Optional[str] = None
    query_hash: Optional[str] = None
    sources_accessed: Optional[List[str]] = None
    factuality_score: Optional[float] = None
    governance_compliant: Optional[bool] = None
    response_time_ms: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Generate query hash if query provided but hash not set
        if self.query and not self.query_hash:
            self.query_hash = hashlib.sha256(self.query.encode()).hexdigest()[:16]


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(
        self,
        log_file: str = "logs/audit.log",
        max_file_size_mb: int = 100,
        backup_count: int = 10,
        compress_backups: bool = True,
        enable_console_output: bool = False,
        pii_masking: bool = True
    ):
        self.log_file = Path(log_file)
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self.compress_backups = compress_backups
        self.enable_console_output = enable_console_output
        self.pii_masking = pii_masking
        
        # Create logs directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger("audit_logger")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            filename=self.log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler if enabled
        if enable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Thread-safe event counter
        self._event_counter = 0
        self._lock = threading.Lock()
        
        # PII patterns for masking
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-?\d{3}-?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        }
    
    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        try:
            # Apply PII masking if enabled
            if self.pii_masking:
                event = self._mask_pii(event)
            
            # Convert to JSON
            event_dict = asdict(event)
            event_dict['event_type'] = event.event_type.value
            event_dict['level'] = event.level.value
            
            # Log as JSON
            json_log = json.dumps(event_dict, ensure_ascii=False, separators=(',', ':'))
            
            # Log based on level
            if event.level == AuditLevel.DEBUG:
                self.logger.debug(json_log)
            elif event.level == AuditLevel.INFO:
                self.logger.info(json_log)
            elif event.level == AuditLevel.WARNING:
                self.logger.warning(json_log)
            elif event.level == AuditLevel.ERROR:
                self.logger.error(json_log)
            elif event.level == AuditLevel.CRITICAL:
                self.logger.critical(json_log)
            
        except Exception as e:
            # Fallback logging for audit failures
            fallback_logger = logging.getLogger("audit_fallback")
            fallback_logger.error(f"Audit logging failed: {e}")
    
    def log_user_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log user query event."""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.USER_QUERY,
            level=AuditLevel.INFO,
            timestamp=datetime.utcnow().isoformat(),
            component="FactualRAG",
            operation="query",
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=user_agent,
            query=query,
            metadata=metadata or {}
        )
        
        self.log_event(event)
        return event_id
    
    def log_document_access(
        self,
        sources_accessed: List[str],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        query_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log document access event."""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.DOCUMENT_ACCESS,
            level=AuditLevel.INFO,
            timestamp=datetime.utcnow().isoformat(),
            component="HybridRetriever",
            operation="retrieve_sources",
            user_id=user_id,
            session_id=session_id,
            query_hash=query_hash,
            sources_accessed=sources_accessed,
            metadata=metadata or {}
        )
        
        self.log_event(event)
        return event_id
    
    def log_factuality_check(
        self,
        query: str,
        factuality_score: float,
        claims_verified: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log factuality verification event."""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.FACTUALITY_CHECK,
            level=AuditLevel.INFO if factuality_score >= 0.8 else AuditLevel.WARNING,
            timestamp=datetime.utcnow().isoformat(),
            component="FactualityDetector",
            operation="verify_factuality",
            user_id=user_id,
            session_id=session_id,
            query=query,
            factuality_score=factuality_score,
            metadata={
                "claims_verified": claims_verified,
                **(metadata or {})
            }
        )
        
        self.log_event(event)
        return event_id
    
    def log_governance_check(
        self,
        query: str,
        governance_compliant: bool,
        policies_checked: List[str],
        violations: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log governance compliance check."""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.GOVERNANCE_CHECK,
            level=AuditLevel.INFO if governance_compliant else AuditLevel.ERROR,
            timestamp=datetime.utcnow().isoformat(),
            component="GovernanceChecker",
            operation="check_compliance",
            user_id=user_id,
            session_id=session_id,
            query=query,
            governance_compliant=governance_compliant,
            metadata={
                "policies_checked": policies_checked,
                "violations": violations or [],
                **(metadata or {})
            }
        )
        
        self.log_event(event)
        return event_id
    
    def log_security_event(
        self,
        event_description: str,
        severity: AuditLevel,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        threat_type: Optional[str] = None,
        action_taken: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log security event."""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.SECURITY_EVENT,
            level=severity,
            timestamp=datetime.utcnow().isoformat(),
            component="SecurityManager",
            operation="security_check",
            user_id=user_id,
            client_ip=client_ip,
            metadata={
                "event_description": event_description,
                "threat_type": threat_type,
                "action_taken": action_taken,
                **(metadata or {})
            }
        )
        
        self.log_event(event)
        return event_id
    
    def log_performance_event(
        self,
        operation: str,
        response_time_ms: float,
        component: str,
        success: bool,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log performance event."""
        event_id = self._generate_event_id()
        
        # Determine level based on performance
        level = AuditLevel.INFO
        if response_time_ms > 10000:  # > 10 seconds
            level = AuditLevel.WARNING
        elif response_time_ms > 30000:  # > 30 seconds
            level = AuditLevel.ERROR
        
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.PERFORMANCE_EVENT,
            level=level,
            timestamp=datetime.utcnow().isoformat(),
            component=component,
            operation=operation,
            user_id=user_id,
            session_id=session_id,
            response_time_ms=response_time_ms,
            metadata={
                "success": success,
                **(metadata or {})
            }
        )
        
        self.log_event(event)
        return event_id
    
    def log_error_event(
        self,
        error_message: str,
        error_category: str,
        component: str,
        operation: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log error event."""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.ERROR_EVENT,
            level=AuditLevel.ERROR,
            timestamp=datetime.utcnow().isoformat(),
            component=component,
            operation=operation,
            user_id=user_id,
            session_id=session_id,
            error_details={
                "message": error_message,
                "category": error_category,
                "stack_trace": stack_trace
            },
            metadata=metadata or {}
        )
        
        self.log_event(event)
        return event_id
    
    def log_data_modification(
        self,
        operation: str,
        data_type: str,
        affected_records: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log data modification event."""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.DATA_MODIFICATION,
            level=AuditLevel.INFO,
            timestamp=datetime.utcnow().isoformat(),
            component="KnowledgeBase",
            operation=operation,
            user_id=user_id,
            session_id=session_id,
            metadata={
                "data_type": data_type,
                "affected_records": affected_records,
                **(metadata or {})
            }
        )
        
        self.log_event(event)
        return event_id
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        with self._lock:
            self._event_counter += 1
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            return f"AUD-{timestamp}-{self._event_counter:06d}"
    
    def _mask_pii(self, event: AuditEvent) -> AuditEvent:
        """Mask personally identifiable information."""
        import re
        
        if event.query:
            masked_query = event.query
            for pii_type, pattern in self.pii_patterns.items():
                masked_query = re.sub(pattern, f"[MASKED_{pii_type.upper()}]", masked_query)
            event.query = masked_query
        
        # Mask PII in metadata
        if event.metadata:
            event.metadata = self._mask_dict_pii(event.metadata)
        
        return event
    
    def _mask_dict_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask PII in dictionary."""
        import re
        
        masked_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                masked_value = value
                for pii_type, pattern in self.pii_patterns.items():
                    masked_value = re.sub(pattern, f"[MASKED_{pii_type.upper()}]", masked_value)
                masked_data[key] = masked_value
            elif isinstance(value, dict):
                masked_data[key] = self._mask_dict_pii(value)
            elif isinstance(value, list):
                masked_data[key] = [
                    self._mask_dict_pii(item) if isinstance(item, dict)
                    else re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[MASKED_EMAIL]', str(item))
                    for item in value
                ]
            else:
                masked_data[key] = value
        
        return masked_data
    
    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        try:
            file_size = self.log_file.stat().st_size if self.log_file.exists() else 0
            
            return {
                "log_file": str(self.log_file),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "max_file_size_mb": self.max_file_size_mb,
                "backup_count": self.backup_count,
                "compress_backups": self.compress_backups,
                "pii_masking_enabled": self.pii_masking,
                "events_logged": self._event_counter
            }
        except Exception as e:
            return {"error": str(e)}
    
    def search_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search audit events (basic implementation)."""
        try:
            events = []
            
            if not self.log_file.exists():
                return events
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        
                        # Apply filters
                        if event_type and event.get('event_type') != event_type.value:
                            continue
                        
                        if user_id and event.get('user_id') != user_id:
                            continue
                        
                        if start_time and event.get('timestamp', '') < start_time:
                            continue
                        
                        if end_time and event.get('timestamp', '') > end_time:
                            continue
                        
                        events.append(event)
                        
                        if len(events) >= limit:
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            return events
            
        except Exception as e:
            return [{"error": f"Search failed: {e}"}]
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Clean up old log files."""
        try:
            log_dir = self.log_file.parent
            cutoff_time = datetime.utcnow().timestamp() - (days_to_keep * 24 * 3600)
            
            deleted_files = []
            total_size_deleted = 0
            
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    deleted_files.append(str(log_file))
                    total_size_deleted += file_size
            
            return {
                "deleted_files": len(deleted_files),
                "total_size_deleted_mb": round(total_size_deleted / (1024 * 1024), 2),
                "files": deleted_files
            }
            
        except Exception as e:
            return {"error": f"Cleanup failed: {e}"}


# Global audit logger instance
audit_logger = AuditLogger()


def audit_event(
    event_type: AuditEventType,
    component: str,
    operation: str,
    level: AuditLevel = AuditLevel.INFO,
    **kwargs
):
    """Decorator for automatic audit logging of function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            start_time = datetime.utcnow()
            success = True
            error_message = None
            
            try:
                result = func(*args, **func_kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                # Calculate execution time
                end_time = datetime.utcnow()
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                
                # Log audit event
                event_id = audit_logger._generate_event_id()
                
                event = AuditEvent(
                    event_id=event_id,
                    event_type=event_type,
                    level=level if success else AuditLevel.ERROR,
                    timestamp=start_time.isoformat(),
                    component=component,
                    operation=operation,
                    response_time_ms=response_time_ms,
                    error_details={"message": error_message} if error_message else None,
                    metadata={
                        "success": success,
                        "function_name": func.__name__,
                        **kwargs
                    }
                )
                
                audit_logger.log_event(event)
        
        return wrapper
    return decorator