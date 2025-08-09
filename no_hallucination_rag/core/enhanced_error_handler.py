
"""
Advanced error handling and fault tolerance system.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import functools


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    NETWORK = "network"
    COMPUTATION = "computation"
    RESOURCE = "resource"
    SECURITY = "security"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ErrorEvent:
    """Detailed error event information."""
    timestamp: datetime
    error_type: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    component: str
    operation: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    occurrence_count: int = 1


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    success_threshold: int = 3  # For half-open state


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._lock = threading.Lock()
    
    def __call__(self, func):
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        
        return (datetime.utcnow() - self.last_failure_time).total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, exception: Exception):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} opened due to failure in HALF_OPEN state")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
    
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} reset to CLOSED state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ErrorPatternDetector:
    """Detect patterns in errors for proactive handling."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.error_history: deque = deque(maxlen=window_size)
        self.pattern_thresholds = {
            "burst": {"count": 5, "time_window": 60},  # 5 errors in 1 minute
            "sustained": {"count": 20, "time_window": 300},  # 20 errors in 5 minutes
            "cascade": {"different_components": 3, "time_window": 120}  # 3+ components in 2 minutes
        }
        self.logger = logging.getLogger(__name__)
    
    def record_error(self, error_event: ErrorEvent):
        """Record error event and detect patterns."""
        self.error_history.append(error_event)
        patterns = self._detect_patterns()
        
        if patterns:
            self.logger.warning(f"Error patterns detected: {patterns}")
            self._handle_patterns(patterns)
    
    def _detect_patterns(self) -> List[str]:
        """Detect error patterns in recent history."""
        patterns = []
        now = datetime.utcnow()
        
        # Check for burst pattern
        recent_errors = [e for e in self.error_history 
                        if (now - e.timestamp).total_seconds() <= self.pattern_thresholds["burst"]["time_window"]]
        if len(recent_errors) >= self.pattern_thresholds["burst"]["count"]:
            patterns.append("burst")
        
        # Check for sustained pattern
        sustained_errors = [e for e in self.error_history 
                           if (now - e.timestamp).total_seconds() <= self.pattern_thresholds["sustained"]["time_window"]]
        if len(sustained_errors) >= self.pattern_thresholds["sustained"]["count"]:
            patterns.append("sustained")
        
        # Check for cascade pattern
        cascade_errors = [e for e in self.error_history 
                         if (now - e.timestamp).total_seconds() <= self.pattern_thresholds["cascade"]["time_window"]]
        unique_components = set(e.component for e in cascade_errors)
        if len(unique_components) >= self.pattern_thresholds["cascade"]["different_components"]:
            patterns.append("cascade")
        
        return patterns
    
    def _handle_patterns(self, patterns: List[str]):
        """Handle detected error patterns."""
        for pattern in patterns:
            if pattern == "burst":
                self.logger.warning("Burst error pattern detected - temporary rate limiting recommended")
            elif pattern == "sustained":
                self.logger.error("Sustained error pattern detected - system degradation likely")
            elif pattern == "cascade":
                self.logger.critical("Cascade error pattern detected - system-wide failure possible")


class RobustErrorHandler:
    """Enhanced error handling system with patterns, circuit breakers, and recovery."""
    
    def __init__(self):
        self.error_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pattern_detector = ErrorPatternDetector()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
        # Setup default recovery strategies
        self._setup_recovery_strategies()
    
    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Add circuit breaker for a component."""
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorEvent:
        """Handle error with comprehensive analysis and recovery."""
        context = context or {}
        
        # Classify error
        category = self._classify_error(exception)
        severity = self._determine_severity(exception, category)
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.utcnow(),
            error_type=type(exception).__name__,
            category=category,
            severity=severity,
            message=str(exception),
            component=context.get("component", "unknown"),
            operation=context.get("operation", "unknown"),
            context=context,
            stack_trace=self._get_stack_trace()
        )
        
        # Record error for pattern detection
        self.pattern_detector.record_error(error_event)
        
        # Update statistics
        self._update_stats(error_event)
        
        # Attempt recovery
        self._attempt_recovery(error_event)
        
        # Log error
        self._log_error(error_event)
        
        return error_event
    
    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify error into categories."""
        error_type = type(exception).__name__.lower()
        
        if "validation" in error_type or "value" in error_type:
            return ErrorCategory.VALIDATION
        elif "network" in error_type or "connection" in error_type or "timeout" in error_type:
            return ErrorCategory.NETWORK
        elif "memory" in error_type or "resource" in error_type:
            return ErrorCategory.RESOURCE
        elif "permission" in error_type or "auth" in error_type or "security" in error_type:
            return ErrorCategory.SECURITY
        elif "computation" in error_type or "math" in error_type or "overflow" in error_type:
            return ErrorCategory.COMPUTATION
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on type and category."""
        critical_types = ["SystemExit", "MemoryError", "KeyboardInterrupt"]
        high_types = ["ConnectionError", "TimeoutError", "SecurityError"]
        
        error_type = type(exception).__name__
        
        if error_type in critical_types:
            return ErrorSeverity.CRITICAL
        elif error_type in high_types or category == ErrorCategory.SECURITY:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.NETWORK, ErrorCategory.RESOURCE]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_stack_trace(self) -> str:
        """Get formatted stack trace."""
        import traceback
        return traceback.format_exc()
    
    def _update_stats(self, error_event: ErrorEvent):
        """Update error statistics."""
        self.error_stats[error_event.component][error_event.error_type] += 1
        self.error_stats["global"][error_event.category.value] += 1
        self.error_stats["global"]["total"] += 1
    
    def _attempt_recovery(self, error_event: ErrorEvent):
        """Attempt automatic error recovery."""
        recovery_strategies = self.recovery_strategies.get(error_event.category, [])
        
        error_event.recovery_attempted = len(recovery_strategies) > 0
        
        for strategy in recovery_strategies:
            try:
                success = strategy(error_event)
                if success:
                    error_event.recovery_successful = True
                    self.logger.info(f"Recovery successful for {error_event.error_type}")
                    break
            except Exception as e:
                self.logger.warning(f"Recovery strategy failed: {e}")
    
    def _setup_recovery_strategies(self):
        """Setup default recovery strategies."""
        
        def retry_with_backoff(error_event: ErrorEvent) -> bool:
            """Generic retry with exponential backoff."""
            if "retry_count" not in error_event.context:
                error_event.context["retry_count"] = 0
            
            retry_count = error_event.context["retry_count"]
            if retry_count >= 3:
                return False
            
            wait_time = 2 ** retry_count
            time.sleep(wait_time)
            error_event.context["retry_count"] += 1
            return True
        
        def clear_cache(error_event: ErrorEvent) -> bool:
            """Clear caches on resource errors."""
            # Placeholder for cache clearing logic
            self.logger.info("Clearing caches due to resource error")
            return True
        
        def reset_connection(error_event: ErrorEvent) -> bool:
            """Reset network connections."""
            # Placeholder for connection reset logic
            self.logger.info("Resetting connections due to network error")
            return True
        
        # Register recovery strategies
        self.recovery_strategies[ErrorCategory.NETWORK].append(retry_with_backoff)
        self.recovery_strategies[ErrorCategory.NETWORK].append(reset_connection)
        self.recovery_strategies[ErrorCategory.RESOURCE].append(clear_cache)
        self.recovery_strategies[ErrorCategory.COMPUTATION].append(retry_with_backoff)
    
    def _log_error(self, error_event: ErrorEvent):
        """Log error with appropriate level."""
        log_message = (
            f"[{error_event.category.value.upper()}] {error_event.component}.{error_event.operation}: "
            f"{error_event.error_type} - {error_event.message}"
        )
        
        if error_event.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_event.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            "total_errors": self.error_stats["global"]["total"],
            "errors_by_category": dict(self.error_stats["global"]),
            "errors_by_component": {k: dict(v) for k, v in self.error_stats.items() if k != "global"},
            "circuit_breaker_states": {name: cb.get_state() for name, cb in self.circuit_breakers.items()},
            "pattern_detector_status": {
                "window_size": self.pattern_detector.window_size,
                "current_errors": len(self.pattern_detector.error_history)
            }
        }
    
    def add_recovery_strategy(self, category: ErrorCategory, strategy: Callable[[ErrorEvent], bool]):
        """Add custom recovery strategy."""
        self.recovery_strategies[category].append(strategy)


# Decorators for common error handling patterns

def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to add retry logic with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        time.sleep(wait_time)
                        logging.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} in {wait_time}s")
                    else:
                        logging.error(f"All retries failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        return wrapper
    return decorator


def safe_execute(func, *args, default=None, logger=None, **kwargs):
    """Safely execute function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return default
