"""
Centralized error handling and recovery strategies.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
import functools
import threading
from collections import defaultdict, deque
import asyncio


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    RETRIEVAL = "retrieval"
    PROCESSING = "processing"
    EXTERNAL_API = "external_api"
    STORAGE = "storage"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    user_query: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorDetails:
    """Detailed error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    context: Optional[ErrorContext] = None
    recovery_suggestions: Optional[List[str]] = None
    user_message: Optional[str] = None
    should_retry: bool = False
    retry_delay: float = 0.0


class RAGException(Exception):
    """Base exception for RAG system."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recovery_suggestions: Optional[List[str]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context
        self.recovery_suggestions = recovery_suggestions or []
        self.user_message = user_message or message


class ValidationError(RAGException):
    """Input validation error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_message="Please check your input and try again."
        )


class RetrievalError(RAGException):
    """Source retrieval error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, should_retry: bool = True):
        super().__init__(
            message=message,
            category=ErrorCategory.RETRIEVAL,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestions=["Try a different query", "Check knowledge base availability"],
            user_message="Unable to retrieve sources. Please try rephrasing your query."
        )


class ProcessingError(RAGException):
    """Processing/computation error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestions=["Retry with simplified query", "Check system resources"],
            user_message="Processing error occurred. Please try again."
        )


class RateLimitError(RAGException):
    """Rate limiting error."""
    
    def __init__(self, message: str, retry_delay: float = 60.0, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_message=f"Rate limit exceeded. Please wait {retry_delay} seconds before trying again."
        )


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_count: Dict[ErrorCategory, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        
        # Initialize recovery strategies
        self._setup_recovery_strategies()
    
    def handle_error(
        self,
        error: Union[Exception, ErrorDetails],
        context: Optional[ErrorContext] = None
    ) -> ErrorDetails:
        """
        Handle error with appropriate recovery strategy.
        
        Args:
            error: Exception or ErrorDetails to handle
            context: Optional error context
            
        Returns:
            ErrorDetails with recovery information
        """
        try:
            # Convert exception to ErrorDetails if needed
            if isinstance(error, Exception):
                error_details = self._exception_to_error_details(error, context)
            else:
                error_details = error
            
            # Log error
            self._log_error(error_details)
            
            # Update error statistics
            self._update_error_stats(error_details)
            
            # Apply recovery strategies
            self._apply_recovery_strategies(error_details)
            
            return error_details
            
        except Exception as e:
            # Fallback error handling
            self.logger.critical(f"Error in error handler: {e}")
            return ErrorDetails(
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                message="Critical system error in error handler",
                original_exception=e
            )
    
    def _exception_to_error_details(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None
    ) -> ErrorDetails:
        """Convert exception to ErrorDetails."""
        
        if isinstance(exception, RAGException):
            return ErrorDetails(
                category=exception.category,
                severity=exception.severity,
                message=str(exception),
                original_exception=exception,
                context=exception.context or context,
                recovery_suggestions=exception.recovery_suggestions,
                user_message=exception.user_message
            )
        
        # Map common exceptions to categories
        exception_mapping = {
            ValueError: ErrorCategory.VALIDATION,
            TypeError: ErrorCategory.VALIDATION,
            KeyError: ErrorCategory.PROCESSING,
            AttributeError: ErrorCategory.PROCESSING,
            FileNotFoundError: ErrorCategory.STORAGE,
            MemoryError: ErrorCategory.SYSTEM,
            TimeoutError: ErrorCategory.NETWORK,
            ConnectionError: ErrorCategory.NETWORK,
        }
        
        category = ErrorCategory.SYSTEM
        severity = ErrorSeverity.MEDIUM
        
        for exc_type, exc_category in exception_mapping.items():
            if isinstance(exception, exc_type):
                category = exc_category
                break
        
        # Determine severity based on exception type
        if isinstance(exception, (MemoryError, SystemError)):
            severity = ErrorSeverity.CRITICAL
        elif isinstance(exception, (TimeoutError, ConnectionError)):
            severity = ErrorSeverity.HIGH
        
        return ErrorDetails(
            category=category,
            severity=severity,
            message=str(exception),
            original_exception=exception,
            context=context,
            user_message="An error occurred while processing your request."
        )
    
    def _log_error(self, error_details: ErrorDetails) -> None:
        """Log error with appropriate level."""
        
        log_message = f"[{error_details.category.value.upper()}] {error_details.message}"
        
        if error_details.context:
            context_info = []
            if error_details.context.user_query:
                context_info.append(f"query='{error_details.context.user_query[:100]}'")
            if error_details.context.component:
                context_info.append(f"component={error_details.context.component}")
            if error_details.context.operation:
                context_info.append(f"operation={error_details.context.operation}")
            
            if context_info:
                log_message += f" | Context: {', '.join(context_info)}"
        
        if error_details.original_exception:
            log_message += f" | Exception: {type(error_details.original_exception).__name__}"
        
        # Log based on severity
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=error_details.original_exception)
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=error_details.original_exception)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _update_error_stats(self, error_details: ErrorDetails) -> None:
        """Update error statistics."""
        category = error_details.category
        self.error_count[category] = self.error_count.get(category, 0) + 1
        
        # Log periodic statistics
        total_errors = sum(self.error_count.values())
        if total_errors % 100 == 0:
            self.logger.info(f"Error statistics: {dict(self.error_count)}")
    
    def _apply_recovery_strategies(self, error_details: ErrorDetails) -> None:
        """Apply recovery strategies for error category."""
        strategies = self.recovery_strategies.get(error_details.category, [])
        
        for strategy in strategies:
            try:
                strategy(error_details)
            except Exception as e:
                self.logger.warning(f"Recovery strategy failed: {e}")
    
    def _setup_recovery_strategies(self) -> None:
        """Setup recovery strategies for different error categories."""
        
        self.recovery_strategies = {
            ErrorCategory.VALIDATION: [
                self._log_validation_error,
            ],
            ErrorCategory.RETRIEVAL: [
                self._log_retrieval_error,
                self._suggest_query_alternatives,
            ],
            ErrorCategory.PROCESSING: [
                self._log_processing_error,
            ],
            ErrorCategory.NETWORK: [
                self._log_network_error,
                self._suggest_retry,
            ],
            ErrorCategory.RATE_LIMIT: [
                self._log_rate_limit_error,
                self._suggest_backoff,
            ],
        }
    
    def _log_validation_error(self, error_details: ErrorDetails) -> None:
        """Log validation error details."""
        if error_details.context and error_details.context.user_query:
            self.logger.info(f"Invalid input detected: {error_details.context.user_query[:200]}")
    
    def _log_retrieval_error(self, error_details: ErrorDetails) -> None:
        """Log retrieval error details."""
        self.logger.info("Retrieval failure - checking knowledge base availability")
    
    def _log_processing_error(self, error_details: ErrorDetails) -> None:
        """Log processing error details."""
        self.logger.info("Processing failure - system resource check recommended")
    
    def _log_network_error(self, error_details: ErrorDetails) -> None:
        """Log network error details."""
        self.logger.info("Network error - connectivity issues detected")
        error_details.should_retry = True
        error_details.retry_delay = 30.0
    
    def _log_rate_limit_error(self, error_details: ErrorDetails) -> None:
        """Log rate limit error details."""
        self.logger.info("Rate limit exceeded - implementing backoff strategy")
        error_details.should_retry = True
        error_details.retry_delay = 60.0
    
    def _suggest_query_alternatives(self, error_details: ErrorDetails) -> None:
        """Suggest query alternatives for retrieval errors."""
        if not error_details.recovery_suggestions:
            error_details.recovery_suggestions = []
        
        error_details.recovery_suggestions.extend([
            "Try using different keywords",
            "Make your query more specific",
            "Check if the topic is covered in available knowledge bases"
        ])
    
    def _suggest_retry(self, error_details: ErrorDetails) -> None:
        """Suggest retry for transient errors."""
        if not error_details.recovery_suggestions:
            error_details.recovery_suggestions = []
        
        error_details.recovery_suggestions.append("Please try again in a few moments")
        error_details.should_retry = True
        error_details.retry_delay = 5.0
    
    def _suggest_backoff(self, error_details: ErrorDetails) -> None:
        """Suggest backoff for rate-limited errors."""
        if not error_details.recovery_suggestions:
            error_details.recovery_suggestions = []
        
        error_details.recovery_suggestions.append("Please wait before making another request")
        error_details.should_retry = True
        error_details.retry_delay = max(error_details.retry_delay, 60.0)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": dict(self.error_count),
            "total_errors": sum(self.error_count.values()),
            "most_common_error": max(self.error_count.items(), key=lambda x: x[1])[0].value if self.error_count else None
        }


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions on error.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to retry on
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise e
                    
                    logger = logging.getLogger(func.__module__)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    error_handler: ErrorHandler,
    context: Optional[ErrorContext] = None,
    fallback_result: Any = None
) -> Any:
    """
    Safely execute function with error handling.
    
    Args:
        func: Function to execute
        error_handler: Error handler instance
        context: Error context
        fallback_result: Result to return on error
        
    Returns:
        Function result or fallback result on error
    """
    try:
        return func()
    except Exception as e:
        error_details = error_handler.handle_error(e, context)
        
        if error_details.should_retry:
            # Could implement retry logic here
            pass
        
        return fallback_result


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Failures to trigger open state
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    success_threshold: int = 2           # Successes to close circuit
    timeout: float = 30.0                # Request timeout
    sliding_window_size: int = 10        # Size of sliding window for failure tracking


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        
        # Sliding window for failure tracking
        self.failure_window = deque(maxlen=self.config.sliding_window_size)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = defaultdict(int)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: Original function exceptions
        """
        with self._lock:
            self.total_requests += 1
            
            # Check if circuit should be opened
            if self.state == CircuitBreakerState.CLOSED:
                if self._should_open_circuit():
                    self._open_circuit()
            
            # If circuit is open, check if we should try half-open
            elif self.state == CircuitBreakerState.OPEN:
                if self._should_try_half_open():
                    self._half_open_circuit()
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is open. "
                        f"Try again in {self._time_until_half_open():.1f} seconds."
                    )
        
        # Execute the function
        try:
            start_time = time.time()
            
            # Apply timeout if configured
            if self.config.timeout > 0:
                result = self._execute_with_timeout(func, args, kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise
    
    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with timeout."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.config.timeout)
            except concurrent.futures.TimeoutError:
                raise ProcessingError(
                    f"Function {func.__name__} timed out after {self.config.timeout}s"
                )
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened."""
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate in sliding window
        if len(self.failure_window) >= self.config.sliding_window_size:
            failure_rate = sum(self.failure_window) / len(self.failure_window)
            return failure_rate > 0.5  # 50% failure rate
        
        return False
    
    def _should_try_half_open(self) -> bool:
        """Check if we should try half-open state."""
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _time_until_half_open(self) -> float:
        """Time until we can try half-open."""
        return max(0, self.config.recovery_timeout - (time.time() - self.last_failure_time))
    
    def _open_circuit(self) -> None:
        """Open the circuit."""
        self.state = CircuitBreakerState.OPEN
        self.last_failure_time = time.time()
        self.state_changes[CircuitBreakerState.OPEN] += 1
        self.logger.warning(f"Circuit breaker '{self.name}' opened due to {self.failure_count} failures")
    
    def _half_open_circuit(self) -> None:
        """Set circuit to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.state_changes[CircuitBreakerState.HALF_OPEN] += 1
        self.logger.info(f"Circuit breaker '{self.name}' trying half-open state")
    
    def _close_circuit(self) -> None:
        """Close the circuit."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.state_changes[CircuitBreakerState.CLOSED] += 1
        self.logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
    
    def _record_success(self, execution_time: float) -> None:
        """Record successful execution."""
        with self._lock:
            self.total_successes += 1
            self.failure_window.append(0)  # 0 = success
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._close_circuit()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self, exception: Exception) -> None:
        """Record failed execution."""
        with self._lock:
            self.total_failures += 1
            self.last_failure_time = time.time()
            self.failure_window.append(1)  # 1 = failure
            
            if self.state == CircuitBreakerState.CLOSED:
                self.failure_count += 1
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Failed in half-open, go back to open
                self._open_circuit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            failure_rate = self.total_failures / max(self.total_requests, 1)
            
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.total_requests,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "failure_rate": failure_rate,
                "current_failure_count": self.failure_count,
                "time_until_half_open": self._time_until_half_open() if self.state == CircuitBreakerState.OPEN else 0,
                "state_changes": dict(self.state_changes)
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0.0
            self.failure_window.clear()
            self.logger.info(f"Circuit breaker '{self.name}' reset")


class CircuitBreakerOpenError(RAGException):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, retry_delay: float = 60.0):
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=["Wait for service recovery", "Try alternative approach"],
            user_message="Service temporarily unavailable. Please try again later."
        )


class CircuitBreakerManager:
    """Manage multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
    
    def get_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker."""
        with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name, config)
                self.logger.info(f"Created circuit breaker: {name}")
            
            return self.breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self.breakers.values():
                breaker.reset()
            self.logger.info("All circuit breakers reset")


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorator to apply circuit breaker to a function.
    
    Args:
        name: Circuit breaker name
        config: Optional circuit breaker configuration
    """
    def decorator(func):
        breaker = circuit_breaker_manager.get_breaker(name, config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator