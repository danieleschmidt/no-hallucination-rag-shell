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