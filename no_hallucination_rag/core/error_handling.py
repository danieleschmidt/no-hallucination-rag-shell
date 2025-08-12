"""
Error handling for RAG system.
Generation 1: Basic error handling with logging.
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import traceback


class ErrorCategory(Enum):
    """Categories of errors."""
    VALIDATION = "validation"
    RETRIEVAL = "retrieval" 
    PROCESSING = "processing"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for errors."""
    user_query: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorDetails:
    """Details of an error."""
    category: ErrorCategory
    message: str
    user_message: str
    context: Optional[ErrorContext] = None
    traceback: Optional[str] = None


class ValidationError(Exception):
    """Input validation error."""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context


class RetrievalError(Exception):
    """Source retrieval error."""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context


class ProcessingError(Exception):
    """Processing error."""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        import time
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
    def retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        import time
        import random
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    raise e
                    
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                jitter = random.uniform(0, 0.1) * delay
                time.sleep(delay + jitter)
                
        raise Exception(f"Failed after {self.max_retries} retries")


class ErrorHandler:
    """Enhanced error handler with circuit breaker and retry logic."""
    
    def __init__(self, enable_circuit_breaker: bool = True, enable_retry: bool = True):
        self.logger = logging.getLogger(__name__)
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self.retry_manager = RetryManager() if enable_retry else None
        self.error_counts = {}
        self.alert_thresholds = {
            ErrorCategory.VALIDATION: 10,
            ErrorCategory.RETRIEVAL: 5, 
            ErrorCategory.PROCESSING: 8,
            ErrorCategory.SYSTEM: 3
        }
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> ErrorDetails:
        """Handle and categorize an error."""
        
        # Determine error category
        if isinstance(error, ValidationError):
            category = ErrorCategory.VALIDATION
        elif isinstance(error, RetrievalError):
            category = ErrorCategory.RETRIEVAL
        elif isinstance(error, ProcessingError):
            category = ErrorCategory.PROCESSING
        else:
            category = ErrorCategory.SYSTEM
        
        # Create user-friendly message
        user_message = self._get_user_message(error, category)
        
        # Track error counts
        self.error_counts[category] = self.error_counts.get(category, 0) + 1
        
        # Log the error
        self.logger.error(f"Error in {context.component if context else 'unknown'}: {str(error)}")
        
        return ErrorDetails(
            category=category,
            message=str(error),
            user_message=user_message,
            context=context,
            traceback=traceback.format_exc()
        )
    
    def _get_user_message(self, error: Exception, category: ErrorCategory) -> str:
        """Get user-friendly error message."""
        
        if category == ErrorCategory.VALIDATION:
            return "Please check your input and try again."
        elif category == ErrorCategory.RETRIEVAL:
            return "Unable to retrieve information at this time. Please try again later."
        elif category == ErrorCategory.PROCESSING:
            return "Unable to process your request. Please try rephrasing your question."
        else:
            return "A system error occurred. Please try again later."
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": dict(self.error_counts),
            "total_errors": sum(self.error_counts.values())
        }


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    logging.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
            return None
        return wrapper
    return decorator


def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}")
        return None