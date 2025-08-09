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


class ErrorHandler:
    """Handles and categorizes errors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
    
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