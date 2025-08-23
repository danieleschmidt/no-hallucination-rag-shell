
"""
Comprehensive input validation and sanitization system.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import html
import urllib.parse
from datetime import datetime


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Any = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class InputSanitizer:
    """Advanced input sanitization system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Dangerous patterns to detect/remove  
        self.sql_injection_patterns = [
            r"\'\s*;|;\s*\'",
            r"(select|insert|update|delete|drop|create|alter)\s+",
            r"union\s+select",
            r"or\s+1=1",
            r"--\s*$",
            r"/\*.*\*/"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>"
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$]",
            r"\.\.(/|\\)",
            r"(rm|del|format|shutdown)\s+",
            r">(\s*[/\\]|\w)",
        ]
    
    def sanitize_string(self, text: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # HTML escape
        text = html.escape(text, quote=True)
        
        # URL decode if needed
        if '%' in text:
            try:
                text = urllib.parse.unquote(text)
            except Exception:
                pass  # Keep original if URL decode fails
        
        # Trim whitespace
        text = text.strip()
        
        # Length limit
        if max_length and len(text) > max_length:
            text = text[:max_length]
            self.logger.warning(f"Input truncated to {max_length} characters")
        
        return text
    
    def detect_sql_injection(self, text: str) -> bool:
        """Detect potential SQL injection attempts."""
        text_lower = text.lower()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                self.logger.warning(f"Potential SQL injection detected: pattern {pattern}")
                return True
        
        return False
    
    def detect_xss(self, text: str) -> bool:
        """Detect potential XSS attempts."""
        text_lower = text.lower()
        
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                self.logger.warning(f"Potential XSS detected: pattern {pattern}")
                return True
        
        return False
    
    def detect_command_injection(self, text: str) -> bool:
        """Detect potential command injection attempts."""
        for pattern in self.command_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self.logger.warning(f"Potential command injection detected: pattern {pattern}")
                return True
        
        return False
    
    def is_suspicious(self, text: str) -> bool:
        """Check if input contains suspicious patterns."""
        return (self.detect_sql_injection(text) or 
                self.detect_xss(text) or 
                self.detect_command_injection(text))


class ValidationRules:
    """Collection of validation rules."""
    
    @staticmethod
    def required(value: Any) -> ValidationResult:
        """Validate required field."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return ValidationResult(False, ["Field is required"])
        return ValidationResult(True, [])
    
    @staticmethod
    def string_length(value: str, min_length: int = 0, max_length: int = None) -> ValidationResult:
        """Validate string length."""
        if not isinstance(value, str):
            return ValidationResult(False, ["Value must be a string"])
        
        errors = []
        if len(value) < min_length:
            errors.append(f"String too short (minimum {min_length} characters)")
        
        if max_length and len(value) > max_length:
            errors.append(f"String too long (maximum {max_length} characters)")
        
        return ValidationResult(len(errors) == 0, errors)
    
    @staticmethod
    def email_format(email: str) -> ValidationResult:
        """Validate email format."""
        if not isinstance(email, str):
            return ValidationResult(False, ["Email must be a string"])
        
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email):
            return ValidationResult(False, ["Invalid email format"])
        
        return ValidationResult(True, [])
    
    @staticmethod
    def numeric_range(value: Union[int, float], min_value: Union[int, float] = None, 
                     max_value: Union[int, float] = None) -> ValidationResult:
        """Validate numeric range."""
        if not isinstance(value, (int, float)):
            return ValidationResult(False, ["Value must be numeric"])
        
        errors = []
        if min_value is not None and value < min_value:
            errors.append(f"Value too small (minimum {min_value})")
        
        if max_value is not None and value > max_value:
            errors.append(f"Value too large (maximum {max_value})")
        
        return ValidationResult(len(errors) == 0, errors)
    
    @staticmethod
    def regex_pattern(value: str, pattern: str, pattern_name: str = "pattern") -> ValidationResult:
        """Validate against regex pattern."""
        if not isinstance(value, str):
            return ValidationResult(False, ["Value must be a string"])
        
        if not re.match(pattern, value):
            return ValidationResult(False, [f"Value does not match {pattern_name} pattern"])
        
        return ValidationResult(True, [])
    
    @staticmethod
    def allowed_values(value: Any, allowed: List[Any]) -> ValidationResult:
        """Validate value is in allowed list."""
        if value not in allowed:
            return ValidationResult(False, [f"Value must be one of: {allowed}"])
        
        return ValidationResult(True, [])


class AdvancedValidator:
    """Advanced input validation system."""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.logger = logging.getLogger(__name__)
        self.validation_cache = {}
    
    def validate_query(self, query: str) -> ValidationResult:
        """Validate user query input."""
        errors = []
        warnings = []
        
        # Basic checks
        if not query or not query.strip():
            return ValidationResult(False, ["Query cannot be empty"], [])
        
        # Length validation
        if len(query) > 10000:  # 10K character limit
            errors.append("Query too long (maximum 10,000 characters)")
        
        if len(query.split()) > 1000:  # 1K word limit
            errors.append("Query has too many words (maximum 1,000)")
        
        # Security validation
        if self.sanitizer.is_suspicious(query):
            errors.append("Query contains potentially malicious content")
        
        # Content quality checks
        if len(query.strip()) < 3:
            warnings.append("Very short query may not produce good results")
        
        # Check for excessive repetition
        words = query.lower().split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            warnings.append("Query contains excessive word repetition")
        
        # Sanitize query
        sanitized = self.sanitizer.sanitize_string(query, max_length=10000)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized,
            metadata={
                "original_length": len(query),
                "word_count": len(words),
                "unique_words": len(set(words))
            }
        )
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration parameters."""
        errors = []
        warnings = []
        
        required_fields = ["factuality_threshold"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Required configuration field missing: {field}")
        
        # Validate specific config values
        if "factuality_threshold" in config:
            threshold = config["factuality_threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                errors.append("factuality_threshold must be a number between 0 and 1")
        
        if "max_sources" in config:
            max_sources = config["max_sources"]
            if not isinstance(max_sources, int) or max_sources < 1:
                errors.append("max_sources must be a positive integer")
            elif max_sources > 100:
                warnings.append("max_sources > 100 may impact performance")
        
        if "min_sources" in config:
            min_sources = config["min_sources"]
            if not isinstance(min_sources, int) or min_sources < 1:
                errors.append("min_sources must be a positive integer")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"validated_fields": list(config.keys())}
        )
    
    def validate_source_data(self, source: Dict[str, Any]) -> ValidationResult:
        """Validate source document data."""
        errors = []
        warnings = []
        
        required_fields = ["content", "title"]
        for field in required_fields:
            if field not in source or not source[field]:
                errors.append(f"Source missing required field: {field}")
        
        # Validate content
        if "content" in source:
            content = source["content"]
            if len(content) < 10:
                warnings.append("Source content very short (< 10 characters)")
            elif len(content) > 1000000:  # 1MB limit
                errors.append("Source content too large (> 1MB)")
            
            if self.sanitizer.is_suspicious(content):
                errors.append("Source content contains suspicious patterns")
        
        # Validate URL if present
        if "url" in source and source["url"]:
            url = source["url"]
            url_pattern = r"^https?://[\w\.-]+\.[a-zA-Z]{2,}(/.*)?$"
            if not re.match(url_pattern, url):
                warnings.append("URL format appears invalid")
        
        # Validate authority score
        if "authority_score" in source:
            score = source["authority_score"]
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                errors.append("authority_score must be between 0 and 1")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "content_length": len(source.get("content", "")),
                "has_url": "url" in source and source["url"]
            }
        )
    
    def batch_validate(self, items: List[Any], validator_func: Callable) -> Dict[str, Any]:
        """Validate multiple items and return summary."""
        results = []
        total_errors = 0
        total_warnings = 0
        
        for i, item in enumerate(items):
            try:
                result = validator_func(item)
                results.append({"index": i, "result": result})
                total_errors += len(result.errors)
                total_warnings += len(result.warnings)
            except Exception as e:
                self.logger.error(f"Validation failed for item {i}: {e}")
                results.append({
                    "index": i, 
                    "result": ValidationResult(False, [f"Validation error: {e}"])
                })
                total_errors += 1
        
        valid_count = sum(1 for r in results if r["result"].is_valid)
        
        return {
            "total_items": len(items),
            "valid_items": valid_count,
            "invalid_items": len(items) - valid_count,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "results": results,
            "success_rate": valid_count / len(items) if items else 1.0
        }
