"""
Input validation and sanitization for RAG system.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import html
import urllib.parse


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: str
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class InputValidator:
    """Validates and sanitizes user inputs for security and safety."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Malicious patterns to detect
        self.malicious_patterns = [
            # Injection attacks
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
            
            # SQL injection patterns
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            
            # Command injection
            r';\s*rm\s+-rf',
            r'&&\s*rm',
            r'\|\s*rm',
            r'`.*`',
            r'\$\(.*\)',
            
            # Path traversal
            r'\.\./.*\.\.',
            r'\.\.[\\/]',
            
            # Other suspicious patterns
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.malicious_patterns]
        
        # Maximum input lengths
        self.max_lengths = {
            'query': 2000,
            'document_content': 100000,
            'metadata_value': 1000,
            'filename': 255
        }
    
    def validate_query(self, query: str) -> ValidationResult:
        """
        Validate user query for safety and format.
        
        Args:
            query: User input query
            
        Returns:
            ValidationResult with validation status and sanitized input
        """
        if not isinstance(query, str):
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                errors=["Query must be a string"]
            )
        
        warnings = []
        errors = []
        
        # Check length
        if len(query) > self.max_lengths['query']:
            errors.append(f"Query too long (max {self.max_lengths['query']} characters)")
        
        if len(query.strip()) == 0:
            errors.append("Query cannot be empty")
        
        # Check for malicious patterns
        malicious_found = []
        for pattern in self.compiled_patterns:
            if pattern.search(query):
                malicious_found.append(pattern.pattern)
        
        if malicious_found:
            errors.append(f"Potentially malicious content detected: {malicious_found}")
        
        # Sanitize input
        sanitized = self._sanitize_text(query)
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in sanitized if not c.isalnum() and not c.isspace()) / max(len(sanitized), 1)
        if special_char_ratio > 0.3:
            warnings.append("High ratio of special characters detected")
        
        # Check for repeated characters (potential DoS)
        if re.search(r'(.)\1{20,}', sanitized):
            warnings.append("Excessive character repetition detected")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized,
            warnings=warnings,
            errors=errors
        )
    
    def validate_document_content(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate document content for knowledge base insertion.
        
        Args:
            content: Document content
            metadata: Optional document metadata
            
        Returns:
            ValidationResult with validation status
        """
        warnings = []
        errors = []
        
        if not isinstance(content, str):
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                errors=["Content must be a string"]
            )
        
        # Check length
        if len(content) > self.max_lengths['document_content']:
            errors.append(f"Document too long (max {self.max_lengths['document_content']} characters)")
        
        if len(content.strip()) == 0:
            errors.append("Document content cannot be empty")
        
        # Validate metadata if provided
        if metadata:
            metadata_validation = self._validate_metadata(metadata)
            warnings.extend(metadata_validation.warnings)
            errors.extend(metadata_validation.errors)
        
        # Sanitize content
        sanitized = self._sanitize_text(content)
        
        # Check for potential privacy issues
        privacy_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b',  # Phone
        ]
        
        privacy_found = []
        for pattern in privacy_patterns:
            if re.search(pattern, content):
                privacy_found.append("PII detected")
        
        if privacy_found:
            warnings.extend(privacy_found)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized,
            warnings=warnings,
            errors=errors
        )
    
    def validate_url(self, url: str) -> ValidationResult:
        """
        Validate URL for safety.
        
        Args:
            url: URL to validate
            
        Returns:
            ValidationResult with validation status
        """
        warnings = []  
        errors = []
        
        if not isinstance(url, str):
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                errors=["URL must be a string"]
            )
        
        # Basic URL format validation
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                errors.append("Invalid URL format")
        except Exception:
            errors.append("Failed to parse URL")
        
        # Check for dangerous schemes
        dangerous_schemes = ['javascript', 'vbscript', 'data', 'file']
        if parsed.scheme.lower() in dangerous_schemes:
            errors.append(f"Dangerous URL scheme: {parsed.scheme}")
        
        # Check for suspicious domains
        suspicious_domains = ['localhost', '127.0.0.1', '0.0.0.0', '10.', '192.168.', '172.16.']
        if any(suspicious in parsed.netloc for suspicious in suspicious_domains):
            warnings.append("Internal/private network URL detected")
        
        # Sanitize URL
        sanitized = url.strip()
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized,
            warnings=warnings,
            errors=errors
        )
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate document metadata."""
        warnings = []
        errors = []
        
        for key, value in metadata.items():
            if not isinstance(key, str):
                errors.append(f"Metadata key must be string: {type(key)}")
                continue
            
            if len(key) > 100:
                errors.append(f"Metadata key too long: {key}")
                continue
            
            # Validate value
            if isinstance(value, str):
                if len(value) > self.max_lengths['metadata_value']:
                    errors.append(f"Metadata value too long for key '{key}'")
                
                # Check for malicious content in metadata
                for pattern in self.compiled_patterns:
                    if pattern.search(value):
                        errors.append(f"Malicious content in metadata '{key}': {pattern.pattern}")
            
            elif isinstance(value, (int, float, bool)):
                pass  # These types are safe
            
            elif isinstance(value, list):
                if len(value) > 100:
                    warnings.append(f"Large list in metadata '{key}' ({len(value)} items)")
                
                for item in value:
                    if isinstance(item, str) and len(item) > self.max_lengths['metadata_value']:
                        errors.append(f"List item too long in metadata '{key}'")
            
            else:
                warnings.append(f"Unusual metadata type for '{key}': {type(value)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input="",  # Metadata sanitization handled separately
            warnings=warnings,
            errors=errors
        )
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text input.
        
        Args:
            text: Raw text input
            
        Returns:
            Sanitized text
        """
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove control characters except common ones
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        # Limit consecutive special characters
        sanitized = re.sub(r'([^\w\s])\1{5,}', r'\1\1\1\1\1', sanitized)
        
        return sanitized.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Raw filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Limit length
        if len(filename) > self.max_lengths['filename']:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_len = self.max_lengths['filename'] - len(ext) - 1
            filename = name[:max_name_len] + ('.' + ext if ext else '')
        
        # Ensure not empty
        if not filename or filename == '.':
            filename = 'document.txt'
        
        return filename