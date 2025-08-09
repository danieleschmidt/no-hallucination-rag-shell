"""
Input validation for RAG system.
Generation 1: Basic validation rules.
"""

import logging
import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class InputValidator:
    """Validates user inputs for safety and quality."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_query_length = 1000
        self.min_query_length = 3
        
        self.logger.info("InputValidator initialized (Generation 1)")
    
    def validate_query(self, query: str) -> ValidationResult:
        """Validate a user query."""
        errors = []
        warnings = []
        
        # Basic checks
        if not query or not query.strip():
            errors.append("Query cannot be empty")
            return ValidationResult(False, errors, warnings)
        
        query = query.strip()
        
        # Length checks
        if len(query) < self.min_query_length:
            errors.append(f"Query too short (minimum {self.min_query_length} characters)")
        
        if len(query) > self.max_query_length:
            errors.append(f"Query too long (maximum {self.max_query_length} characters)")
        
        # Content checks
        if query.count('?') > 5:
            warnings.append("Query contains many question marks")
        
        # Potentially harmful patterns
        harmful_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        for pattern in harmful_patterns:
            if pattern.lower() in query.lower():
                errors.append(f"Query contains potentially harmful content: {pattern}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings)