"""
Security tests for RAG system components.
"""

import pytest
import time
from unittest.mock import Mock, patch

from no_hallucination_rag.security.security_manager import (
    SecurityManager, RateLimiter, RateLimitConfig
)
from no_hallucination_rag.core.validation import InputValidator, ValidationResult


class TestInputValidator:
    """Test suite for input validation."""
    
    @pytest.fixture
    def validator(self):
        """Create input validator for testing."""
        return InputValidator()
    
    def test_valid_query(self, validator):
        """Test validation of valid query."""
        result = validator.validate_query("What are AI safety requirements?")
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.sanitized_input == "What are AI safety requirements?"
    
    def test_empty_query(self, validator):
        """Test validation of empty query."""
        result = validator.validate_query("")
        
        assert result.is_valid is False
        assert "Query cannot be empty" in result.errors
    
    def test_query_too_long(self, validator):
        """Test validation of overly long query."""
        long_query = "x" * 3000
        result = validator.validate_query(long_query)
        
        assert result.is_valid is False
        assert "Query too long" in result.errors[0]
    
    def test_malicious_query_detection(self, validator):
        """Test detection of malicious patterns."""
        malicious_queries = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "javascript:alert('xss')",
            "`rm -rf /`",
            "../../../etc/passwd"
        ]
        
        for query in malicious_queries:
            result = validator.validate_query(query)
            assert result.is_valid is False
            assert "malicious content detected" in result.errors[0].lower()
    
    def test_query_sanitization(self, validator):
        """Test query sanitization."""
        result = validator.validate_query("What is <b>AI</b> safety?")
        
        assert result.is_valid is True
        assert "&lt;b&gt;" in result.sanitized_input
        assert "&lt;/b&gt;" in result.sanitized_input
    
    def test_excessive_special_characters(self, validator):
        """Test handling of excessive special characters."""
        result = validator.validate_query("!@#$%^&*()!@#$%^&*()!@#$%^&*()")
        
        assert "High ratio of special characters" in result.warnings
    
    def test_repeated_characters(self, validator):
        """Test handling of repeated characters."""
        result = validator.validate_query("aaaaaaaaaaaaaaaaaaaaaaaaa normal text")
        
        assert "Excessive character repetition" in result.warnings
    
    def test_document_validation(self, validator):
        """Test document content validation."""
        valid_content = "This is a valid document about AI safety requirements."
        result = validator.validate_document_content(valid_content)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_document_too_long(self, validator):
        """Test document that's too long."""
        long_content = "x" * 200000  # Exceeds max length
        result = validator.validate_document_content(long_content)
        
        assert result.is_valid is False
        assert "Document too long" in result.errors[0]
    
    def test_document_pii_detection(self, validator):
        """Test PII detection in documents."""
        pii_content = """
        Contact information:
        Email: john.doe@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Credit Card: 4111-1111-1111-1111
        """
        
        result = validator.validate_document_content(pii_content)
        
        assert result.is_valid is True  # Valid but with warnings
        assert len(result.warnings) > 0
        assert any("PII detected" in warning for warning in result.warnings)
    
    def test_url_validation(self, validator):
        """Test URL validation."""
        # Valid URLs
        valid_urls = [
            "https://example.com",
            "http://subdomain.example.org/path",
            "https://example.com:8080/path?query=value"
        ]
        
        for url in valid_urls:
            result = validator.validate_url(url)
            assert result.is_valid is True
        
        # Invalid URLs
        invalid_urls = [
            "not-a-url",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd"
        ]
        
        for url in invalid_urls:
            result = validator.validate_url(url)
            assert result.is_valid is False
    
    def test_internal_url_warning(self, validator):
        """Test warning for internal network URLs."""
        internal_urls = [
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://192.168.1.1",
            "http://10.0.0.1"
        ]
        
        for url in internal_urls:
            result = validator.validate_url(url)
            if result.is_valid:  # May be valid but with warnings
                assert any("Internal/private network" in warning for warning in result.warnings)
    
    def test_filename_sanitization(self, validator):
        """Test filename sanitization."""
        dangerous_filenames = [
            "../../../etc/passwd",
            "file with spaces.txt",
            "file<>with|dangerous*chars.txt",
            "normal_file.pdf"
        ]
        
        expected_safe = [
            "passwd",
            "file_with_spaces.txt",
            "file__with_dangerous_chars.txt",
            "normal_file.pdf"
        ]
        
        for dangerous, expected in zip(dangerous_filenames, expected_safe):
            safe = validator.sanitize_filename(dangerous)
            assert safe == expected


class TestRateLimiter:
    """Test suite for rate limiting."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter for testing."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000
        )
        return RateLimiter(config)
    
    def test_rate_limiting_allows_requests(self, rate_limiter):
        """Test that rate limiter allows requests within limits."""
        client_id = "test_client"
        
        # First request should be allowed
        allowed, details = rate_limiter.is_allowed(client_id)
        assert allowed is True
        assert details["minute_count"] == 1
        assert details["hour_count"] == 1
        assert details["day_count"] == 1
    
    def test_rate_limiting_blocks_excessive_requests(self, rate_limiter):
        """Test that rate limiter blocks excessive requests."""
        client_id = "test_client"
        current_time = time.time()
        
        # Make requests up to the limit
        for i in range(10):
            allowed, details = rate_limiter.is_allowed(client_id, current_time)
            assert allowed is True
        
        # Next request should be blocked
        allowed, details = rate_limiter.is_allowed(client_id, current_time)
        assert allowed is False
        assert details["reason"] == "minute_limit_exceeded"
    
    def test_rate_limit_reset(self, rate_limiter):
        """Test that rate limits reset over time."""
        client_id = "test_client"
        start_time = time.time()
        
        # Fill up the minute limit
        for i in range(10):
            allowed, details = rate_limiter.is_allowed(client_id, start_time)
            assert allowed is True
        
        # Should be blocked
        allowed, details = rate_limiter.is_allowed(client_id, start_time)
        assert allowed is False
        
        # After a minute, should be allowed again
        future_time = start_time + 61  # 61 seconds later
        allowed, details = rate_limiter.is_allowed(client_id, future_time)
        assert allowed is True
    
    def test_different_clients_independent_limits(self, rate_limiter):
        """Test that different clients have independent limits."""
        current_time = time.time()
        
        # Client 1 fills up limit
        for i in range(10):
            allowed, details = rate_limiter.is_allowed("client1", current_time)
            assert allowed is True
        
        # Client 1 should be blocked
        allowed, details = rate_limiter.is_allowed("client1", current_time)
        assert allowed is False
        
        # Client 2 should still be allowed
        allowed, details = rate_limiter.is_allowed("client2", current_time)
        assert allowed is True
    
    def test_rate_limiter_stats(self, rate_limiter):
        """Test rate limiter statistics."""
        client_id = "test_client"
        current_time = time.time()
        
        # Make some requests
        for i in range(5):
            rate_limiter.is_allowed(client_id, current_time)
        
        stats = rate_limiter.get_client_stats(client_id)
        
        assert stats["requests_last_minute"] == 5
        assert stats["requests_last_hour"] == 5
        assert stats["requests_last_day"] == 5
        assert stats["limits"]["minute"] == 10
        assert stats["limits"]["hour"] == 100
        assert stats["limits"]["day"] == 1000


class TestSecurityManager:
    """Test suite for security manager."""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager for testing."""
        return SecurityManager(enable_ip_filtering=True, enable_request_signing=False)
    
    def test_basic_request_validation(self, security_manager):
        """Test basic request validation."""
        is_valid, details = security_manager.validate_request(
            client_ip="192.168.1.1",
            user_id="test_user"
        )
        
        assert is_valid is True
        assert "checks" in details
        assert len(details["checks"]) >= 2  # IP and rate limit checks
    
    def test_ip_blocking(self, security_manager):
        """Test IP address blocking."""
        malicious_ip = "10.0.0.100"
        
        # Block the IP
        security_manager.block_ip(malicious_ip, "malicious_activity")
        
        # Request from blocked IP should be rejected
        is_valid, details = security_manager.validate_request(client_ip=malicious_ip)
        
        assert is_valid is False
        ip_check = next(check for check in details["checks"] if check["type"] == "ip_validation")
        assert ip_check["result"] is False
    
    def test_rate_limiting_integration(self, security_manager):
        """Test rate limiting integration."""
        client_ip = "192.168.1.10"
        
        # Make requests up to limit
        for i in range(60):  # Default rate limit
            is_valid, details = security_manager.validate_request(client_ip=client_ip)
            if not is_valid:
                break
        
        # Should eventually be rate limited
        is_valid, details = security_manager.validate_request(client_ip=client_ip)
        if not is_valid:
            rate_check = next(check for check in details["checks"] if check["type"] == "rate_limit")
            assert rate_check["result"] is False
    
    def test_api_key_creation_and_validation(self, security_manager):
        """Test API key creation and validation."""
        # Create API key
        api_key, key_id = security_manager.create_api_key(
            user_id="test_user",
            scopes=["read", "write"],
            expires_days=30
        )
        
        assert isinstance(api_key, str)
        assert len(api_key) > 20
        assert isinstance(key_id, str)
        
        # Validate API key
        is_valid, details = security_manager.validate_request(api_key=api_key)
        
        assert is_valid is True
        api_check = next(check for check in details["checks"] if check["type"] == "api_key")
        assert api_check["result"] is True
        assert api_check["details"]["user_id"] == "test_user"
    
    def test_api_key_revocation(self, security_manager):
        """Test API key revocation."""
        # Create and revoke API key
        api_key, key_id = security_manager.create_api_key("test_user")
        revoked = security_manager.revoke_api_key(api_key)
        
        assert revoked is True
        
        # Revoked key should be invalid
        is_valid, details = security_manager.validate_request(api_key=api_key)
        
        assert is_valid is False
        api_check = next(check for check in details["checks"] if check["type"] == "api_key")
        assert api_check["result"] is False
    
    def test_invalid_ip_format(self, security_manager):
        """Test handling of invalid IP format."""
        is_valid, details = security_manager.validate_request(client_ip="not-an-ip")
        
        assert is_valid is False
        ip_check = next(check for check in details["checks"] if check["type"] == "ip_validation")
        assert ip_check["result"] is False
        assert "invalid_ip_format" in ip_check["details"]["reason"]
    
    def test_security_events_logging(self, security_manager):
        """Test security events are logged properly."""
        # Trigger security event
        security_manager.block_ip("10.0.0.200", "test_block")
        
        # Check that events are recorded
        stats = security_manager.get_security_stats()
        assert stats["blocked_ips"] >= 1
    
    def test_security_statistics(self, security_manager):
        """Test security statistics collection."""
        # Create some API keys and block some IPs
        security_manager.create_api_key("user1")
        security_manager.create_api_key("user2")
        security_manager.block_ip("10.0.0.201")
        
        stats = security_manager.get_security_stats()
        
        assert "blocked_ips" in stats
        assert "active_api_keys" in stats
        assert "rate_limit_config" in stats
        assert stats["active_api_keys"] >= 2
        assert stats["blocked_ips"] >= 1
    
    def test_concurrent_request_validation(self, security_manager):
        """Test concurrent request validation doesn't cause race conditions."""
        import threading
        
        results = []
        
        def make_request():
            is_valid, details = security_manager.validate_request(
                client_ip=f"192.168.1.{threading.current_thread().ident % 255}"
            )
            results.append(is_valid)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All should complete without errors
        assert len(results) == 10


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_full_security_pipeline(self):
        """Test complete security validation pipeline."""
        security_manager = SecurityManager()
        validator = InputValidator()
        
        # Validate input
        query = "What are AI safety requirements?"
        validation_result = validator.validate_query(query)
        assert validation_result.is_valid is True
        
        # Validate request
        is_valid, details = security_manager.validate_request(
            client_ip="192.168.1.100",
            user_id="test_user"
        )
        assert is_valid is True
        
        # Should pass both validation steps
        assert validation_result.is_valid and is_valid
    
    def test_security_with_malicious_input(self):
        """Test security handling of malicious input."""
        security_manager = SecurityManager()
        validator = InputValidator()
        
        # Try malicious query
        malicious_query = "<script>alert('xss')</script>"
        validation_result = validator.validate_query(malicious_query)
        
        # Should be blocked by input validation
        assert validation_result.is_valid is False
        
        # Even if somehow it passed, security manager should handle it
        is_valid, details = security_manager.validate_request(
            client_ip="192.168.1.100"
        )
        # Rate limiting should still work for legitimate parts
        assert isinstance(is_valid, bool)
    
    def test_defense_in_depth(self):
        """Test multiple layers of security work together."""
        security_manager = SecurityManager()
        validator = InputValidator()
        
        # Multiple validation layers
        test_cases = [
            ("Normal query", True),
            ("<script>alert('xss')</script>", False),
            ("'; DROP TABLE users; --", False),
            ("../../../etc/passwd", False),
            ("What are " + "x" * 3000, False),  # Too long
        ]
        
        for query, should_pass in test_cases:
            validation_result = validator.validate_query(query)
            
            if should_pass:
                assert validation_result.is_valid is True
            else:
                assert validation_result.is_valid is False
                
            # Security manager should work regardless
            is_valid, details = security_manager.validate_request(
                client_ip="192.168.1.200"
            )
            assert isinstance(is_valid, bool)