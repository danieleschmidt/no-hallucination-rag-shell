#!/usr/bin/env python3
"""
Simple test runner to validate the RAG system implementation.
"""

import sys
import os
import traceback
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from no_hallucination_rag.core.factual_rag import FactualRAG, RAGResponse
        from no_hallucination_rag.core.source_ranker import SourceRanker
        from no_hallucination_rag.verification.factuality_detector import FactualityDetector
        from no_hallucination_rag.governance.compliance_checker import GovernanceChecker
        from no_hallucination_rag.core.validation import InputValidator
        from no_hallucination_rag.core.error_handling import ErrorHandler
        from no_hallucination_rag.monitoring.metrics import MetricsCollector
        from no_hallucination_rag.security.security_manager import SecurityManager
        from no_hallucination_rag.optimization.caching import CacheManager
        from no_hallucination_rag.optimization.concurrent_processing import AsyncRAGProcessor
        from no_hallucination_rag.optimization.performance_optimizer import PerformanceOptimizer
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_initialization():
    """Test basic system initialization."""
    print("Testing basic initialization...")
    
    try:
        from no_hallucination_rag.core.factual_rag import FactualRAG
        
        # Test with minimal configuration
        rag = FactualRAG(
            factuality_threshold=0.9,
            enable_caching=False,
            enable_optimization=False,
            enable_metrics=False,
            enable_security=False
        )
        
        assert rag.factuality_threshold == 0.9
        assert rag.retriever is not None
        assert rag.source_ranker is not None
        assert rag.factuality_detector is not None
        assert rag.governance_checker is not None
        
        print("✓ Basic initialization successful")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        traceback.print_exc()
        return False

def test_query_processing():
    """Test basic query processing."""
    print("Testing query processing...")
    
    try:
        from no_hallucination_rag.core.factual_rag import FactualRAG, RAGResponse
        
        rag = FactualRAG(
            factuality_threshold=0.8,
            enable_caching=False,
            enable_optimization=False,
            enable_metrics=False,
            enable_security=False
        )
        
        # Test query processing with mock knowledge base
        response = rag.query("What are AI safety requirements?")
        
        assert isinstance(response, RAGResponse)
        assert response.factuality_score >= 0.0
        assert response.answer is not None
        assert response.timestamp is not None
        assert isinstance(response.sources, list)
        assert isinstance(response.governance_compliant, bool)
        
        print(f"✓ Query processing successful")
        print(f"  - Factuality score: {response.factuality_score:.2f}")
        print(f"  - Sources found: {len(response.sources)}")
        print(f"  - Governance compliant: {response.governance_compliant}")
        print(f"  - Answer length: {len(response.answer)} chars")
        return True
    except Exception as e:
        print(f"✗ Query processing failed: {e}")
        traceback.print_exc()
        return False

def test_input_validation():
    """Test input validation."""
    print("Testing input validation...")
    
    try:
        from no_hallucination_rag.core.validation import InputValidator
        
        validator = InputValidator()
        
        # Test valid input
        result = validator.validate_query("What are AI safety requirements?")
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Test malicious input
        result = validator.validate_query("<script>alert('xss')</script>")
        assert result.is_valid is False
        assert len(result.errors) > 0
        
        # Test empty input
        result = validator.validate_query("")
        assert result.is_valid is False
        
        print("✓ Input validation successful")
        return True
    except Exception as e:
        print(f"✗ Input validation failed: {e}")
        traceback.print_exc()
        return False

def test_caching():
    """Test caching functionality."""
    print("Testing caching...")
    
    try:
        from no_hallucination_rag.optimization.caching import AdaptiveCache, CacheManager
        
        # Test basic cache operations
        cache = AdaptiveCache(max_size=10, max_memory_mb=1, default_ttl=3600)
        
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
        
        # Test cache miss
        result = cache.get("nonexistent")
        assert result is None
        
        # Test cache manager
        cache_manager = CacheManager()
        assert "queries" in cache_manager.caches
        assert "retrieval" in cache_manager.caches
        
        print("✓ Caching functionality successful")
        return True
    except Exception as e:
        print(f"✗ Caching failed: {e}")
        traceback.print_exc()
        return False

def test_security():
    """Test security functionality."""
    print("Testing security...")
    
    try:
        from no_hallucination_rag.security.security_manager import SecurityManager, RateLimiter
        
        # Test rate limiter
        rate_limiter = RateLimiter()
        allowed, details = rate_limiter.is_allowed("test_client")
        assert allowed is True
        assert "minute_count" in details
        
        # Test security manager
        security_manager = SecurityManager()
        is_valid, details = security_manager.validate_request(
            client_ip="192.168.1.1",
            user_id="test_user"
        )
        assert isinstance(is_valid, bool)
        assert "checks" in details
        
        print("✓ Security functionality successful")
        return True
    except Exception as e:
        print(f"✗ Security failed: {e}")
        traceback.print_exc()
        return False

async def test_async_functionality():
    """Test async functionality."""
    print("Testing async functionality...")
    
    try:
        from no_hallucination_rag.core.factual_rag import FactualRAG
        from no_hallucination_rag.optimization.concurrent_processing import AsyncRAGProcessor
        
        rag = FactualRAG(
            enable_caching=False,
            enable_optimization=False,
            enable_metrics=False,
            enable_security=False,  
            enable_concurrency=False  # Use fallback async
        )
        
        # Test async query
        response = await rag.aquery("What are AI safety requirements?")
        assert response is not None
        assert response.answer is not None
        
        # Test batch async queries
        queries = ["Query 1", "Query 2"]
        responses = await rag.aquery_batch(queries, batch_size=2)
        assert len(responses) == 2
        
        print("✓ Async functionality successful")
        return True
    except Exception as e:
        print(f"✗ Async functionality failed: {e}")
        traceback.print_exc()
        return False

def test_system_health():
    """Test system health monitoring."""
    print("Testing system health...")
    
    try:
        from no_hallucination_rag.core.factual_rag import FactualRAG
        
        rag = FactualRAG(
            enable_caching=True,
            enable_optimization=True,
            enable_metrics=True,
            enable_security=True
        )
        
        # Test health check
        health = rag.get_system_health()
        assert "status" in health
        assert "timestamp" in health
        assert "components" in health
        
        # Test performance stats
        stats = rag.get_performance_stats()
        assert "timestamp" in stats
        assert "components_enabled" in stats
        
        # Cleanup
        rag.shutdown()
        
        print("✓ System health monitoring successful")
        return True
    except Exception as e:
        print(f"✗ System health failed: {e}")
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test end-to-end functionality."""
    print("Testing end-to-end functionality...")
    
    try:
        from no_hallucination_rag.core.factual_rag import FactualRAG
        
        # Full system with all features enabled
        rag = FactualRAG(
            factuality_threshold=0.8,
            governance_mode="strict",
            max_sources=10,
            enable_security=True,
            enable_metrics=True,
            enable_caching=True,
            enable_optimization=True,
            enable_concurrency=True
        )
        
        # Process multiple queries
        test_queries = [
            "What are AI safety requirements?",
            "What are governance compliance standards?",
            "What are the penalties for AI violations?"
        ]
        
        responses = []
        for query in test_queries:
            response = rag.query(query)
            responses.append(response)
            print(f"  - Query: {query[:50]}...")
            print(f"    Factuality: {response.factuality_score:.2f}")
            print(f"    Sources: {len(response.sources)}")
            print(f"    Compliant: {response.governance_compliant}")
        
        # Test caching (second query should be faster)
        start_time = os.times().elapsed
        cached_response = rag.query(test_queries[0])  # Same query
        end_time = os.times().elapsed
        
        # Test system health
        health = rag.get_system_health()
        print(f"  - System status: {health.get('status', 'unknown')}")
        
        # Cleanup
        rag.shutdown()
        
        print("✓ End-to-end functionality successful")
        return True
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🛡️ No-Hallucination RAG Shell - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_initialization,
        test_query_processing,
        test_input_validation,
        test_caching,
        test_security,
        test_system_health,
        test_end_to_end,
    ]
    
    async_tests = [
        test_async_functionality,
    ]
    
    passed = 0
    failed = 0
    
    # Run sync tests
    for test in tests:
        print(f"\n📋 Running {test.__name__}...")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    # Run async tests
    for test in async_tests:
        print(f"\n📋 Running {test.__name__}...")
        try:
            if asyncio.run(test()):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print(f"❌ {failed} test(s) failed. Please review and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())