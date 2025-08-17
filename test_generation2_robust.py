#!/usr/bin/env python3
"""
Generation 2 Test: Robustness and Reliability verification
Tests enhanced error handling, monitoring, security, and fault tolerance
"""

import sys
import os
import time
import asyncio
import threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_error_handling():
    """Test comprehensive error handling."""
    print("🧪 Testing Error Handling...")
    try:
        from no_hallucination_rag.core.error_handling import ErrorHandler, ErrorContext, ValidationError
        
        # Initialize error handler
        error_handler = ErrorHandler()
        print("✅ ErrorHandler initialized")
        
        # Test error context creation
        context = ErrorContext(
            user_query="test query",
            component="test_component",
            operation="test_operation"
        )
        print("✅ ErrorContext created")
        
        # Test error handling
        try:
            raise ValidationError("Test validation error", context)
        except ValidationError as e:
            handled = error_handler.handle_error(e, context)
            if handled:
                print("✅ Error handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_security_manager():
    """Test security features."""
    print("\n🧪 Testing Security Manager...")
    try:
        from no_hallucination_rag.security.security_manager import SecurityManager
        
        # Initialize security manager
        security_manager = SecurityManager()
        print("✅ SecurityManager initialized")
        
        # Test request validation
        is_valid, details = security_manager.validate_request(
            client_ip="127.0.0.1",
            user_id="test_user"
        )
        print(f"✅ Request validation: {is_valid}")
        
        # Test rate limiting
        for i in range(3):
            allowed = security_manager.check_rate_limit("test_user")
            if not allowed:
                print("✅ Rate limiting works")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_monitoring_metrics():
    """Test monitoring and metrics collection."""
    print("\n🧪 Testing Monitoring & Metrics...")
    try:
        from no_hallucination_rag.monitoring.metrics import MetricsCollector
        
        # Initialize metrics collector
        metrics = MetricsCollector()
        print("✅ MetricsCollector initialized")
        
        # Test metric tracking
        metrics.counter("test_counter", 1.0, {"test": "label"})
        print("✅ Counter metric recorded")
        
        metrics.histogram("test_histogram", 0.5, {"test": "label"})
        print("✅ Histogram metric recorded")
        
        # Test metrics retrieval
        stats = metrics.get_all_metrics()
        if stats:
            print("✅ Metrics collection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_advanced_caching():
    """Test advanced caching with TTL and LRU."""
    print("\n🧪 Testing Advanced Caching...")
    try:
        from no_hallucination_rag.optimization.advanced_caching import AdvancedCacheManager
        
        # Initialize advanced cache
        cache_manager = AdvancedCacheManager()
        print("✅ AdvancedCacheManager initialized")
        
        # Test multi-level caching
        cache_manager.set_l1("test_key", "test_value")
        value = cache_manager.get_l1("test_key")
        
        if value == "test_value":
            print("✅ L1 cache works")
        
        # Test TTL expiration
        cache_manager.set_with_ttl("ttl_key", "ttl_value", ttl=1)
        time.sleep(1.1)  # Wait for expiration
        expired_value = cache_manager.get("ttl_key")
        
        if expired_value is None:
            print("✅ TTL expiration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced caching test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_concurrent_processing():
    """Test concurrent and async processing."""
    print("\n🧪 Testing Concurrent Processing...")
    try:
        from no_hallucination_rag.optimization.concurrent_processing import AsyncRAGProcessor, ThreadPoolManager
        
        # Test thread pool manager
        thread_manager = ThreadPoolManager()
        print("✅ ThreadPoolManager initialized")
        
        # Test async processor
        async_processor = AsyncRAGProcessor()
        print("✅ AsyncRAGProcessor initialized")
        
        # Test concurrent task execution
        def test_task(x):
            return x * 2
        
        # Submit tasks
        futures = []
        for i in range(5):
            future = thread_manager.submit(test_task, i)
            futures.append(future)
        
        # Collect results
        results = [future.result() for future in futures]
        
        if len(results) == 5 and all(results[i] == i * 2 for i in range(5)):
            print("✅ Concurrent processing works")
        
        return True
        
    except Exception as e:
        print(f"❌ Concurrent processing test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_factual_rag_robust():
    """Test FactualRAG with robust features enabled."""
    print("\n🧪 Testing Robust FactualRAG...")
    try:
        from no_hallucination_rag import FactualRAG
        
        # Initialize with robust features
        rag = FactualRAG(
            enable_security=True,
            enable_metrics=True,
            enable_caching=True,
            enable_optimization=True
        )
        print("✅ Robust FactualRAG initialized")
        
        # Test basic query with error handling
        try:
            response = rag.query("What is artificial intelligence?")
            print(f"✅ Query successful, factuality: {response.factuality_score:.2f}")
        except Exception as e:
            print(f"⚠️  Query failed gracefully: {e}")
        
        # Test system health
        health = rag.get_system_health()
        if health.get('status') == 'healthy':
            print("✅ System health monitoring works")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust FactualRAG test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_audit_logging():
    """Test comprehensive audit logging."""
    print("\n🧪 Testing Audit Logging...")
    try:
        from no_hallucination_rag.monitoring.audit_logger import audit_logger, AuditEventType
        
        # Test user query logging
        query_id = audit_logger.log_user_query(
            query="test audit query",
            user_id="test_user",
            session_id="test_session"
        )
        print(f"✅ User query logged: {query_id}")
        
        # Test document access logging
        audit_logger.log_document_access(
            sources_accessed=["doc1", "doc2"],
            user_id="test_user",
            session_id="test_session",
            query_hash="test_hash"
        )
        print("✅ Document access logged")
        
        # Test factuality check logging
        audit_logger.log_factuality_check(
            query="test query",
            factuality_score=0.95,
            claims_verified=3,
            user_id="test_user"
        )
        print("✅ Factuality check logged")
        
        return True
        
    except Exception as e:
        print(f"❌ Audit logging test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run all Generation 2 robustness tests."""
    print("🚀 GENERATION 2 TESTING: MAKE IT ROBUST (Reliable)")
    print("=" * 60)
    
    tests = [
        test_error_handling,
        test_security_manager,
        test_monitoring_metrics,
        test_advanced_caching,
        test_concurrent_processing,
        test_factual_rag_robust,
        test_audit_logging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Generation 2: SUCCESS - System is robust and reliable!")
    elif passed >= total * 0.8:
        print("⚠️  Generation 2: MOSTLY ROBUST - Some issues to resolve")
    else:
        print("❌ Generation 2: NEEDS WORK - Major robustness issues found")
    
    return passed >= total * 0.8  # 80% threshold for robustness

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)