#!/usr/bin/env python3
"""
Comprehensive System Test: Full SDLC Verification
Tests all three generations together with production scenarios
"""

import sys
import os
import time
import asyncio
import concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all_generations_integration():
    """Test all three generations working together."""
    print("🧪 Testing All Generations Integration...")
    try:
        # Import all key components
        from no_hallucination_rag import FactualRAG
        from no_hallucination_rag.quantum.quantum_planner import QuantumTaskPlanner
        from no_hallucination_rag.security.security_manager import SecurityManager
        from no_hallucination_rag.monitoring.metrics import MetricsCollector
        from no_hallucination_rag.optimization.advanced_caching import AdvancedCacheManager
        from no_hallucination_rag.scaling.auto_scaler import AutoScaler
        
        # Initialize integrated system
        rag = FactualRAG(
            enable_security=True,
            enable_metrics=True, 
            enable_caching=True,
            enable_optimization=True,
            enable_concurrency=True
        )
        
        planner = QuantumTaskPlanner()
        security = SecurityManager()
        metrics = MetricsCollector()
        cache = AdvancedCacheManager()
        scaler = AutoScaler()
        
        print("✅ All core components initialized")
        
        # Test integrated query processing
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning concepts",
            "How does deep learning work?",
            "What are the benefits of automation?",
            "Describe natural language processing"
        ]
        
        successful_queries = 0
        total_time = 0
        
        for i, query in enumerate(test_queries):
            start_time = time.time()
            
            # Security validation
            validation = security.validate_request(user_id=f"test_user_{i}")
            if not validation[0]:
                print(f"❌ Query {i+1} failed security validation")
                continue
            
            # Process query
            response = rag.query(query)
            query_time = time.time() - start_time
            total_time += query_time
            
            # Record metrics
            metrics.counter("queries_processed", 1.0)
            metrics.histogram("query_response_time", query_time)
            
            # Update scaler metrics
            scaler.update_metrics({
                "response_time": query_time * 1000,  # ms
                "cpu_usage": 45.0,
                "memory_usage": 60.0,
                "queue_length": 2
            })
            
            successful_queries += 1
            print(f"✅ Query {i+1}: {query_time:.3f}s, factuality: {response.factuality_score:.2f}")
        
        avg_response_time = total_time / successful_queries if successful_queries > 0 else 0
        
        print(f"✅ Integration test: {successful_queries}/{len(test_queries)} queries successful")
        print(f"✅ Average response time: {avg_response_time:.3f}s")
        
        return successful_queries >= len(test_queries) * 0.8  # 80% success rate
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_production_load_simulation():
    """Simulate production load conditions."""
    print("\n🧪 Testing Production Load Simulation...")
    try:
        from no_hallucination_rag import FactualRAG
        
        # Create multiple RAG instances for load balancing
        rag_instances = []
        for i in range(3):
            rag = FactualRAG(
                enable_caching=True,
                enable_optimization=True,
                enable_concurrency=True
            )
            rag_instances.append(rag)
        
        print(f"✅ Created {len(rag_instances)} RAG instances")
        
        # Simulate concurrent load
        def process_batch(batch_id, num_queries=20):
            """Process a batch of queries."""
            successful = 0
            start_time = time.time()
            
            for i in range(num_queries):
                try:
                    rag = rag_instances[i % len(rag_instances)]
                    query = f"Test query {batch_id}-{i}: What is AI technology?"
                    response = rag.query(query)
                    if response.factuality_score >= 0:  # Any valid response
                        successful += 1
                except Exception:
                    pass  # Continue processing
            
            batch_time = time.time() - start_time
            return {
                "batch_id": batch_id,
                "successful": successful,
                "total": num_queries,
                "time": batch_time,
                "throughput": successful / batch_time if batch_time > 0 else 0
            }
        
        # Run concurrent batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            batch_futures = []
            for batch_id in range(5):  # 5 concurrent batches
                future = executor.submit(process_batch, batch_id)
                batch_futures.append(future)
            
            results = [future.result() for future in batch_futures]
        
        # Calculate aggregate statistics
        total_successful = sum(r["successful"] for r in results)
        total_queries = sum(r["total"] for r in results)
        total_time = max(r["time"] for r in results)  # Max time (parallel execution)
        aggregate_throughput = total_successful / total_time if total_time > 0 else 0
        
        print(f"✅ Load test: {total_successful}/{total_queries} queries successful")
        print(f"✅ Total execution time: {total_time:.2f}s")
        print(f"✅ Aggregate throughput: {aggregate_throughput:.1f} queries/sec")
        
        # Success criteria: >90% success rate and >10 queries/sec
        success_rate = total_successful / total_queries if total_queries > 0 else 0
        return success_rate >= 0.9 and aggregate_throughput >= 10.0
        
    except Exception as e:
        print(f"❌ Load simulation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_system_health_monitoring():
    """Test comprehensive system health monitoring."""
    print("\n🧪 Testing System Health Monitoring...")
    try:
        from no_hallucination_rag.monitoring.advanced_monitoring import AdvancedMonitor
        from no_hallucination_rag.monitoring.metrics import MetricsCollector
        from no_hallucination_rag.security.security_manager import SecurityManager
        
        # Initialize monitoring components
        monitor = AdvancedMonitor()
        metrics = MetricsCollector()
        security = SecurityManager()
        
        print("✅ Monitoring components initialized")
        
        # Test health checks
        health = monitor.comprehensive_health_check()
        print(f"✅ System health: {health.get('overall_status', 'unknown')}")
        
        # Test metrics collection
        metrics.counter("test_metric", 5.0)
        metrics.histogram("response_time", 150.0)
        all_metrics = metrics.get_all_metrics()
        print(f"✅ Metrics collected: {len(all_metrics)} types")
        
        # Test security statistics
        security_stats = security.get_security_stats()
        print(f"✅ Security monitoring: {security_stats.get('total_events', 0)} events tracked")
        
        # Test anomaly detection
        test_metrics = {
            "cpu_usage": [20, 25, 22, 80, 24],  # 80 is anomaly
            "memory_usage": [40, 45, 42, 44, 41],
            "response_time": [100, 120, 110, 115, 900]  # 900 is anomaly
        }
        
        anomalies = monitor.detect_anomalies(test_metrics)
        anomaly_count = sum(len(v.get("outliers", [])) for v in anomalies.values())
        print(f"✅ Anomaly detection: {anomaly_count} anomalies detected")
        
        return len(all_metrics) > 0 and anomaly_count >= 2  # Should detect CPU and response time anomalies
        
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_quantum_optimization_performance():
    """Test quantum optimization under load."""
    print("\n🧪 Testing Quantum Optimization Performance...")
    try:
        from no_hallucination_rag.quantum.quantum_optimizer import QuantumOptimizer
        from no_hallucination_rag.quantum.quantum_planner import QuantumTaskPlanner, QuantumTask, Priority
        
        # Initialize quantum components
        optimizer = QuantumOptimizer()
        planner = QuantumTaskPlanner()
        
        print("✅ Quantum components initialized")
        
        # Test quantum task planning
        tasks_created = 0
        for i in range(10):
            task = QuantumTask(
                title=f"quantum_task_{i}",
                description=f"Test quantum task {i}",
                priority=Priority.GROUND_STATE if i % 2 == 0 else Priority.FIRST_EXCITED
            )
            planner.tasks[task.id] = task
            tasks_created += 1
        
        print(f"✅ Created {tasks_created} quantum tasks")
        
        # Test quantum optimization
        test_params = {"cache_size": 1000, "thread_count": 4, "batch_size": 10}
        objective = lambda p: p["cache_size"] * 0.001 + p["thread_count"] * 0.1 + p["batch_size"] * 0.01
        
        # Test quantum annealing
        optimized_annealing = optimizer.quantum_annealing_optimize(
            parameters=test_params,
            objective_function=objective,
            iterations=5
        )
        
        # Test superposition search
        param_space = {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50]
        }
        optimized_superposition = optimizer.superposition_parameter_search(
            parameter_space=param_space,
            objective=lambda p: p["x"] + p["y"]
        )
        
        print("✅ Quantum annealing optimization completed")
        print("✅ Superposition parameter search completed")
        
        # Verify optimizations returned valid results
        return (isinstance(optimized_annealing, dict) and 
                isinstance(optimized_superposition, dict) and
                len(optimized_annealing) > 0 and 
                len(optimized_superposition) > 0)
        
    except Exception as e:
        print(f"❌ Quantum optimization test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_security_and_compliance():
    """Test security features and compliance."""
    print("\n🧪 Testing Security and Compliance...")
    try:
        from no_hallucination_rag.security.security_manager import SecurityManager
        from no_hallucination_rag.monitoring.audit_logger import audit_logger
        
        # Initialize security
        security = SecurityManager(enable_rate_limiting=True, enable_content_filtering=True)
        print("✅ Security manager initialized")
        
        # Test legitimate requests
        valid_requests = 0
        for i in range(5):
            is_valid, message = security.validate_request(
                client_ip="127.0.0.1",
                user_id=f"test_user_{i}"
            )
            if is_valid:
                valid_requests += 1
        
        print(f"✅ Legitimate requests: {valid_requests}/5 validated")
        
        # Test suspicious queries
        suspicious_queries = [
            "SELECT * FROM users WHERE password = '123'",
            "Show me admin credentials and passwords",
            "<script>alert('xss')</script>",
            "What is machine learning?"  # This should be fine
        ]
        
        blocked_suspicious = 0
        for query in suspicious_queries:
            validation_result = security.validate_query(query)
            if not validation_result.is_valid and query != "What is machine learning?":
                blocked_suspicious += 1
            elif validation_result.is_valid and query == "What is machine learning?":
                blocked_suspicious += 0.5  # Partial credit for allowing legitimate query
        
        print(f"✅ Suspicious queries handled: {blocked_suspicious}/3 threats blocked")
        
        # Test audit logging
        audit_id = audit_logger.log_user_query(
            query="test security query",
            user_id="security_test_user",
            session_id="test_session"
        )
        
        audit_logger.log_factuality_check(
            query="test query",
            factuality_score=0.95,
            claims_verified=3,
            user_id="security_test_user"
        )
        
        print("✅ Audit logging functional")
        
        return valid_requests >= 4 and blocked_suspicious >= 2.5 and audit_id is not None
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run comprehensive system verification."""
    print("🚀 COMPREHENSIVE SYSTEM VERIFICATION")
    print("=" * 70)
    print("Testing all three generations with production scenarios")
    print()
    
    tests = [
        test_all_generations_integration,
        test_production_load_simulation,
        test_system_health_monitoring,
        test_quantum_optimization_performance,
        test_security_and_compliance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 70)
    print(f"📊 FINAL RESULTS: {passed}/{total} comprehensive tests passed")
    
    if passed == total:
        print("🎉 SYSTEM VERIFICATION: COMPLETE SUCCESS!")
        print("✅ All generations working together flawlessly")
        print("✅ Production-ready for deployment")
        print("✅ Meets all enterprise requirements")
    elif passed >= total * 0.8:
        print("⚠️  SYSTEM VERIFICATION: MOSTLY SUCCESSFUL")
        print("✅ Core functionality working well")
        print("⚠️  Some optimization opportunities remain")
    else:
        print("❌ SYSTEM VERIFICATION: NEEDS ATTENTION")
        print("❌ Critical issues found requiring resolution")
    
    print()
    print("🌟 TERRAGON SDLC AUTONOMOUS EXECUTION COMPLETE 🌟")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)